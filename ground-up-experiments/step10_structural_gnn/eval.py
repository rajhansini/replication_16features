"""Step 10 — Eval: ratio2 for V*(s) GNN with structural features.

Action selection derived from V* predictions (correct ADP formulation):
    Q_hat(s,i) = r_test(i,s) + Σ_g P(g|s,i) * V_hat(s')
    action     = argmax_i Q_hat(s,i) over untested persons

Logs comparison against step9 (Q* per person) for every config.
"""
from __future__ import annotations

import json
import pickle
import sys
import time
from collections import deque
from datetime import datetime
from pathlib import Path

import numpy as np
import torch

HERE         = Path(__file__).resolve().parent
EXPERIMENTS  = HERE.parent
PROJECT_ROOT = EXPERIMENTS.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(EXPERIMENTS))

from shared.model import PedigreeGNN_Vstar
from shared.data_gen import FAMILY_CASES
from genetic_dp.utils.pedigree_generator import generate_deterministic_pedigree
from genetic_dp.exact_dp.utils import GENOTYPE_STATES, lift_tuple_posteriors_to_genes
from genetic_dp.models.belief  import InferenceResult
from genetic_dp.models.reward  import r_reward, r_reward_test

GENES         = ("GeneA", "GeneB", "GeneC")
STRUCT_DIM    = 3
NODE_FEAT_DIM = 3 * len(GENES) + 1 + STRUCT_DIM   # 13
COST_DIM      = 3 * len(GENES) + 2                 # 11

CACHE_DIR   = HERE.parent / "step9_gnn_3gene" / "results" / "cache"
RESULTS_DIR = HERE / "results"

ALLELE_REGIMES = ["LowHigh", "MediumEven", "LowLow", "HighHigh", "MixedA", "MixedB"]

TRAIN_FAMILIES = ["Trio", "Nuclear"]
TEST_FAMILIES  = ["ThreeGeneration"]   # unseen bigger family — the scaling benchmark

ALL_KEYS = [
    f"{family}_{regime}_{preset}_3gene"
    for family in TRAIN_FAMILIES + TEST_FAMILIES
    for regime  in ALLELE_REGIMES
    for preset  in ["Base", "Aggressive"]
]

# Every ThreeGeneration config is unseen — never trained on this topology
UNSEEN_KEYS = {
    f"ThreeGeneration_{regime}_{preset}_3gene"
    for regime in ALLELE_REGIMES
    for preset in ["Base", "Aggressive"]
}

TRAIN_KEYS = [
    f"{family}_{regime}_{preset}_3gene"
    for family in TRAIN_FAMILIES
    for regime  in ALLELE_REGIMES
    for preset  in ["Base", "Aggressive"]
]

TEST_KEYS = [
    f"{family}_{regime}_{preset}_3gene"
    for family in TEST_FAMILIES
    for regime  in ALLELE_REGIMES
    for preset  in ["Base", "Aggressive"]
]


# ─── structural features (same as run.py) ────────────────────────────────────

def compute_structural_features(pedigree, individuals):
    idx = {p: i for i, p in enumerate(individuals)}
    n   = len(individuals)
    n_parents_arr  = np.zeros(n, dtype=np.float32)
    n_children_arr = np.zeros(n, dtype=np.float32)
    depth_arr      = np.zeros(n, dtype=np.float32)
    children_map: dict = {p: [] for p in individuals}
    for child in pedigree.get_offspring():
        if child not in idx: continue
        parents = [p for p in pedigree.get_parents(child) if p in idx]
        n_parents_arr[idx[child]] = float(len(parents))
        for p in parents:
            n_children_arr[idx[p]] += 1.0
            children_map[p].append(child)
    # Topological max-depth: longest path from any root → each person.
    # Fixes ThreeGeneration Child getting depth=1 (via Mother) instead of
    # the genealogically correct depth=2 (via Grandfather→Father→Child).
    in_deg = {p: int(n_parents_arr[idx[p]]) for p in individuals}
    queue  = deque([p for p in individuals if in_deg[p] == 0])
    while queue:
        person = queue.popleft()
        for child in children_map[person]:
            depth_arr[idx[child]] = max(depth_arr[idx[child]], depth_arr[idx[person]] + 1.0)
            in_deg[child] -= 1
            if in_deg[child] == 0:
                queue.append(child)
    return np.stack([n_parents_arr, n_children_arr, depth_arr], axis=-1)


def config_to_cost_vec(config, genes):
    vec = []
    for g in genes:
        vec.append(min(config.a_gene[g].values(), key=abs))
        vec.append(min(config.b_gene[g].values(), key=abs))
        vec.append(list(config.delta_gene[g].values())[0])
    vec.append(config.fixed_cost)
    vec.append(config.variable_cost)
    return np.array(vec, dtype=np.float32)


def build_edge_index(family_label, individuals):
    pedigree = generate_deterministic_pedigree(FAMILY_CASES[family_label])
    idx = {p: i for i, p in enumerate(individuals)}
    srcs, dsts = [], []
    for child in pedigree.get_offspring():
        for parent in pedigree.get_parents(child):
            if parent in idx and child in idx:
                srcs.append(idx[parent]); dsts.append(idx[child])
    return np.array([srcs, dsts], dtype=np.int64) if srcs else np.zeros((2, 0), dtype=np.int64)


def extract_node_features(ds, individuals, struct_feats):
    X            = ds["X"]
    states       = ds["states"]
    n_gene_feats = 3 * len(ds.get("genes", GENES))
    n_people     = len(individuals)
    person_idx   = {p: i for i, p in enumerate(individuals)}
    N            = len(states)
    nf           = X.reshape(N, n_people, n_gene_feats).copy()
    tested_col   = np.zeros((N, n_people, 1), dtype=np.float32)
    for i, state in enumerate(states):
        for person, _ in state:
            tested_col[i, person_idx[person], 0] = 1.0
    struct_broadcast = np.tile(struct_feats[np.newaxis, :, :], (N, 1, 1))
    return np.concatenate([nf, tested_col, struct_broadcast], axis=-1)


# ─── batched V* inference ─────────────────────────────────────────────────────

def precompute_vstar(model, ds, individuals, edge_index, struct_feats, device,
                     batch_size=4096):
    """
    One batched forward pass → dict {state: V_hat(s)}.
    V_hat is used to derive Q_hat(s,i) at action selection time.
    """
    model.eval()
    nf_all   = extract_node_features(ds, individuals, struct_feats)
    cost_vec = config_to_cost_vec(ds["config"], GENES)
    states   = ds["states"]
    ei_t     = torch.LongTensor(edge_index).to(device)
    N        = len(states)
    v_all    = np.empty(N, dtype=np.float64)
    gf_all   = np.tile(cost_vec, (N, 1))

    with torch.no_grad():
        for start in range(0, N, batch_size):
            end  = min(start + batch_size, N)
            nf_b = torch.FloatTensor(nf_all[start:end]).to(device)
            gf_b = torch.FloatTensor(gf_all[start:end]).to(device)
            v_all[start:end] = model.forward_batch(nf_b, ei_t, gf_b).cpu().numpy()

    return {state: float(v_all[i]) for i, state in enumerate(states)}


# ─── ratio2 via V*-derived actions ───────────────────────────────────────────

def compute_ratio2(v_hat, ds, log=print):
    """
    Greedy rollout using V*-derived action selection:
        Q_hat(s,i) = r_test(i,s) + Σ_g P(g|s,i) * V_hat(s')
        action     = argmax_i Q_hat(s,i)
        stop if    max Q_hat(s,i) <= v_stop(s)

    This is the correct ADP-style formulation.
    """
    belief      = ds["belief"]
    individuals = ds["individuals"]
    config      = ds["config"]
    genes       = ds.get("genes", GENES)

    def _get_entry(state):
        entry = belief[state]
        if isinstance(entry, InferenceResult):
            return entry.get_per_gene_probs(), entry.get_tuple_pmfs()
        return lift_tuple_posteriors_to_genes(entry, genes, GENOTYPE_STATES), None

    def _stop_val(per_gene, tested):
        return float(sum(
            r_reward(k, None, config.a, config.b, config.c, config.delta,
                     per_gene_probs=per_gene, a_gene=config.a_gene,
                     b_gene=config.b_gene, c_gene=config.c_gene,
                     delta_gene=config.delta_gene)
            for k in individuals if k not in tested
        ))

    def _test_r(i, per_gene):
        return float(r_reward_test(
            i, None, config.a, config.b, config.c, config.delta,
            config.fixed_cost, config.variable_cost,
            per_gene_probs=per_gene, a_gene=config.a_gene,
            c_gene=config.c_gene, delta_gene=config.delta_gene,
        ))

    def _pmf(tuple_pmfs, per_gene, person):
        # tuple_pmfs: {person: {(g_A,g_B,g_C): prob}} from InferenceResult
        # per_gene fallback was broken: per_gene has structure {gene:{person:dist}},
        # so per_gene.get(person, {}) always returns {} (person is not a top-level key).
        # For 3-gene inference we always have InferenceResult → tuple_pmfs is never None.
        if tuple_pmfs is None:
            raise RuntimeError(
                "tuple_pmfs is None — multi-gene Q_hat requires InferenceResult in belief map"
            )
        return tuple_pmfs.get(person, {})

    def _q_hat(state, person, per_gene, tuple_pmfs):
        """Q_hat(s,i) = r_test(i,s) + Σ_g P(g|s,i) * V_hat(s')."""
        r_i = _test_r(person, per_gene)
        pmf = _pmf(tuple_pmfs, per_gene, person)
        exp_v = sum(
            prob * v_hat.get(frozenset(state | {(person, g)}), 0.0)
            for g, prob in pmf.items()
            if prob > 1e-12 and frozenset(state | {(person, g)}) in belief
        )
        return r_i + exp_v

    true_memo: dict = {}

    def value_at(state) -> float:
        if state in true_memo:
            return true_memo[state]
        per_gene, tuple_pmfs = _get_entry(state)
        tested  = {i for i, _ in state}
        v_stop  = _stop_val(per_gene, tested)

        if len(tested) == len(individuals):
            true_memo[state] = 0.0
            return 0.0

        untested = [i for i in individuals if i not in tested]

        # Derive action from V* predictions via Bellman equation
        q_hats      = {i: _q_hat(state, i, per_gene, tuple_pmfs) for i in untested}
        best_person = max(untested, key=lambda i: q_hats[i])
        best_q      = q_hats[best_person]

        if best_q <= v_stop:
            true_memo[state] = v_stop
            return v_stop

        r_best   = _test_r(best_person, per_gene)
        pmf_best = _pmf(tuple_pmfs, per_gene, best_person)
        exp_true = sum(
            prob_g * value_at(frozenset(state | {(best_person, g)}))
            for g, prob_g in pmf_best.items()
            if prob_g > 1e-12 and frozenset(state | {(best_person, g)}) in belief
        )
        result = r_best + exp_true
        true_memo[state] = result
        return result

    L      = value_at(frozenset())
    V_root = ds["V_root"]
    V_stop = ds["V_stop_root"]
    denom  = V_root - V_stop
    ratio2 = (V_root - L) / denom if abs(denom) > 1e-12 else 0.0
    return float(ratio2), float(L)


# ─── main ────────────────────────────────────────────────────────────────────

def main(device="cpu", fresh=False):
    model_path   = RESULTS_DIR / "gnn_structural_model.pt"
    partial_path = RESULTS_DIR / "partial_eval.json"
    RESULTS_DIR.mkdir(exist_ok=True)

    log_f = open(RESULTS_DIR / "eval.log", "a")
    log_f.write(f"\n{'='*60}\n[EVAL] {datetime.now().isoformat()}\n{'='*60}\n")
    def log(msg=""):
        print(msg, flush=True); log_f.write(msg + "\n"); log_f.flush()

    if not model_path.exists():
        raise FileNotFoundError(f"No model at {model_path}. Run run.py first.")

    model = PedigreeGNN_Vstar(node_feat_dim=NODE_FEAT_DIM, hidden_dim=32,
                               n_rounds=2, global_feat_dim=COST_DIM)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device).eval()
    log(f"Loaded PedigreeGNN_Vstar  device={device}  node_feat_dim={NODE_FEAT_DIM}")
    log(f"Action selection: Q_hat(s,i) = r_test(i,s) + E[V_hat(s')]  (derived from V*)")
    log(f"TRAIN families: {TRAIN_FAMILIES}  — seen during training")
    log(f"TEST  families: {TEST_FAMILIES}   — UNSEEN topology, scaling benchmark")

    # Load step9 results for comparison
    step9_path = HERE.parent / "step9_gnn_3gene" / "results" / "results.json"
    step9_res  = json.loads(step9_path.read_text()).get("families", {}) if step9_path.exists() else {}
    log(f"step9 results loaded: {len(step9_res)} configs for comparison")

    # Precompute structural features per family
    struct_cache: dict = {}
    for family in TRAIN_FAMILIES + TEST_FAMILIES:
        with open(CACHE_DIR / f"{family}_LowHigh_Base_3gene.pkl", "rb") as f:
            sds = pickle.load(f)
        struct_cache[family] = compute_structural_features(
            generate_deterministic_pedigree(FAMILY_CASES[family]), sds["individuals"]
        )
        del sds

    all_results = {} if fresh else (
        json.loads(partial_path.read_text()) if partial_path.exists() else {}
    )
    pending = [k for k in ALL_KEYS if k not in all_results]
    log(f"Pending: {len(pending)}/{len(ALL_KEYS)}")

    edge_index_cache: dict = {}

    for key in pending:
        split  = "TEST_UNSEEN" if key in UNSEEN_KEYS else "TRAIN"
        family = key.split("_")[0]
        log(f"\n[{split}] {key}")

        pkl = CACHE_DIR / f"{key}.pkl"
        if not pkl.exists():
            log(f"  MISSING pkl — skipping"); continue

        t0 = time.time()
        with open(pkl, "rb") as f:
            ds = pickle.load(f)

        individuals = ds["individuals"]
        sf = struct_cache[family]

        if family not in edge_index_cache:
            edge_index_cache[family] = build_edge_index(family, individuals)
            log(f"  [{family}] {len(individuals)} people  {edge_index_cache[family].shape[1]} edges")
        edge_index = edge_index_cache[family]

        log(f"  Precomputing V_hat(s) for {len(ds['states']):,} states...")
        t1 = time.time()
        v_hat = precompute_vstar(model, ds, individuals, edge_index, sf, device)
        log(f"  V_hat done ({time.time()-t1:.1f}s)  "
            f"range=[{min(v_hat.values()):.4f}, {max(v_hat.values()):.4f}]")
        log(f"  True V_root={ds['V_root']:.6f}  V_hat(root)={v_hat[frozenset()]:.6f}")

        log(f"  Computing ratio2 (V*-derived greedy rollout)...")
        t2 = time.time()
        ratio2, L = compute_ratio2(v_hat, ds, log=log)
        log(f"  ratio2={ratio2:.6f}  L={L:.6f}  ({time.time()-t2:.1f}s)")

        # Compare with step9
        if key in step9_res:
            s9_r2  = step9_res[key]["ratio2"]
            delta  = ratio2 - s9_r2
            sign   = "WORSE" if delta > 1e-4 else ("BETTER" if delta < -1e-4 else "SAME")
            log(f"  step9 ratio2={s9_r2:.6f}  delta={delta:+.6f}  [{sign}]")

        # Rollout trace with V*-derived actions
        log(f"  ROLLOUT TRACE (V*-derived actions):")
        _rbelief = ds["belief"]; _rind = ds["individuals"]; _rcfg = ds["config"]
        _rgenes  = ds.get("genes", GENES)

        def _r_pg(state):
            e = _rbelief[state]
            if isinstance(e, InferenceResult):
                return e.get_per_gene_probs(), e.get_tuple_pmfs()
            return lift_tuple_posteriors_to_genes(e, _rgenes, GENOTYPE_STATES), None

        def _r_vstop(pg, tested):
            return float(sum(r_reward(k, None, _rcfg.a, _rcfg.b, _rcfg.c, _rcfg.delta,
                per_gene_probs=pg, a_gene=_rcfg.a_gene, b_gene=_rcfg.b_gene,
                c_gene=_rcfg.c_gene, delta_gene=_rcfg.delta_gene)
                for k in _rind if k not in tested))

        def _r_q_hat(state, person, pg, tpmfs):
            r_i  = float(r_reward_test(person, None, _rcfg.a, _rcfg.b, _rcfg.c, _rcfg.delta,
                _rcfg.fixed_cost, _rcfg.variable_cost, per_gene_probs=pg,
                a_gene=_rcfg.a_gene, c_gene=_rcfg.c_gene, delta_gene=_rcfg.delta_gene))
            pmf  = tpmfs.get(person, {}) if tpmfs is not None else {}
            ev   = sum(prob * v_hat.get(frozenset(state | {(person, g)}), 0.0)
                       for g, prob in pmf.items()
                       if prob > 1e-12 and frozenset(state | {(person, g)}) in _rbelief)
            return r_i + ev

        _rstate = frozenset(); _rstep = 0
        while len({p for p, _ in _rstate}) < len(_rind) and _rstep < 20:
            _tested = {p for p, _ in _rstate}
            _pg, _tpmfs = _r_pg(_rstate)
            _vs    = _r_vstop(_pg, _tested)
            _untst = [p for p in _rind if p not in _tested]
            _qhats = {p: _r_q_hat(_rstate, p, _pg, _tpmfs) for p in _untst}
            _best  = max(_untst, key=lambda p: _qhats[p])
            _bq    = _qhats[_best]
            log(f"    step {_rstep}: tested={sorted(_tested) or '{}'}")
            log(f"      Q_hat (from V*): { {p: round(v,5) for p, v in _qhats.items()} }")
            log(f"      v_stop={_vs:.5f}  best={_best}  best_Q_hat={_bq:.5f}")
            if _bq <= _vs:
                log(f"      → STOP"); break
            log(f"      → TEST {_best}")
            _pg_p, _tp_p = _r_pg(_rstate)
            _pmf_b = _tp_p.get(_best, {}) if _tp_p is not None else {}
            _mlg   = max(_pmf_b, key=lambda g: _pmf_b[g]) if _pmf_b else None
            if _mlg is None: break
            _rstate = frozenset(_rstate | {(_best, _mlg)})
            _rstep += 1
        log(f"    trace done after {_rstep} steps")

        all_results[key] = {
            "split":        split,
            "ratio2":       ratio2,
            "L":            float(L),
            "V_root":       float(ds["V_root"]),
            "V_stop_root":  float(ds["V_stop_root"]),
            "v_hat_root":   v_hat[frozenset()],
            "step9_ratio2": step9_res.get(key, {}).get("ratio2"),
            "total_sec":    time.time() - t0,
        }
        partial_path.write_text(json.dumps(all_results, indent=2))
        del ds, v_hat

    summary = {
        "model":              "PedigreeGNN_Vstar",
        "formulation":        "V*(s) prediction, action derived via Bellman",
        "node_feat_dim":      NODE_FEAT_DIM,
        "structural_features": ["n_parents", "n_children", "depth"],
        "hidden_dim":         32,
        "n_rounds":           2,
        "n_genes":            len(GENES),
        "genes":              list(GENES),
        "train_keys":         sorted(TRAIN_KEYS),
        "test_keys":          sorted(TEST_KEYS),
        "families":           all_results,
    }
    (RESULTS_DIR / "results.json").write_text(json.dumps(summary, indent=2))

    log(f"\n[DONE] → {RESULTS_DIR / 'results.json'}")
    log(f"\n  {'Key':<50} {'Split':<12} {'s10_ratio2':>10}  {'s9_ratio2':>10}  {'delta':>8}")
    log(f"  {'-'*98}")
    for key, r in all_results.items():
        s9  = r.get("step9_ratio2")
        s9s = f"{s9:.6f}" if s9 is not None else "     N/A"
        ds  = f"{r['ratio2']-s9:+.6f}" if s9 is not None else "     N/A"
        log(f"  {key:<50} {r['split']:<12} {r['ratio2']:>10.6f}  {s9s:>10}  {ds:>8}")

    # Summarise unseen vs seen
    seen   = [(k,r) for k,r in all_results.items() if r["split"] == "TRAIN"]
    unseen = [(k,r) for k,r in all_results.items() if r["split"] == "TEST_UNSEEN"]
    if seen:
        log(f"\n  SEEN    avg ratio2 = {sum(r['ratio2'] for _,r in seen)/len(seen):.4f}  ({len(seen)} configs)")
    if unseen:
        log(f"  UNSEEN  avg ratio2 = {sum(r['ratio2'] for _,r in unseen)/len(unseen):.4f}  ({len(unseen)} configs)  ← ThreeGeneration scaling test")


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--device", default="cpu")
    p.add_argument("--fresh",  action="store_true")
    args = p.parse_args()
    main(device=args.device, fresh=args.fresh)
