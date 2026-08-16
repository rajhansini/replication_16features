"""Step 9 — Eval: ratio2 for the trained 3-gene GNN.

Loads gnn3_model.pt, then for each of the 4 configs:
  1. Batch-infers GNN values for all states in one forward pass (fast).
  2. Runs greedy rollout from root using precomputed values → ratio2.

Saves results to step9_gnn_3gene/results/results.json.
"""
from __future__ import annotations

import json
import pickle
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import torch

HERE         = Path(__file__).resolve().parent
EXPERIMENTS  = HERE.parent
PROJECT_ROOT = EXPERIMENTS.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(EXPERIMENTS))

from shared.model import PedigreeGNN
from shared.data_gen import FAMILY_CASES
from genetic_dp.utils.pedigree_generator import generate_deterministic_pedigree
from genetic_dp.exact_dp.utils import GENOTYPE_STATES, lift_tuple_posteriors_to_genes
from genetic_dp.models.belief  import InferenceResult
from genetic_dp.models.reward  import r_reward, r_reward_test

GENES         = ("GeneA", "GeneB", "GeneC")
NODE_FEAT_DIM = 3 * len(GENES) + 1  # 10
COST_DIM      = 3 * len(GENES) + 2  # 11
CACHE_DIR     = HERE / "results" / "cache"
RESULTS_DIR   = HERE / "results"

FAMILIES = ["Trio", "Nuclear", "ThreeGeneration"]
ALLELE_REGIMES = ["LowHigh", "MediumEven", "LowLow", "HighHigh", "MixedA", "MixedB"]

TEST_KEYS = {
    "ThreeGeneration_MediumEven_Aggressive_3gene",
    "ThreeGeneration_HighHigh_Aggressive_3gene",
    "Nuclear_MediumEven_Aggressive_3gene",
    "Nuclear_HighHigh_Aggressive_3gene",
    "Trio_MediumEven_Aggressive_3gene",
    "Trio_HighHigh_Aggressive_3gene",
}

ALL_KEYS = [
    f"{family}_{regime}_{preset}_3gene"
    for family in FAMILIES
    for regime  in ALLELE_REGIMES
    for preset  in ["Base", "Aggressive"]
]
TRAIN_KEYS = {k for k in ALL_KEYS if k not in TEST_KEYS}


# ─── fast vectorized node feature extraction (same as run.py) ────────────────

def config_to_cost_vec(config, genes) -> np.ndarray:
    """Extract scalar cost params: (a, b, delta) per gene + fixed + variable. dim=11."""
    vec = []
    for g in genes:
        vec.append(min(config.a_gene[g].values(), key=abs))
        vec.append(min(config.b_gene[g].values(), key=abs))
        vec.append(list(config.delta_gene[g].values())[0])
    vec.append(config.fixed_cost)
    vec.append(config.variable_cost)
    return np.array(vec, dtype=np.float32)


def extract_node_features(ds: dict, individuals: list) -> np.ndarray:
    """Returns (N, n_people, 10) — fast, no belief map calls."""
    X      = ds["X"]
    states = ds["states"]
    n_gene_feats = 3 * len(ds.get("genes", GENES))
    n_people     = len(individuals)
    person_idx   = {p: i for i, p in enumerate(individuals)}
    N   = len(states)
    nf  = X.reshape(N, n_people, n_gene_feats).copy()
    tested_col = np.zeros((N, n_people, 1), dtype=np.float32)
    for i, state in enumerate(states):
        for person, _ in state:
            tested_col[i, person_idx[person], 0] = 1.0
    return np.concatenate([nf, tested_col], axis=-1)   # (N, 5, 10)


def build_edge_index(family_label: str, individuals: list) -> np.ndarray:
    pedigree = generate_deterministic_pedigree(FAMILY_CASES[family_label])
    idx      = {p: i for i, p in enumerate(individuals)}
    srcs, dsts = [], []
    for child in pedigree.get_offspring():
        for parent in pedigree.get_parents(child):
            if parent in idx and child in idx:
                srcs.append(idx[parent])
                dsts.append(idx[child])
    return np.array([srcs, dsts], dtype=np.int64) if srcs else np.zeros((2, 0), dtype=np.int64)


# ─── batched GNN inference → state lookup dict ───────────────────────────────

def precompute_gnn_q_values(model: PedigreeGNN, ds: dict, individuals: list,
                            edge_index: np.ndarray, device: str,
                            batch_size: int = 4096) -> dict:
    """
    One batched forward pass → dict {state: q_array}.
    q_array[j] = Q(s, individuals[j]) — predicted value of testing person j.
    Action selection: argmax_j q_array[j] over untested persons.
    """
    model.eval()
    nf_all   = extract_node_features(ds, individuals)
    cost_vec = config_to_cost_vec(ds["config"], GENES)
    states   = ds["states"]
    ei_t     = torch.LongTensor(edge_index).to(device)
    N        = len(states)
    N_people = len(individuals)
    q_all    = np.empty((N, N_people), dtype=np.float64)

    gf_all = np.tile(cost_vec, (N, 1))  # (N, COST_DIM)

    with torch.no_grad():
        for start in range(0, N, batch_size):
            end  = min(start + batch_size, N)
            nf_b = torch.FloatTensor(nf_all[start:end]).to(device)
            gf_b = torch.FloatTensor(gf_all[start:end]).to(device)
            q_all[start:end] = model.forward_batch(nf_b, ei_t, gf_b).cpu().numpy()

    return {state: q_all[i] for i, state in enumerate(states)}


# ─── ratio2 using precomputed Q values ───────────────────────────────────────

def compute_ratio2(gnn_q_vals: dict, ds: dict) -> tuple[float, float]:
    """
    Greedy rollout from root using pre-computed GNN Q values.
    At each state: action = argmax_i Q(s, i) over untested persons.
    If max Q(s,i) <= V_stop(s), stop testing.
    """
    belief      = ds["belief"]
    individuals = ds["individuals"]
    config      = ds["config"]
    genes       = ds.get("genes", GENES)
    pid         = {p: j for j, p in enumerate(individuals)}

    def _get_entry(state):
        entry = belief[state]
        if isinstance(entry, InferenceResult):
            per_gene   = entry.get_per_gene_probs()
            tuple_pmfs = entry.get_tuple_pmfs()
        else:
            per_gene   = lift_tuple_posteriors_to_genes(entry, genes, GENOTYPE_STATES)
            tuple_pmfs = None
        return per_gene, tuple_pmfs

    def _stop_val(per_gene, tested):
        return float(sum(
            r_reward(k, None, config.a, config.b, config.c, config.delta,
                     per_gene_probs=per_gene,
                     a_gene=config.a_gene, b_gene=config.b_gene,
                     c_gene=config.c_gene, delta_gene=config.delta_gene)
            for k in individuals if k not in tested
        ))

    def _test_r(i, per_gene):
        return float(r_reward_test(
            i, None, config.a, config.b, config.c, config.delta,
            config.fixed_cost, config.variable_cost,
            per_gene_probs=per_gene,
            a_gene=config.a_gene, c_gene=config.c_gene, delta_gene=config.delta_gene,
        ))

    def _pmf(tuple_pmfs, per_gene, person):
        if tuple_pmfs is not None:
            return tuple_pmfs.get(person, {})
        return per_gene.get(person, {})

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

        # Pick action using GNN Q values — model explicitly predicts per-person value
        untested = [i for i in individuals if i not in tested]
        q_vec    = gnn_q_vals[state]          # (N_people,) array
        best_person = max(untested, key=lambda i: q_vec[pid[i]])
        best_q      = float(q_vec[pid[best_person]])

        # Stop if model says stopping is better
        if best_q <= v_stop:
            true_memo[state] = v_stop
            return v_stop

        # Execute best action, compute TRUE value recursively
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

def main(device: str = "cpu", fresh: bool = False):
    model_path   = RESULTS_DIR / "gnn3_model.pt"
    partial_path = RESULTS_DIR / "partial_eval.json"
    RESULTS_DIR.mkdir(exist_ok=True)

    log_f = open(RESULTS_DIR / "eval.log", "a")
    log_f.write(f"\n{'='*60}\n[EVAL] {datetime.now().isoformat()}\n{'='*60}\n")
    def log(msg=""):
        print(msg, flush=True); log_f.write(msg + "\n"); log_f.flush()

    if not model_path.exists():
        raise FileNotFoundError(f"No trained model at {model_path}. Run run.py first.")

    model = PedigreeGNN(node_feat_dim=NODE_FEAT_DIM, hidden_dim=32, n_rounds=2, global_feat_dim=COST_DIM)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device).eval()
    log(f"Loaded model from {model_path}  device={device}")

    all_results = {} if fresh else (
        json.loads(partial_path.read_text()) if partial_path.exists() else {}
    )
    pending = [k for k in ALL_KEYS if k not in all_results]
    log(f"Pending: {len(pending)}/{len(ALL_KEYS)}")

    edge_index_cache: dict = {}   # family_label → edge_index

    for key in pending:
        split  = "TEST" if key in TEST_KEYS else "TRAIN"
        family = key.split("_")[0]
        log(f"\n[{split}] {key}")
        pkl = CACHE_DIR / f"{key}.pkl"
        if not pkl.exists():
            log(f"  MISSING pkl — skipping"); continue

        t0 = time.time()
        with open(pkl, "rb") as f:
            ds = pickle.load(f)

        individuals = ds["individuals"]
        if family not in edge_index_cache:
            edge_index_cache[family] = build_edge_index(family, individuals)
            log(f"  [{family}] pedigree: {len(individuals)} people, {edge_index_cache[family].shape[1]} edges")
        edge_index = edge_index_cache[family]

        log(f"  batch GNN Q-value inference on {len(ds['states']):,} states...")
        t1 = time.time()
        gnn_q_vals = precompute_gnn_q_values(model, ds, individuals, edge_index, device)
        log(f"  inference done ({time.time()-t1:.1f}s)")

        log(f"  computing ratio2 (Q-value greedy rollout)...")
        t2 = time.time()
        ratio2, L = compute_ratio2(gnn_q_vals, ds)
        log(f"  ratio2={ratio2:.6f}  L={L:.6f}  ({time.time()-t2:.1f}s)")

        # --- rollout trace (log every decision from root) ---
        log(f"  ROLLOUT TRACE from root:")
        _rbelief = ds["belief"]; _rind = ds["individuals"]; _rcfg = ds["config"]
        _rgenes  = ds.get("genes", GENES); _rpid = {p: j for j, p in enumerate(_rind)}
        def _r_pg(state):
            e = _rbelief[state]
            if isinstance(e, InferenceResult):
                return e.get_per_gene_probs(), e.get_tuple_pmfs()
            return lift_tuple_posteriors_to_genes(e, _rgenes, GENOTYPE_STATES), None
        def _r_vstop(pg, tested):
            from genetic_dp.models.reward import r_reward as _rr
            return float(sum(_rr(k, None, _rcfg.a, _rcfg.b, _rcfg.c, _rcfg.delta,
                per_gene_probs=pg, a_gene=_rcfg.a_gene, b_gene=_rcfg.b_gene,
                c_gene=_rcfg.c_gene, delta_gene=_rcfg.delta_gene)
                for k in _rind if k not in tested))
        _rstate = frozenset(); _rstep = 0
        while len({p for p, _ in _rstate}) < len(_rind) and _rstep < 20:
            _tested = {p for p, _ in _rstate}
            _pg, _  = _r_pg(_rstate)
            _vs     = _r_vstop(_pg, _tested)
            _untst  = [p for p in _rind if p not in _tested]
            _qvec   = gnn_q_vals[_rstate]
            _qs     = {p: float(_qvec[_rpid[p]]) for p in _untst}
            _best   = max(_untst, key=lambda p: _qs[p])
            _bq     = _qs[_best]
            log(f"    step {_rstep}: tested={sorted(_tested) or '{}'}")
            log(f"      Q values: { {p: round(v,5) for p, v in _qs.items()} }")
            log(f"      v_stop={_vs:.5f}  best={_best}  best_Q={_bq:.5f}")
            if _bq <= _vs:
                log(f"      → STOP (best_Q={_bq:.5f} <= v_stop={_vs:.5f})"); break
            log(f"      → TEST {_best}")
            # advance state using expected outcome (just pick most likely outcome)
            _pg_p, _tpmfs_p = _r_pg(_rstate)
            _pmf_best = (_tpmfs_p or {}).get(_best, {})
            if not _pmf_best:
                _pmf_best = _pg_p.get(_best, {})
            _most_likely_g = max(_pmf_best, key=lambda g: _pmf_best[g]) if _pmf_best else None
            if _most_likely_g is None: break
            _rstate = frozenset(_rstate | {(_best, _most_likely_g)})
            _rstep += 1
        log(f"    trace complete after {_rstep} steps")

        all_results[key] = {
            "split":      split,
            "ratio2":     ratio2,
            "L":          float(L),
            "V_root":     float(ds["V_root"]),
            "V_stop_root": float(ds["V_stop_root"]),
            "total_sec":  time.time() - t0,
        }
        partial_path.write_text(json.dumps(all_results, indent=2))
        del ds, gnn_q_vals

    summary = {
        "model":         "PedigreeGNN",
        "node_feat_dim": NODE_FEAT_DIM,
        "hidden_dim":    32,
        "n_rounds":      2,
        "n_genes":       len(GENES),
        "genes":         list(GENES),
        "train_keys":    sorted(TRAIN_KEYS),
        "test_keys":     sorted(TEST_KEYS),
        "families":      FAMILIES,
        "families":      all_results,
    }
    (RESULTS_DIR / "results.json").write_text(json.dumps(summary, indent=2))

    log(f"\n[DONE] → {RESULTS_DIR / 'results.json'}")
    log(f"\n  {'Key':<50} {'Split':<6} {'ratio2':>8}  {'V_root':>10}  {'V_stop':>10}")
    log(f"  {'-'*88}")
    for key, r in all_results.items():
        log(f"  {key:<50} {r['split']:<6} {r['ratio2']:>8.6f}  {r['V_root']:>10.6f}  {r['V_stop_root']:>10.6f}")


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--device", default="cpu")
    p.add_argument("--fresh",  action="store_true")
    args = p.parse_args()
    main(device=args.device, fresh=args.fresh)
