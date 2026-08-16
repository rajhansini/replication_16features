"""Step 5: GNN over pedigree graph.

Replace flat MLP with a graph neural network that respects pedigree structure.
Node features: [P(g=0)_GeneA, P(g=1)_GeneA, P(g=2)_GeneA,
                P(g=0)_GeneB, P(g=1)_GeneB, P(g=2)_GeneB,
                is_tested]   → 7 features per person/node.

Edges: parent → child (directed).

Compare ratio2 vs step 3/4 MLP results.

Run:
    python step5_gnn/run.py
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
import torch.nn as nn

HERE         = Path(__file__).resolve().parent
EXPERIMENTS  = HERE.parent
PROJECT_ROOT = EXPERIMENTS.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(EXPERIMENTS))

from shared.data_gen import build_two_gene_dataset
from shared.model    import PedigreeGNN
from genetic_dp.exact_dp.utils import GENOTYPE_STATES, lift_tuple_posteriors_to_genes
from genetic_dp.models.belief  import InferenceResult
from genetic_dp.models.reward  import r_reward, r_reward_test


# ─── state → graph ───────────────────────────────────────────────────────────

def config_to_cost_vec(config, genes) -> np.ndarray:
    """Extract scalar cost params as a flat vector for GNN conditioning.

    For each gene: [a_base, b_base, delta] (founder's values = preset scalars).
    Then: [fixed_cost, variable_cost].
    dim = 3*n_genes + 2  (8 for 2 genes, 11 for 3 genes).
    """
    vec = []
    for g in genes:
        vec.append(min(config.a_gene[g].values(), key=abs))
        vec.append(min(config.b_gene[g].values(), key=abs))
        vec.append(list(config.delta_gene[g].values())[0])
    vec.append(config.fixed_cost)
    vec.append(config.variable_cost)
    return np.array(vec, dtype=np.float32)


def state_to_graph(state, belief, individuals, pedigree, genes=("GeneA", "GeneB")):
    """
    Convert a belief state into (node_features, edge_index).

    node_features : (n_nodes, 7)  — 6 gene probs + is_tested flag
    edge_index    : (2, n_edges)  — directed parent→child edges
    """
    n = len(individuals)
    idx = {p: i for i, p in enumerate(individuals)}

    entry = belief[state]
    if isinstance(entry, InferenceResult):
        per_gene = entry.get_per_gene_probs()
    else:
        per_gene = lift_tuple_posteriors_to_genes(entry, genes, GENOTYPE_STATES)
    tested = {p for p, _ in state}

    feats = np.zeros((n, 7), dtype=np.float32)
    for j, person in enumerate(individuals):
        for gi, gene in enumerate(genes):
            dist = per_gene[gene][person]
            feats[j, gi * 3: gi * 3 + 3] = [dist[0], dist[1], dist[2]]
        feats[j, 6] = 1.0 if person in tested else 0.0

    # Build edge index (parent → child)
    srcs, dsts = [], []
    for child in pedigree.get_offspring():
        for parent in pedigree.get_parents(child):
            if parent in idx and child in idx:
                srcs.append(idx[parent])
                dsts.append(idx[child])
    if srcs:
        edge_index = np.array([srcs, dsts], dtype=np.int64)
    else:
        edge_index = np.zeros((2, 0), dtype=np.int64)

    return feats, edge_index


def compute_q_labels(ds: dict, individuals: list, genes: tuple) -> tuple[list, np.ndarray]:
    """
    Compute Q*(s, i) for every state s and person i.
    Q*(s, i) = r_test(i, s) + E[V*(s')] for untested persons; NaN for tested.
    Returns (states_list, q_array) where q_array is (N_states, N_people).
    """
    v_star  = ds["V_star"]   # {state: V*} from exact DP
    belief  = ds["belief"]
    config  = ds["config"]
    states  = list(v_star.keys())
    N       = len(states)
    P       = len(individuals)
    pid     = {p: j for j, p in enumerate(individuals)}

    q = np.full((N, P), np.nan, dtype=np.float32)

    for i_s, s in enumerate(states):
        tested = {person for person, _ in s}
        if len(tested) == P:
            continue

        entry = belief[s]
        if isinstance(entry, InferenceResult):
            per_gene   = entry.get_per_gene_probs()
            tuple_pmfs = entry.get_tuple_pmfs()
        else:
            per_gene   = lift_tuple_posteriors_to_genes(entry, genes, GENOTYPE_STATES)
            tuple_pmfs = None

        for person in individuals:
            if person in tested:
                continue
            j = pid[person]

            r_i = float(r_reward_test(
                person, None, config.a, config.b, config.c, config.delta,
                config.fixed_cost, config.variable_cost,
                per_gene_probs=per_gene,
                a_gene=config.a_gene, c_gene=config.c_gene,
                delta_gene=config.delta_gene,
            ))

            pmf = (tuple_pmfs or {}).get(person, {})
            if not pmf and not isinstance(entry, InferenceResult):
                pmf = entry.get(person, {})

            exp_val = sum(
                prob * v_star.get(frozenset(s | {(person, g)}), 0.0)
                for g, prob in pmf.items() if prob > 1e-12
            )

            q[i_s, j] = r_i + exp_val

    return states, q


# ─── training loop for GNN ───────────────────────────────────────────────────

def train_gnn(datasets_list, model: PedigreeGNN,
              epochs: int = 500, start_epoch: int = 1,
              lr: float = 1e-3, val_frac: float = 0.2,
              device: str = "cpu", print_every: int = 50,
              ckpt_path=None, ckpt_every: int = 50,
              history=None) -> tuple[PedigreeGNN, dict]:
    """Train GNN on Q*(s,i) labels with masked MSE loss."""
    model     = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    if history is None:
        history = {"train_loss": [], "val_loss": []}

    samples = datasets_list[0].get("_samples", [])
    if not samples:
        for ds in datasets_list:
            individuals = ds["individuals"]
            pedigree    = ds["pedigree"]
            belief      = ds["belief"]
            genes       = ds.get("genes", ("GeneA", "GeneB"))
            cost_vec    = config_to_cost_vec(ds["config"], genes)
            states, q_labels = compute_q_labels(ds, individuals, genes)
            for i_s, state in enumerate(states):
                nf, ei = state_to_graph(state, belief, individuals, pedigree, genes)
                samples.append((nf, ei, cost_vec, q_labels[i_s]))

    # Group by edge_index topology
    groups: dict[bytes, dict] = {}
    for nf, ei, gf, q in samples:
        key = ei.tobytes()
        if key not in groups:
            groups[key] = {"ei": torch.LongTensor(ei).to(device), "nf": [], "gf": [], "q": []}
        groups[key]["nf"].append(nf)
        groups[key]["gf"].append(gf)
        groups[key]["q"].append(q)

    for g in groups.values():
        g["nf"] = torch.FloatTensor(np.stack(g["nf"])).to(device)
        g["gf"] = torch.FloatTensor(np.stack(g["gf"])).to(device)
        g["q"]  = torch.FloatTensor(np.stack(g["q"])).to(device)   # (M, N_people)

    def masked_mse(q_pred, q_true):
        mask = ~torch.isnan(q_true)
        return ((q_pred[mask] - q_true[mask]) ** 2).mean()

    train_groups, val_groups = {}, {}
    for key, g in groups.items():
        M   = g["nf"].shape[0]
        n_v = max(1, int(M * val_frac))
        n_t = M - n_v
        train_groups[key] = {"ei": g["ei"], "nf": g["nf"][:n_t], "gf": g["gf"][:n_t], "q": g["q"][:n_t]}
        val_groups[key]   = {"ei": g["ei"], "nf": g["nf"][n_t:], "gf": g["gf"][n_t:], "q": g["q"][n_t:]}

    n_train = sum(g["nf"].shape[0] for g in train_groups.values())
    n_val   = sum(g["nf"].shape[0] for g in val_groups.values())
    batch_size = 256

    for epoch in range(start_epoch, epochs + 1):
        model.train()
        train_loss = 0.0
        for g in train_groups.values():
            M   = g["nf"].shape[0]
            idx = torch.randperm(M)
            for start in range(0, M, batch_size):
                sl = idx[start: start + batch_size]
                optimizer.zero_grad()
                q_pred = model.forward_batch(g["nf"][sl], g["ei"], g["gf"][sl])
                loss   = masked_mse(q_pred, g["q"][sl])
                loss.backward()
                optimizer.step()
                train_loss += loss.item() * len(sl)
        train_loss /= n_train

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for g in val_groups.values():
                M = g["nf"].shape[0]
                for start in range(0, M, batch_size):
                    end = min(start + batch_size, M)
                    q_pred = model.forward_batch(g["nf"][start:end], g["ei"], g["gf"][start:end])
                    val_loss += masked_mse(q_pred, g["q"][start:end]).item() * (end - start)
        val_loss /= n_val

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)

        if epoch % print_every == 0:
            print(f"  epoch {epoch:4d}/{epochs}  train_loss={train_loss:.6f}  val_loss={val_loss:.6f}")

        if ckpt_path is not None and epoch % ckpt_every == 0:
            tmp = Path(str(ckpt_path) + ".tmp")
            torch.save({"epoch": epoch, "model_state": model.state_dict(), "history": history}, tmp)
            tmp.replace(Path(ckpt_path))

    return model, history


# ─── ratio2 for GNN ──────────────────────────────────────────────────────────

def compute_ratio2_gnn(model: PedigreeGNN, ds: dict, device: str = "cpu") -> tuple[float, float]:
    belief      = ds["belief"]
    individuals = ds["individuals"]
    config      = ds["config"]
    pedigree    = ds["pedigree"]
    genes       = ds.get("genes", ("GeneA", "GeneB"))
    pid         = {p: j for j, p in enumerate(individuals)}

    cost_vec = torch.FloatTensor(config_to_cost_vec(config, genes)).to(device)
    model.eval()

    def _gnn_q(state) -> np.ndarray:
        """Run GNN on state → (N_people,) Q values."""
        nf, ei = state_to_graph(state, belief, individuals, pedigree, genes)
        with torch.no_grad():
            return model(
                torch.FloatTensor(nf).to(device),
                torch.LongTensor(ei).to(device),
                cost_vec,
            ).cpu().numpy()   # (N_people,)

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

    def _get_entry(state):
        entry = belief[state]
        if isinstance(entry, InferenceResult):
            marg       = entry.marginals
            per_gene   = entry.get_per_gene_probs()
            tuple_pmfs = entry.get_tuple_pmfs()
        else:
            marg       = entry
            per_gene   = lift_tuple_posteriors_to_genes(entry, genes, GENOTYPE_STATES)
            tuple_pmfs = None
        return marg, per_gene, tuple_pmfs

    def _pmf_for_person(tuple_pmfs, marg, person):
        if tuple_pmfs is not None:
            return tuple_pmfs.get(person, {})
        return marg[person]

    true_memo: dict = {}

    def value_at(state) -> float:
        if state in true_memo:
            return true_memo[state]
        marg, per_gene, tuple_pmfs = _get_entry(state)
        tested = {i for i, _ in state}
        v_stop = _stop_val(per_gene, tested)

        if len(tested) == len(individuals):
            true_memo[state] = 0.0
            return 0.0

        # Pick action using GNN Q values directly
        q_vec    = _gnn_q(state)                              # (N_people,)
        untested = [i for i in individuals if i not in tested]
        best_person = max(untested, key=lambda i: q_vec[pid[i]])
        best_q      = float(q_vec[pid[best_person]])

        if best_q <= v_stop:
            true_memo[state] = v_stop
            return v_stop

        r_best   = _test_r(best_person, per_gene)
        pmf_best = _pmf_for_person(tuple_pmfs, marg, best_person)
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

CONFIGS = [
    {"family_label": "Extended",        "allele_freqs": {"GeneA": 0.02, "GeneB": 0.15}, "preset_label": "Base"},
    {"family_label": "Extended",        "allele_freqs": {"GeneA": 0.02, "GeneB": 0.15}, "preset_label": "Aggressive"},
    {"family_label": "Extended",        "allele_freqs": {"GeneA": 0.08, "GeneB": 0.08}, "preset_label": "Base"},
    {"family_label": "Extended",        "allele_freqs": {"GeneA": 0.08, "GeneB": 0.08}, "preset_label": "Aggressive"},
    {"family_label": "ThreeGeneration", "allele_freqs": {"GeneA": 0.02, "GeneB": 0.15}, "preset_label": "Base"},
    {"family_label": "ThreeGeneration", "allele_freqs": {"GeneA": 0.02, "GeneB": 0.15}, "preset_label": "Aggressive"},
    {"family_label": "ThreeGeneration", "allele_freqs": {"GeneA": 0.08, "GeneB": 0.08}, "preset_label": "Base"},
    {"family_label": "ThreeGeneration", "allele_freqs": {"GeneA": 0.08, "GeneB": 0.08}, "preset_label": "Aggressive"},
]
# Train on 6 (same as MLP step4): all Extended + ThreeGeneration LowHigh
# Test on 2: ThreeGeneration MediumEven
TRAIN_CFGS = [c for c in CONFIGS if not (c["family_label"] == "ThreeGeneration" and c["allele_freqs"]["GeneA"] == 0.08)]
def cfg_key(cfg): return f"{cfg['family_label']}__{cfg['preset_label']}__{cfg['allele_freqs']['GeneA']}"


def main(device: str = "cpu", target_epochs: int = 500, fresh: bool = False):
    results_dir = HERE / "results"
    cache_dir   = results_dir / "cache"
    ckpt_path   = results_dir / "gnn_ckpt.pt"
    partial_path= results_dir / "partial_eval.json"
    results_dir.mkdir(exist_ok=True)
    cache_dir.mkdir(exist_ok=True)

    # logging
    log_f = open(results_dir / "run.log", "a")
    log_f.write(f"\n{'='*60}\n[RUN] {datetime.now().isoformat()}\n{'='*60}\n")
    def log(msg=""):
        print(msg); log_f.write(msg + "\n"); log_f.flush()

    keys       = [cfg_key(c) for c in CONFIGS]
    train_keys = [cfg_key(c) for c in TRAIN_CFGS]
    test_keys  = [k for k in keys if k not in train_keys]

    # ── 1. Build / cache datasets ─────────────────────────────────────────────
    log("\n[1] Datasets:")
    for cfg in CONFIGS:
        key      = cfg_key(cfg)
        pkl_path = cache_dir / f"{key}.pkl"
        if not fresh and pkl_path.exists():
            try:
                with open(pkl_path, "rb") as f: pickle.load(f)  # validate
                log(f"    {key}  [CACHED]")
                continue
            except Exception:
                log(f"    {key}  [CORRUPT — rebuilding]")
                pkl_path.unlink(missing_ok=True)
        t0 = time.time()
        log(f"    {key}  [building...]")
        ds  = build_two_gene_dataset(**cfg)
        tmp = pkl_path.with_suffix(".tmp")
        with open(tmp, "wb") as f: pickle.dump(ds, f)
        tmp.replace(pkl_path)
        log(f"    done in {time.time()-t0:.1f}s  ({len(ds['X'])} states)")

    # ── 2. Train GNN (with checkpointing every 50 epochs) ────────────────────
    log(f"\n[2] Training GNN:")
    model = PedigreeGNN(node_feat_dim=7, hidden_dim=32, n_rounds=2, global_feat_dim=8)

    start_epoch = 1
    history     = {"train_loss": [], "val_loss": []}
    if not fresh and ckpt_path.exists():
        ckpt        = torch.load(ckpt_path, map_location=device)
        start_epoch = ckpt["epoch"] + 1
        model.load_state_dict(ckpt["model_state"])
        history     = ckpt.get("history", history)
        log(f"    Resumed from checkpoint at epoch {ckpt['epoch']}/{target_epochs}")
    else:
        log(f"    Starting fresh (0/{target_epochs})")

    if start_epoch <= target_epochs:
        # Load train datasets, compute Q* labels
        samples = []
        for cfg in CONFIGS:
            key = cfg_key(cfg)
            if key not in train_keys: continue
            with open(cache_dir / f"{key}.pkl", "rb") as f: ds = pickle.load(f)
            individuals = ds["individuals"]
            pedigree    = ds["pedigree"]
            belief      = ds["belief"]
            genes       = ds.get("genes", ("GeneA", "GeneB"))
            cost_vec    = config_to_cost_vec(ds["config"], genes)
            states, q_labels = compute_q_labels(ds, individuals, genes)
            for i_s, state in enumerate(states):
                nf, ei = state_to_graph(state, belief, individuals, pedigree, genes)
                samples.append((nf, ei, cost_vec, q_labels[i_s]))
            del ds
        log(f"    Train samples: {len(samples)}")

        # ── Verification block (writes to verify.log) ─────────────────────
        vlog_path = results_dir / "verify.log"
        vlog_f    = open(vlog_path, "w")
        def vlog(msg=""):
            print(msg, flush=True); vlog_f.write(msg + "\n"); vlog_f.flush()

        vlog(f"{'='*70}")
        vlog(f"STEP5 PIPELINE VERIFICATION  —  {datetime.now().isoformat()}")
        vlog(f"{'='*70}")

        _vcfg = TRAIN_CFGS[0]
        _vkey = cfg_key(_vcfg)
        vlog(f"\nUsing config: {_vkey}")
        with open(cache_dir / f"{_vkey}.pkl", "rb") as f:
            _vds = pickle.load(f)

        _vind    = _vds["individuals"]
        _vstar   = _vds["V_star"]       # {frozenset: float}
        _vstates = list(_vstar.keys())
        _vbelief = _vds["belief"]
        _vconfig = _vds["config"]
        _vgenes  = _vds.get("genes", ("GeneA", "GeneB"))
        _vpedigree = _vds["pedigree"]

        # [A] DP value sanity
        vlog(f"\n[A] DP VALUE SANITY")
        vlog(f"    Individuals:         {_vind}")
        vlog(f"    Total states:        {len(_vstates):,}")
        vlog(f"    V_root (ds field):   {_vds['V_root']:.8f}")
        vlog(f"    V_stop_root (ds):    {_vds['V_stop_root']:.8f}")
        _v_root_dp = _vstar.get(frozenset(), float("nan"))
        _match = abs(_v_root_dp - _vds["V_root"]) < 1e-6
        vlog(f"    V*(empty state):     {_v_root_dp:.8f}  {'OK — matches V_root' if _match else 'MISMATCH!'}")
        _full  = [s for s in _vstates if len(s) == len(_vind)]
        _nz    = sum(1 for s in _full if abs(_vstar[s]) > 1e-6)
        vlog(f"    Fully-tested states: {len(_full)}")
        vlog(f"    V*(fully-tested)!=0: {_nz}  (should be 0)")
        for _s in _full[:3]:
            vlog(f"      V*({dict(_s)}) = {_vstar[_s]:.8f}")

        # [B] Q* formula check
        vlog(f"\n[B] Q* FORMULA CHECK  —  Q*(s,i) = r_test(i,s) + E[V*(s')]")
        _partial = [s for s in _vstates if 0 < len(s) < len(_vind)][:5]
        for _s in _partial:
            vlog(f"\n    State: {dict(_s)}")
            _tested = {p for p, _ in _s}
            _entry  = _vbelief[_s]
            if isinstance(_entry, InferenceResult):
                _pg   = _entry.get_per_gene_probs()
                _pmfs = _entry.get_tuple_pmfs()
            else:
                _pg   = lift_tuple_posteriors_to_genes(_entry, _vgenes, GENOTYPE_STATES)
                _pmfs = None
            for _p in _vind:
                if _p in _tested:
                    vlog(f"      {_p}: TESTED → NaN"); continue
                _ri = float(r_reward_test(
                    _p, None, _vconfig.a, _vconfig.b, _vconfig.c, _vconfig.delta,
                    _vconfig.fixed_cost, _vconfig.variable_cost,
                    per_gene_probs=_pg, a_gene=_vconfig.a_gene,
                    c_gene=_vconfig.c_gene, delta_gene=_vconfig.delta_gene,
                ))
                _pmf = (_pmfs or {}).get(_p, {})
                _ev  = sum(prob * _vstar.get(frozenset(_s | {(_p, g)}), 0.0)
                           for g, prob in _pmf.items() if prob > 1e-12)
                vlog(f"      {_p}: r_test={_ri:.6f}  E[V*]={_ev:.6f}  Q*={_ri+_ev:.6f}")

        # [C] Node feature sanity
        vlog(f"\n[C] NODE FEATURE SANITY")
        _nf0, _ei0 = state_to_graph(frozenset(), _vbelief, _vind, _vpedigree, _vgenes)
        vlog(f"    Node features shape (single state): {_nf0.shape}  (expected ({len(_vind)}, 7))")
        _n_bad = 0
        for _ss in _vstates[:50]:
            _nfs, _ = state_to_graph(_ss, _vbelief, _vind, _vpedigree, _vgenes)
            _tested_ss = {p for p, _ in _ss}
            for _j, _pp in enumerate(_vind):
                for _gi in range(len(_vgenes)):
                    if abs(_nfs[_j, _gi*3:_gi*3+3].sum() - 1.0) > 1e-3:
                        _n_bad += 1
                _exp_flag = 1.0 if _pp in _tested_ss else 0.0
                if abs(_nfs[_j, 6] - _exp_flag) > 0.5:
                    _n_bad += 1
        vlog(f"    Bad entries in first 50 states (prob!=1 or wrong flag): {_n_bad}  (should be 0)")
        vlog(f"    Root-state node features:")
        for _j, _pp in enumerate(_vind):
            vlog(f"      {_pp}: {_nf0[_j].tolist()}")

        # [D] Model output shape
        vlog(f"\n[D] MODEL OUTPUT SHAPE CHECK  (must be (B, N_people), not (B,))")
        _mtest = PedigreeGNN(node_feat_dim=7, hidden_dim=32, n_rounds=2, global_feat_dim=8)
        _mtest.eval()
        _cv_v = config_to_cost_vec(_vconfig, _vgenes)
        _nft_v = torch.FloatTensor(np.stack([_nf0]*4))
        _eit_v = torch.LongTensor(_ei0)
        _gft_v = torch.FloatTensor(np.tile(_cv_v, (4, 1)))
        with torch.no_grad():
            _out_v = _mtest.forward_batch(_nft_v, _eit_v, _gft_v)
        vlog(f"    Output shape: {tuple(_out_v.shape)}  (expected (4, {len(_vind)}))")
        vlog(f"    Sample Q values (untrained): { {_vind[_j]: round(float(_out_v[0,_j]),4) for _j in range(len(_vind))} }")

        # [E] Q* label NaN check
        vlog(f"\n[E] Q* LABEL MASK CHECK")
        _, _ql_v = compute_q_labels(_vds, _vind, _vgenes)
        _nan_cnt   = int(np.isnan(_ql_v).sum())
        _valid_cnt = int((~np.isnan(_ql_v)).sum())
        vlog(f"    Shape: {_ql_v.shape}  (expected ({len(_vstates)}, {len(_vind)}))")
        vlog(f"    NaN entries   (tested persons):  {_nan_cnt}")
        vlog(f"    Valid entries (untested persons): {_valid_cnt}")
        vlog(f"    Q* value range: [{float(np.nanmin(_ql_v)):.6f},  {float(np.nanmax(_ql_v)):.6f}]")

        vlog(f"\n{'='*70}")
        vlog(f"VERIFICATION DONE. Read verify.log.")
        vlog(f"{'='*70}\n")
        vlog_f.close()
        log(f"    Verification written → {vlog_path}")
        del _vds
        # ── End verification ──────────────────────────────────────────────────

        model, history = train_gnn(
            [{"_samples": samples}], model,
            epochs=target_epochs, start_epoch=start_epoch,
            lr=1e-3, device=device, print_every=50,
            ckpt_path=ckpt_path, ckpt_every=50,
            history=history,
        )
        log("    Training complete.")
    else:
        log(f"    Already at {start_epoch-1}/{target_epochs} — skipping training.")

    model.to(device).eval()

    # ── 3. Evaluate (incremental) ─────────────────────────────────────────────
    all_results = json.loads(partial_path.read_text()) if partial_path.exists() else {}
    pending     = [c for c in CONFIGS if cfg_key(c) not in all_results]
    log(f"\n[3] Evaluation — done {len(all_results)}/4, pending {len(pending)}/4")

    for cfg in pending:
        key   = cfg_key(cfg)
        split = "TRAIN" if key in train_keys else "TEST"
        log(f"\n    [{split}] {key} — loading...")
        with open(cache_dir / f"{key}.pkl", "rb") as f: ds = pickle.load(f)

        t0 = time.time()
        ratio2, L = compute_ratio2_gnn(model, ds, device=device)
        log(f"    ratio2={ratio2:.6f}  ({time.time()-t0:.1f}s)")

        # rollout trace
        log(f"    ROLLOUT TRACE:")
        _rind = ds["individuals"]; _rpid = {p: j for j, p in enumerate(_rind)}
        _rcfg = ds["config"]; _rgb = ds["belief"]; _rgenes = ds.get("genes", ("GeneA","GeneB"))
        _rcv  = torch.FloatTensor(config_to_cost_vec(_rcfg, _rgenes)).to(device)
        def _r_pg5(st):
            e = _rgb[st]
            if isinstance(e, InferenceResult):
                return e.get_per_gene_probs(), e.get_tuple_pmfs()
            return lift_tuple_posteriors_to_genes(e, _rgenes, GENOTYPE_STATES), None
        def _r_vs5(pg, tested):
            return float(sum(r_reward(k, None, _rcfg.a, _rcfg.b, _rcfg.c, _rcfg.delta,
                per_gene_probs=pg, a_gene=_rcfg.a_gene, b_gene=_rcfg.b_gene,
                c_gene=_rcfg.c_gene, delta_gene=_rcfg.delta_gene)
                for k in _rind if k not in tested))
        _rst = frozenset(); _rstep5 = 0
        while len({p for p, _ in _rst}) < len(_rind) and _rstep5 < 15:
            _tested5 = {p for p, _ in _rst}
            _pg5, _  = _r_pg5(_rst)
            _vs5     = _r_vs5(_pg5, _tested5)
            _untst5  = [p for p in _rind if p not in _tested5]
            _nf5, _ei5 = state_to_graph(_rst, _rgb, _rind, ds["pedigree"], _rgenes)
            with torch.no_grad():
                _qv5 = model(torch.FloatTensor(_nf5).to(device), torch.LongTensor(_ei5).to(device), _rcv).cpu().numpy()
            _qs5 = {p: float(_qv5[_rpid[p]]) for p in _untst5}
            _best5 = max(_untst5, key=lambda p: _qs5[p])
            _bq5   = _qs5[_best5]
            log(f"      step {_rstep5}: tested={sorted(_tested5) or '{}'}")
            log(f"        Q: { {p: round(v,5) for p, v in _qs5.items()} }  v_stop={_vs5:.5f}")
            if _bq5 <= _vs5:
                log(f"        → STOP (best_Q={_bq5:.5f} <= v_stop={_vs5:.5f})"); break
            log(f"        → TEST {_best5}")
            _pg5r, _pmfs5 = _r_pg5(_rst)
            _pmf5b = (_pmfs5 or {}).get(_best5, {})
            if not _pmf5b:
                _pmf5b = _pg5r.get(_best5, {})
            _mlg5 = max(_pmf5b, key=lambda g: _pmf5b[g]) if _pmf5b else None
            if _mlg5 is None: break
            _rst = frozenset(_rst | {(_best5, _mlg5)}); _rstep5 += 1
        log(f"      trace done ({_rstep5} steps)")

        all_results[key] = {
            "split": split, "ratio2": float(ratio2), "L": float(L),
            "V_root": float(ds["V_root"]), "V_stop_root": float(ds["V_stop_root"]),
        }
        partial_path.write_text(json.dumps(all_results, indent=2))
        del ds

    # ── 4. Save ───────────────────────────────────────────────────────────────
    summary = {
        "model": "PedigreeGNN", "node_feat_dim": 7, "hidden_dim": 32, "n_rounds": 2,
        "families": all_results,
        "train_loss_final": float(history["train_loss"][-1]) if history["train_loss"] else None,
    }
    (results_dir / "results.json").write_text(json.dumps(summary, indent=2))
    torch.save(model.state_dict(), results_dir / "gnn_model.pt")

    log(f"\n[DONE] → {results_dir / 'results.json'}")
    log(f"\n  {'Key':<45} {'Split':<6} {'ratio2':>8}")
    log(f"  {'-'*62}")
    for key, r in all_results.items():
        log(f"  {key:<45} {r['split']:<6} {r['ratio2']:>8.6f}")


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--device", default="cpu")
    p.add_argument("--epochs", type=int, default=500)
    p.add_argument("--fresh",  action="store_true")
    args = p.parse_args()
    main(device=args.device, target_epochs=args.epochs, fresh=args.fresh)
