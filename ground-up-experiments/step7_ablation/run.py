"""Step 7: GNN architecture ablation — n_rounds × hidden_dim.

Same 3 TRAIN families as step5 (Base only, LowHigh + Extended LowHigh).
Same 1 TEST family (Extended__Base__0.08).

Sweeps:
  n_rounds  ∈ {1, 2, 3, 4}
  hidden_dim ∈ {16, 32, 64}
  → 12 configurations × 200 epochs each

Run:
    python step7_ablation/run.py
"""
from __future__ import annotations

import json
import pickle
import sys
import time
from datetime import datetime
from itertools import product
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


# ─── helpers (identical to step5) ───────────────────────────────────────────

def state_to_graph(state, belief, individuals, pedigree, genes=("GeneA", "GeneB")):
    n   = len(individuals)
    idx = {p: i for i, p in enumerate(individuals)}
    entry = belief[state]
    if isinstance(entry, InferenceResult):
        per_gene = entry.get_per_gene_probs()
    else:
        per_gene = lift_tuple_posteriors_to_genes(entry, genes, GENOTYPE_STATES)
    tested = {p for p, _ in state}
    feats  = np.zeros((n, 7), dtype=np.float32)
    for j, person in enumerate(individuals):
        for gi, gene in enumerate(genes):
            dist = per_gene[gene][person]
            feats[j, gi * 3: gi * 3 + 3] = [dist[0], dist[1], dist[2]]
        feats[j, 6] = 1.0 if person in tested else 0.0
    srcs, dsts = [], []
    for child in pedigree.get_offspring():
        for parent in pedigree.get_parents(child):
            if parent in idx and child in idx:
                srcs.append(idx[parent])
                dsts.append(idx[child])
    edge_index = np.array([srcs, dsts], dtype=np.int64) if srcs else np.zeros((2, 0), dtype=np.int64)
    return feats, edge_index


def build_samples(datasets_list):
    samples = []
    for ds in datasets_list:
        pedigree    = ds["pedigree"]
        individuals = ds["individuals"]
        belief      = ds["belief"]
        genes       = ds.get("genes", ("GeneA", "GeneB"))
        for state, v_star in ds["V_star"].items():
            entry = belief[state]
            marg  = entry.marginals if isinstance(entry, InferenceResult) else entry
            if sum(marg[individuals[0]].values()) < 1e-9: continue
            nf, ei = state_to_graph(state, belief, individuals, pedigree, genes)
            samples.append((nf, ei, float(v_star)))
    return samples


def train_one(samples, model, epochs, device, print_every=50):
    model = model.to(device)
    opt   = torch.optim.Adam(model.parameters(), lr=1e-3)
    crit  = nn.MSELoss()

    groups: dict = {}
    for nf, ei, y in samples:
        key = ei.tobytes()
        if key not in groups:
            groups[key] = {"ei": torch.LongTensor(ei).to(device), "nf": [], "y": []}
        groups[key]["nf"].append(nf)
        groups[key]["y"].append(y)
    for g in groups.values():
        g["nf"] = torch.FloatTensor(np.stack(g["nf"])).to(device)
        g["y"]  = torch.FloatTensor(g["y"]).to(device)

    n_total = sum(g["nf"].shape[0] for g in groups.values())
    n_val   = max(1, int(n_total * 0.2))
    n_train = n_total - n_val

    train_g, val_g = {}, {}
    for key, g in groups.items():
        M   = g["nf"].shape[0]
        n_v = max(1, int(M * 0.2))
        n_t = M - n_v
        train_g[key] = {"ei": g["ei"], "nf": g["nf"][:n_t], "y": g["y"][:n_t]}
        val_g[key]   = {"ei": g["ei"], "nf": g["nf"][n_t:], "y": g["y"][n_t:]}

    def batches(grp, bs=256, shuffle=True):
        for g in grp.values():
            M   = g["nf"].shape[0]
            idx = torch.randperm(M) if shuffle else torch.arange(M)
            for start in range(0, M, bs):
                sl = idx[start: start + bs]
                yield g["nf"][sl], g["ei"], g["y"][sl]

    history = {"train_loss": [], "val_loss": []}
    for epoch in range(1, epochs + 1):
        model.train()
        tl = 0.0
        for nf_b, ei_b, y_b in batches(train_g):
            opt.zero_grad()
            loss = crit(model.forward_batch(nf_b, ei_b), y_b)
            loss.backward()
            opt.step()
            tl += loss.item() * len(y_b)
        tl /= n_train
        model.eval()
        vl = 0.0
        with torch.no_grad():
            for nf_b, ei_b, y_b in batches(val_g, shuffle=False):
                vl += crit(model.forward_batch(nf_b, ei_b), y_b).item() * len(y_b)
        vl /= n_val
        history["train_loss"].append(tl)
        history["val_loss"].append(vl)
        if epoch % print_every == 0:
            print(f"    epoch {epoch:4d}/{epochs}  train={tl:.6f}  val={vl:.6f}")
    return model, history


def compute_ratio2_gnn(model, ds, device="cpu"):
    belief      = ds["belief"]
    individuals = ds["individuals"]
    config      = ds["config"]
    pedigree    = ds["pedigree"]
    genes       = ds.get("genes", ("GeneA", "GeneB"))
    model.eval()
    memo: dict = {}

    def _net_val(state):
        if state in memo: return memo[state]
        nf, ei = state_to_graph(state, belief, individuals, pedigree, genes)
        with torch.no_grad():
            v = model(torch.FloatTensor(nf).to(device), torch.LongTensor(ei).to(device)).item()
        memo[state] = v
        return v

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
            return entry.marginals, entry.get_per_gene_probs(), entry.get_tuple_pmfs()
        return entry, lift_tuple_posteriors_to_genes(entry, genes, GENOTYPE_STATES), None

    def _pmf(tuple_pmfs, marg, person):
        return tuple_pmfs.get(person, {}) if tuple_pmfs is not None else marg[person]

    true_memo: dict = {}

    def value_at(state):
        if state in true_memo: return true_memo[state]
        marg, per_gene, tuple_pmfs = _get_entry(state)
        tested  = {i for i, _ in state}
        v_stop  = _stop_val(per_gene, tested)
        if len(tested) == len(individuals):
            true_memo[state] = 0.0
            return 0.0
        best_q, best_p = v_stop, None
        for i in individuals:
            if i in tested: continue
            r_i   = _test_r(i, per_gene)
            pmf_i = _pmf(tuple_pmfs, marg, i)
            exp   = sum(p * _net_val(frozenset(state | {(i, g)}))
                        for g, p in pmf_i.items()
                        if p > 1e-12 and frozenset(state | {(i, g)}) in belief)
            if r_i + exp > best_q:
                best_q, best_p = r_i + exp, i
        if best_p is None:
            true_memo[state] = v_stop
            return v_stop
        r_b  = _test_r(best_p, per_gene)
        pmf_b = _pmf(tuple_pmfs, marg, best_p)
        exp_t = sum(p * value_at(frozenset(state | {(best_p, g)}))
                    for g, p in pmf_b.items()
                    if p > 1e-12 and frozenset(state | {(best_p, g)}) in belief)
        result = r_b + exp_t
        true_memo[state] = result
        return result

    L = value_at(frozenset())
    V_root, V_stop = ds["V_root"], ds["V_stop_root"]
    denom = V_root - V_stop
    return float((V_root - L) / denom if abs(denom) > 1e-12 else 0.0), float(L)


# ─── configs ─────────────────────────────────────────────────────────────────

TRAIN_CONFIGS = [
    {"family_label": "ThreeGeneration", "allele_freqs": {"GeneA": 0.02, "GeneB": 0.15}, "preset_label": "Base"},
    {"family_label": "ThreeGeneration", "allele_freqs": {"GeneA": 0.08, "GeneB": 0.08}, "preset_label": "Base"},
    {"family_label": "Extended",        "allele_freqs": {"GeneA": 0.02, "GeneB": 0.15}, "preset_label": "Base"},
]
TEST_CONFIGS = [
    {"family_label": "Extended",        "allele_freqs": {"GeneA": 0.08, "GeneB": 0.08}, "preset_label": "Base"},
]
ALL_CONFIGS  = TRAIN_CONFIGS + TEST_CONFIGS

def cfg_key(cfg):
    return f"{cfg['family_label']}__{cfg['preset_label']}__{cfg['allele_freqs']['GeneA']}"

SWEEP = list(product([1, 2, 3, 4], [16, 32, 64]))  # (n_rounds, hidden_dim)


def main(device="cpu", epochs=200, fresh=False):
    results_dir = HERE / "results"
    cache_dir   = results_dir / "cache"
    results_dir.mkdir(exist_ok=True)
    cache_dir.mkdir(exist_ok=True)

    step5_cache = HERE.parent / "step5_gnn" / "results" / "cache"

    log_f = open(results_dir / "run.log", "a")
    log_f.write(f"\n{'='*60}\n[RUN] {datetime.now().isoformat()}\n{'='*60}\n")
    def log(msg=""):
        print(msg); log_f.write(msg + "\n"); log_f.flush()

    # ── 1. Build / cache datasets ─────────────────────────────────────────────
    log("\n[1] Datasets:")
    for cfg in ALL_CONFIGS:
        key      = cfg_key(cfg)
        pkl_path = cache_dir / f"{key}.pkl"
        step5_pkl = step5_cache / f"{key}.pkl"
        if not fresh and not pkl_path.exists() and step5_pkl.exists():
            import shutil
            shutil.copy2(step5_pkl, pkl_path)
            log(f"    {key}  [LINKED from step5 cache]")
            continue
        if not fresh and pkl_path.exists():
            try:
                with open(pkl_path, "rb") as f: pickle.load(f)
                log(f"    {key}  [CACHED]")
                continue
            except Exception:
                pkl_path.unlink(missing_ok=True)
        t0 = time.time()
        log(f"    {key}  [building...]")
        ds  = build_two_gene_dataset(**cfg)
        tmp = pkl_path.with_suffix(".tmp")
        with open(tmp, "wb") as f: pickle.dump(ds, f)
        tmp.replace(pkl_path)
        log(f"    done in {time.time()-t0:.1f}s  ({len(ds['X'])} states)")

    # ── 2. Load train datasets once ───────────────────────────────────────────
    log("\n[2] Loading train datasets...")
    train_datasets = []
    for cfg in TRAIN_CONFIGS:
        with open(cache_dir / f"{cfg_key(cfg)}.pkl", "rb") as f:
            train_datasets.append(pickle.load(f))
    samples = build_samples(train_datasets)
    log(f"    {len(samples)} training samples from {len(TRAIN_CONFIGS)} families")

    # Load test datasets
    test_datasets = {}
    for cfg in TEST_CONFIGS:
        key = cfg_key(cfg)
        with open(cache_dir / f"{key}.pkl", "rb") as f:
            test_datasets[key] = pickle.load(f)
    train_eval_datasets = {cfg_key(c): ds for c, ds in zip(TRAIN_CONFIGS, train_datasets)}

    # ── 3. Sweep ──────────────────────────────────────────────────────────────
    results_path = results_dir / "results.json"
    all_results  = json.loads(results_path.read_text()) if results_path.exists() else {}

    log(f"\n[3] Sweep: {len(SWEEP)} configs × {epochs} epochs each")
    log(f"    Already done: {len(all_results)}/{len(SWEEP)}")

    for n_rounds, hidden_dim in SWEEP:
        run_key = f"r{n_rounds}_h{hidden_dim}"
        if run_key in all_results:
            log(f"\n    {run_key}  [DONE — ratio2_test={all_results[run_key].get('ratio2_test_avg', '?'):.4f}]")
            continue

        n_params = sum(p.numel() for p in PedigreeGNN(7, hidden_dim, n_rounds).parameters())
        log(f"\n    [{run_key}]  n_rounds={n_rounds}  hidden_dim={hidden_dim}  params={n_params}")

        model = PedigreeGNN(node_feat_dim=7, hidden_dim=hidden_dim, n_rounds=n_rounds)
        t0    = time.time()
        model, history = train_one(samples, model, epochs=epochs, device=device, print_every=50)
        train_time = time.time() - t0
        log(f"    trained in {train_time:.0f}s  final_train_loss={history['train_loss'][-1]:.6f}")

        # Evaluate on all 4 families
        family_results = {}
        for cfg in TRAIN_CONFIGS:
            key = cfg_key(cfg)
            ratio2, L = compute_ratio2_gnn(model, train_eval_datasets[key], device)
            family_results[key] = {"split": "TRAIN", "ratio2": ratio2, "L": L,
                                   "V_root": float(train_eval_datasets[key]["V_root"]),
                                   "V_stop_root": float(train_eval_datasets[key]["V_stop_root"])}
            log(f"    TRAIN  {key:<40} ratio2={ratio2:.6f}")
        for key, ds in test_datasets.items():
            ratio2, L = compute_ratio2_gnn(model, ds, device)
            family_results[key] = {"split": "TEST", "ratio2": ratio2, "L": L,
                                   "V_root": float(ds["V_root"]),
                                   "V_stop_root": float(ds["V_stop_root"])}
            log(f"    TEST   {key:<40} ratio2={ratio2:.6f}")

        train_r2s = [v["ratio2"] for v in family_results.values() if v["split"] == "TRAIN"]
        test_r2s  = [v["ratio2"] for v in family_results.values() if v["split"] == "TEST"]

        all_results[run_key] = {
            "n_rounds": n_rounds, "hidden_dim": hidden_dim, "n_params": n_params,
            "train_loss_final": float(history["train_loss"][-1]),
            "val_loss_final":   float(history["val_loss"][-1]),
            "train_time_s": round(train_time),
            "ratio2_train_avg": float(np.mean(train_r2s)),
            "ratio2_test_avg":  float(np.mean(test_r2s)),
            "families": family_results,
        }
        results_path.write_text(json.dumps(all_results, indent=2))
        log(f"    → saved  train_avg={np.mean(train_r2s):.4f}  test_avg={np.mean(test_r2s):.4f}")

    # ── 4. Summary table ─────────────────────────────────────────────────────
    log(f"\n[DONE]\n")
    log(f"  {'Config':<12} {'n_rounds':>8} {'hidden':>7} {'params':>7} "
        f"{'train_r2':>10} {'test_r2':>10} {'train_loss':>12}")
    log(f"  {'-'*75}")
    for run_key, r in sorted(all_results.items()):
        log(f"  {run_key:<12} {r['n_rounds']:>8} {r['hidden_dim']:>7} {r['n_params']:>7} "
            f"{r['ratio2_train_avg']:>10.4f} {r['ratio2_test_avg']:>10.4f} "
            f"{r['train_loss_final']:>12.2e}")


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--device", default="cpu")
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--fresh",  action="store_true")
    args = p.parse_args()
    main(device=args.device, target_epochs=args.epochs if hasattr(args, "target_epochs") else args.epochs, fresh=args.fresh)
