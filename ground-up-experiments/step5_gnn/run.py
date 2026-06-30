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


# ─── training loop for GNN ───────────────────────────────────────────────────

def train_gnn(datasets_list, model: PedigreeGNN,
              epochs: int = 500, start_epoch: int = 1,
              lr: float = 1e-3, val_frac: float = 0.2,
              device: str = "cpu", print_every: int = 50,
              ckpt_path=None, ckpt_every: int = 50,
              history=None) -> tuple[PedigreeGNN, dict]:
    """Train GNN with checkpointing and resume support."""
    model     = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    if history is None:
        history = {"train_loss": [], "val_loss": []}

    # Accept pre-built samples list or build from datasets
    if len(datasets_list) == 1 and "_samples" in datasets_list[0]:
        samples = datasets_list[0]["_samples"]
    else:
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

    # Group by shared edge_index (same dataset = same pedigree topology).
    # Within a group every graph has identical edge structure, only node features differ.
    # This lets us batch as (B, N, F) and do pure matrix ops — no per-sample Python loop.
    from collections import defaultdict

    # samples: list of (nf np.ndarray (N,F), ei np.ndarray (2,E), y float)
    # Key = canonical edge_index bytes (unique per topology)
    groups: dict[bytes, dict] = {}
    for nf, ei, y in samples:
        key = ei.tobytes()
        if key not in groups:
            groups[key] = {
                "ei": torch.LongTensor(ei).to(device),
                "nf": [], "y": [],
            }
        groups[key]["nf"].append(nf)
        groups[key]["y"].append(y)

    # Convert to tensors: nf → (M, N, F),  y → (M,)
    for g in groups.values():
        g["nf"] = torch.FloatTensor(np.stack(g["nf"])).to(device)  # (M, N, F)
        g["y"]  = torch.FloatTensor(g["y"]).to(device)             # (M,)

    def iter_batches(groups, batch_size, shuffle=True):
        for g in groups.values():
            M   = g["nf"].shape[0]
            idx = torch.randperm(M) if shuffle else torch.arange(M)
            for start in range(0, M, batch_size):
                sl = idx[start: start + batch_size]
                yield g["nf"][sl], g["ei"], g["y"][sl]

    n_total = sum(g["nf"].shape[0] for g in groups.values())
    n_val   = max(1, int(n_total * val_frac))
    n_train = n_total - n_val

    # Split each group into train/val
    train_groups, val_groups = {}, {}
    for key, g in groups.items():
        M     = g["nf"].shape[0]
        n_v   = max(1, int(M * val_frac))
        n_t   = M - n_v
        train_groups[key] = {"ei": g["ei"], "nf": g["nf"][:n_t], "y": g["y"][:n_t]}
        val_groups[key]   = {"ei": g["ei"], "nf": g["nf"][n_t:], "y": g["y"][n_t:]}

    batch_size = 256

    for epoch in range(start_epoch, epochs + 1):
        model.train()
        train_loss = 0.0
        for nf_b, ei_b, y_b in iter_batches(train_groups, batch_size, shuffle=True):
            optimizer.zero_grad()
            preds = model.forward_batch(nf_b, ei_b)
            loss  = criterion(preds, y_b)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * len(y_b)
        train_loss /= n_train

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for nf_b, ei_b, y_b in iter_batches(val_groups, batch_size, shuffle=False):
                val_loss += criterion(model.forward_batch(nf_b, ei_b), y_b).item() * len(y_b)
        val_loss /= n_val

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)

        if epoch % print_every == 0:
            print(f"  epoch {epoch:4d}/{epochs}  "
                  f"train_loss={train_loss:.6f}  val_loss={val_loss:.6f}")

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
    gen_states  = ds.get("two_gene_states")

    model.eval()
    memo: dict = {}

    def _net_val(state) -> float:
        if state in memo:
            return memo[state]
        nf, ei = state_to_graph(state, belief, individuals, pedigree, genes)
        with torch.no_grad():
            v = model(
                torch.FloatTensor(nf).to(device),
                torch.LongTensor(ei).to(device),
            ).item()
        memo[state] = v
        return v

    def _stop_val(per_gene, tested):
        # Use per_gene directly — avoids wrongly calling lift_tuple_posteriors_to_genes
        # on entry.marginals which is GeneA-only, not tuple posteriors.
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
            marg     = entry.marginals
            per_gene = entry.get_per_gene_probs()
            tuple_pmfs = entry.get_tuple_pmfs()
        else:
            marg = entry
            per_gene = lift_tuple_posteriors_to_genes(entry, genes, GENOTYPE_STATES)
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

        best_net_q  = v_stop
        best_person = None
        for i in individuals:
            if i in tested:
                continue
            r_i     = _test_r(i, per_gene)
            pmf_i   = _pmf_for_person(tuple_pmfs, marg, i)
            exp_net = 0.0
            for g, prob_g in pmf_i.items():
                if prob_g <= 1e-12:
                    continue
                next_s = frozenset(state | {(i, g)})
                if next_s not in belief:
                    continue
                exp_net += prob_g * _net_val(next_s)
            q_i = r_i + exp_net
            if q_i > best_net_q:
                best_net_q  = q_i
                best_person = i

        if best_person is None:
            true_memo[state] = v_stop
            return v_stop

        r_best   = _test_r(best_person, per_gene)
        pmf_best = _pmf_for_person(tuple_pmfs, marg, best_person)
        exp_true = 0.0
        for g, prob_g in pmf_best.items():
            if prob_g <= 1e-12:
                continue
            next_s = frozenset(state | {(best_person, g)})
            if next_s not in belief:
                continue
            exp_true += prob_g * value_at(next_s)

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
    model = PedigreeGNN(node_feat_dim=7, hidden_dim=32, n_rounds=2)

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
        # Load train datasets into sample list
        samples = []
        for cfg in CONFIGS:
            key = cfg_key(cfg)
            if key not in train_keys: continue
            with open(cache_dir / f"{key}.pkl", "rb") as f: ds = pickle.load(f)
            individuals = ds["individuals"]
            pedigree    = ds["pedigree"]
            belief      = ds["belief"]
            genes       = ds.get("genes", ("GeneA", "GeneB"))
            for state, v_star in ds["V_star"].items():
                entry = belief[state]
                marg  = entry.marginals if isinstance(entry, InferenceResult) else entry
                if sum(marg[individuals[0]].values()) < 1e-9: continue
                nf, ei = state_to_graph(state, belief, individuals, pedigree, genes)
                samples.append((nf, ei, float(v_star)))
            del ds
        log(f"    Train samples: {len(samples)}")

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
