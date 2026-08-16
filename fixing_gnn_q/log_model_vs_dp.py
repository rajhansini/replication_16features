"""Trajectory comparison: CE-retrained GNN-Q / MLP-Q vs GROUND TRUTH
(exact DP-optimal policy), 2-gene and 3-gene.

log_myopic_vs_gnn.py compares against myopic, which is a strong heuristic
reference but NOT the optimal policy. This script compares against
ds["policy_dp"] — the actual exact-DP-optimal action at every state,
precomputed during the original DP solve. This is the real "how many
actions did the model get right" number, walked along the MODEL'S OWN
greedy trajectory (most-likely outcome under the actions the model itself
would take) — not just the first (root) action.

Outputs:
    results/{2gene|3gene}/{ce|ce_reweighted}/{gnn|mlp}/seed{S}/model_vs_dp.log
    results/{2gene|3gene}/{ce|ce_reweighted}/{gnn|mlp}/seed{S}/model_vs_dp.json

Usage:
    python log_model_vs_dp.py --genes 3 --seed 0 --model gnn
    python log_model_vs_dp.py --genes 2 --seed 0 --model mlp --variant ce_reweighted
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import pickle
import sys
from datetime import datetime
from pathlib import Path

import torch

HERE     = Path(__file__).resolve().parent
ROOT     = HERE.parent
EXP_ROOT = ROOT / "experiments_after_understanding"
Q_DIR    = EXP_ROOT / "q_learning"
EXPERIMENTS = ROOT / "ground-up-experiments"

sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(EXPERIMENTS))
sys.path.insert(0, str(EXP_ROOT))
sys.path.insert(0, str(Q_DIR))
sys.path.insert(0, str(HERE))


def _load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


from exputils.eval import _get_entry, _stop_val  # noqa: E402
from genetic_dp.utils.pedigree_generator import generate_deterministic_pedigree  # noqa: E402

RESULTS_BASE = HERE / "results"


def fmt_state(state):
    if not state:
        return "root"
    return "{" + ", ".join(f"{p}:{g}" for p, g in sorted(state)) + "}"


def fmt_pick(kind, person):
    return "STOP" if kind == "STOP" else f"TEST {person}"


def most_likely_outcome(state, person, belief, genes):
    _, tuple_pmfs = _get_entry(belief, state, genes)
    pmf = tuple_pmfs.get(person, {})
    if not pmf:
        raise RuntimeError(f"empty pmf for {person} at {fmt_state(state)}")
    return max(pmf.items(), key=lambda kv: kv[1])[0]


def model_pick(state, q_hat, individuals, belief, genes, config):
    tested = {p for p, _ in state}
    untested = [p for p in individuals if p not in tested]
    if not untested:
        return "STOP", None
    per_gene, _ = _get_entry(belief, state, genes)
    v_stop = _stop_val(per_gene, individuals, tested, config)
    q_vals = q_hat[state]
    best = max(untested, key=lambda p: q_vals[p])
    if q_vals[best] <= v_stop:
        return "STOP", None
    return "TEST", best


def dp_pick(state, policy_dp):
    action, person, _ = policy_dp[state]
    if action == "stop":
        return "STOP", None
    return "TEST", person


def walk_and_log(state0, q_hat, policy_dp, individuals, belief, genes, config, log, model_label, max_steps=20):
    """Walk the MODEL's own most-likely-outcome trajectory (the path it would
    actually take), logging the exact-DP-optimal pick at every visited state.
    """
    state = state0
    n_match = n_total = 0
    rows = []
    log(f"  {'step':<5}{'state':<50}{model_label.upper():<16}{'DP (ground truth)':<20}{'match'}")
    for step in range(max_steps):
        gk, gp = model_pick(state, q_hat, individuals, belief, genes, config)
        dk, dp = dp_pick(state, policy_dp)
        g_str, d_str = fmt_pick(gk, gp), fmt_pick(dk, dp)
        match = (gk == dk) and (gp == dp)
        n_total += 1
        n_match += int(match)
        log(f"  {step:<5}{fmt_state(state):<50}{g_str:<16}{d_str:<20}{'YES' if match else 'NO'}")
        rows.append({"step": step, "state": fmt_state(state), model_label: g_str, "dp": d_str, "match": match})

        if gk == "STOP":
            break
        outcome = most_likely_outcome(state, gp, belief, genes)
        nxt = frozenset(state | {(gp, outcome)})
        if nxt not in belief:
            log(f"  (next state missing from belief cache — stopping walk)")
            break
        state = nxt

    rate = n_match / n_total if n_total else 0.0
    log(f"  agreement with DP (ground truth): {n_match}/{n_total} = {rate:.3f}")
    return {"n_match": n_match, "n_total": n_total, "rate": rate, "steps": rows}


# ── 3-gene setup ─────────────────────────────────────────────────────────────

VARIANT_3GENE = {
    ("gnn", "ce"):            ("gnn_q_ce", "gnn_q_ce.pt", "train_gnn_q_ce.py"),
    ("gnn", "ce_reweighted"): ("gnn_q_ce_reweighted", "gnn_q_ce_rw.pt", "train_gnn_q_ce_reweighted.py"),
    ("mlp", "ce"):            ("mlp_q_ce", "mlp_q_ce.pt", "train_mlp_q_ce.py"),
    ("mlp", "ce_reweighted"): ("mlp_q_ce_reweighted", "mlp_q_ce_rw.pt", "train_mlp_q_ce_reweighted.py"),
}
VARIANT_2GENE = {
    ("gnn", "ce"):            ("gnn_q_ce_2gene", "gnn_q_ce_2gene.pt", "train_two_gene_gnn_q_ce.py"),
    ("gnn", "ce_reweighted"): ("gnn_q_ce_reweighted_2gene", "gnn_q_ce_rw_2gene.pt", "train_two_gene_gnn_q_ce_reweighted.py"),
    ("mlp", "ce"):            ("mlp_q_ce_2gene", "mlp_q_ce_2gene.pt", "train_two_gene_mlp_q_ce.py"),
    ("mlp", "ce_reweighted"): ("mlp_q_ce_reweighted_2gene", "mlp_q_ce_rw_2gene.pt", "train_two_gene_mlp_q_ce_reweighted.py"),
}


def run_3gene(seed, device, log, out_dir, variant, model_name):
    gnn_mod = _load_module("gnn_run_mvd3", EXP_ROOT / "gnn" / "run.py")
    mlp_mod = _load_module("mlp_run_mvd3", EXP_ROOT / "mlp" / "run.py")
    q_mod   = _load_module("q_mvd3", Q_DIR / ("gnn_q.py" if model_name == "gnn" else "mlp_q.py"))
    from qsa_data import build_qsa_index  # noqa: F401

    fam_mod = gnn_mod if model_name == "gnn" else mlp_mod

    subdir, fname, trainer = VARIANT_3GENE[(model_name, variant)]
    wt = HERE / "results" / subdir / "seed_runs" / f"seed{seed}" / fname
    if not wt.exists():
        raise FileNotFoundError(f"missing {model_name}/{variant} weights: {wt} — run {trainer} first")

    model = (q_mod.GNNQ() if model_name == "gnn" else q_mod.MLPQ()).to(device)
    model.load_state_dict(torch.load(wt, map_location=device))
    model.eval()
    log(f"loaded {wt}  params={sum(p.numel() for p in model.parameters())}")

    struct_cache, edge_cache = {}, {}
    for fam in fam_mod.TRAIN_FAMILIES + fam_mod.TEST_FAMILIES:
        sample_key = f"{fam}_LowHigh_Base_3gene"
        with open(fam_mod.CACHE_DIR / f"{sample_key}.pkl", "rb") as f:
            sample_ds = pickle.load(f)
        pedigree = generate_deterministic_pedigree(fam_mod.FAMILY_CASES[fam])
        struct_cache[fam] = fam_mod.compute_structural_features(pedigree, sample_ds["individuals"])
        if model_name == "gnn":
            edge_cache[fam] = fam_mod.build_edge_index(pedigree, sample_ds["individuals"])

    summary = {}
    for key in fam_mod.TEST_KEYS:
        fam = key.split("_")[0]
        log(f"\n{'='*80}\n{key}\n{'='*80}")
        with open(fam_mod.CACHE_DIR / f"{key}.pkl", "rb") as f:
            ds = pickle.load(f)

        if model_name == "gnn":
            base = fam_mod.load_dataset(key, struct_cache[fam], edge_cache[fam], device)
            ds["_nf"], ds["_gf"] = base["nf"], base["gf"]
            ei = torch.tensor(edge_cache[fam], device=device)
            q_hat = q_mod.precompute_qhat(model, ds, key, ei, device)
        else:
            base = fam_mod.load_dataset(key, struct_cache[fam], device)
            ds["_nf"], ds["_gf"] = base["nf"], base["gf"]
            q_hat = q_mod.precompute_qhat(model, ds, key, device)

        individuals = ds["individuals"]
        belief = ds["belief"]
        genes = ds.get("genes", fam_mod.GENES)
        config = ds["config"]
        policy_dp = ds["policy_dp"]

        summary[key] = walk_and_log(frozenset(), q_hat, policy_dp, individuals, belief, genes, config, log, model_name)

    return summary


# ── 2-gene setup ─────────────────────────────────────────────────────────────

def run_2gene(seed, device, log, out_dir, variant, model_name):
    tg    = _load_module("two_gene_run_mvd2", EXP_ROOT / "two_gene" / "run.py")
    q_mod = _load_module("two_gene_q_mvd2", Q_DIR / ("two_gene_gnn_q.py" if model_name == "gnn" else "two_gene_mlp_q.py"))

    subdir, fname, trainer = VARIANT_2GENE[(model_name, variant)]
    wt = HERE / "results" / subdir / "seed_runs" / f"seed{seed}" / fname
    if not wt.exists():
        raise FileNotFoundError(f"missing {model_name}/{variant} weights: {wt} — run {trainer} first")

    model = (q_mod.GNNQ() if model_name == "gnn" else q_mod.MLPQ()).to(device)
    model.load_state_dict(torch.load(wt, map_location=device))
    model.eval()
    log(f"loaded {wt}  params={sum(p.numel() for p in model.parameters())}")

    struct_cache, edge_cache = {}, {}
    for fam in tg.TRAIN_FAMILIES + tg.TEST_FAMILIES:
        pedigree = generate_deterministic_pedigree(tg.FAMILY_CASES[fam])
        individuals = pedigree.to_list()
        struct_cache[fam] = tg.compute_structural_features(pedigree, individuals)
        if model_name == "gnn":
            edge_cache[fam] = tg.build_edge_index(pedigree, individuals)

    summary = {}
    for fam, reg, pre in tg.TEST_CONFIGS:
        key = f"{fam}_{reg}_{pre}_2gene"
        log(f"\n{'='*80}\n{key}\n{'='*80}")
        ds = q_mod.build_config(fam, reg, pre)
        base = tg.ds_to_tensors(ds, struct_cache[fam], device)
        ds["_nf"], ds["_gf"] = base["nf"], base["gf"]

        if model_name == "gnn":
            ei = torch.tensor(edge_cache[fam], device=device)
            q_hat = q_mod.precompute_qhat(model, ds, key, ei, device)
        else:
            q_hat = q_mod.precompute_qhat(model, ds, key, device)

        individuals = ds["individuals"]
        belief = ds["belief"]
        genes = ds.get("genes", tg.GENES)
        config = ds["config"]
        policy_dp = ds["policy_dp"]

        summary[key] = walk_and_log(frozenset(), q_hat, policy_dp, individuals, belief, genes, config, log, model_name)

    return summary


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--genes", type=int, required=True, choices=[2, 3])
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="cpu")
    p.add_argument("--variant", default="ce", choices=["ce", "ce_reweighted"])
    p.add_argument("--model", default="gnn", choices=["gnn", "mlp"])
    args = p.parse_args()

    tag = f"{args.genes}gene"
    out_dir = RESULTS_BASE / tag / args.variant / args.model / f"seed{args.seed}"
    out_dir.mkdir(parents=True, exist_ok=True)
    log_path = out_dir / "model_vs_dp.log"
    log_f = open(log_path, "w")

    def log(msg=""):
        print(msg, flush=True)
        log_f.write(msg + "\n")
        log_f.flush()

    log(f"[model_vs_dp] {datetime.now().isoformat()}  genes={args.genes}  seed={args.seed}"
        f"  variant={args.variant}  model={args.model}")
    log(f"output -> {out_dir}")

    device = torch.device(args.device)
    if args.genes == 3:
        summary = run_3gene(args.seed, device, log, out_dir, args.variant, args.model)
    else:
        summary = run_2gene(args.seed, device, log, out_dir, args.variant, args.model)

    n_match = sum(v["n_match"] for v in summary.values())
    n_total = sum(v["n_total"] for v in summary.values())
    rate = n_match / n_total if n_total else 0.0
    root_match = sum(1 for v in summary.values() if v["steps"] and v["steps"][0]["match"])
    root_total = sum(1 for v in summary.values() if v["steps"])

    log(f"\n{'='*80}\nOVERALL")
    log(f"  agreement with DP (ground truth): {n_match}/{n_total} = {rate:.3f}")
    log(f"  root-state agreement:             {root_match}/{root_total} = {(root_match/root_total if root_total else 0):.3f}")

    payload = {
        "genes": args.genes, "seed": args.seed, "variant": args.variant, "model": args.model,
        "overall": {"n_match": n_match, "n_total": n_total, "rate": rate,
                    "root_match": root_match, "root_total": root_total},
        "per_config": {k: {"n_match": v["n_match"], "n_total": v["n_total"], "rate": v["rate"]}
                        for k, v in summary.items()},
        "detail": summary,
    }
    (out_dir / "model_vs_dp.json").write_text(json.dumps(payload, indent=2))
    log(f"saved -> {out_dir/'model_vs_dp.json'}")
    log(f"saved -> {log_path}")
    log_f.close()


if __name__ == "__main__":
    main()
