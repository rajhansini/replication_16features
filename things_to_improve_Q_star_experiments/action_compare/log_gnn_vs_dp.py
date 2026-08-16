"""Concise GNN-Q vs exact-DP action comparison logs (2-gene and 3-gene).

Loads already-trained GNN-Q weights (no retraining). For every held-out test
config, walks TWO most-likely-outcome trajectories and logs one line per step:

  (A) follow GNN's greedy policy  — what GNN actually did, vs what DP wanted
  (B) follow DP's optimal policy  — what DP did, vs what GNN would have done

Format (one line per decision):
    step  state={...}  GNN=TEST Child  DP=TEST Father  match=NO

Outputs (never touches existing q_learning results):
    results/{2gene|3gene}/seed{S}/action_compare.log
    results/{2gene|3gene}/seed{S}/action_compare.json   # match rates

Usage:
    python log_gnn_vs_dp.py --genes 3 --seed 0
    python log_gnn_vs_dp.py --genes 2 --seed 0
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
ABLATION = HERE.parent
ROOT     = ABLATION.parent
EXP_ROOT = ROOT / "experiments_after_understanding"
Q_DIR    = EXP_ROOT / "q_learning"
EXPERIMENTS = ROOT / "ground-up-experiments"

sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(EXPERIMENTS))
sys.path.insert(0, str(EXP_ROOT))
sys.path.insert(0, str(Q_DIR))


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


def gnn_pick(state, q_hat, individuals, belief, genes, config):
    """Greedy GNN-Q action at `state` using precomputed q_hat + exact v_stop."""
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


def walk_and_log(traj_name, drive, state0, q_hat, policy_dp, individuals,
                 belief, genes, config, log, max_steps=20):
    """Walk most-likely-outcome path driven by `drive` in {'GNN','DP'}.

    At every visited state, log GNN pick vs DP pick. Advance by the driver's
    chosen person (most-likely genotype), or stop.
    """
    state = state0
    n_match = n_total = 0
    rows = []
    log(f"\n  --- trajectory: follow {drive} (most-likely outcomes) ---")
    log(f"  {'step':<5}{'state':<55}{'GNN':<18}{'DP':<18}{'match'}")
    for step in range(max_steps):
        gk, gp = gnn_pick(state, q_hat, individuals, belief, genes, config)
        dk, dp = dp_pick(state, policy_dp)
        g_str = fmt_pick(gk, gp)
        d_str = fmt_pick(dk, dp)
        match = (gk == dk) and (gp == dp)
        n_total += 1
        n_match += int(match)
        log(f"  {step:<5}{fmt_state(state):<55}{g_str:<18}{d_str:<18}{'YES' if match else 'NO'}")
        rows.append({
            "step": step, "state": fmt_state(state),
            "gnn": g_str, "dp": d_str, "match": match,
        })
        # advance by driver's action
        kind, person = (gk, gp) if drive == "GNN" else (dk, dp)
        if kind == "STOP":
            break
        outcome = most_likely_outcome(state, person, belief, genes)
        nxt = frozenset(state | {(person, outcome)})
        if nxt not in belief:
            log(f"  (next state missing from belief cache — stopping walk)")
            break
        state = nxt
    rate = n_match / n_total if n_total else 0.0
    log(f"  agreement on this trajectory: {n_match}/{n_total} = {rate:.3f}")
    return {"n_match": n_match, "n_total": n_total, "rate": rate, "steps": rows}


# ── 3-gene setup ─────────────────────────────────────────────────────────────

def run_3gene(seed, device, log, out_dir):
    gnn_mod = _load_module("gnn_run_actcmp3", EXP_ROOT / "gnn" / "run.py")
    gnn_q   = _load_module("gnn_q_actcmp3", Q_DIR / "gnn_q.py")
    from qsa_data import build_qsa_index  # noqa: F401 — ensure cache path resolved

    wt = Q_DIR / "results" / "seed_runs" / f"seed{seed}" / "gnn_q.pt"
    if not wt.exists():
        raise FileNotFoundError(f"missing weights: {wt}")

    model = gnn_q.GNNQ().to(device)
    model.load_state_dict(torch.load(wt, map_location=device))
    model.eval()
    log(f"loaded {wt}  params={sum(p.numel() for p in model.parameters())}")

    struct_cache, edge_cache = {}, {}
    for fam in gnn_mod.TRAIN_FAMILIES + gnn_mod.TEST_FAMILIES:
        sample_key = f"{fam}_LowHigh_Base_3gene"
        with open(gnn_mod.CACHE_DIR / f"{sample_key}.pkl", "rb") as f:
            sample_ds = pickle.load(f)
        pedigree = generate_deterministic_pedigree(gnn_mod.FAMILY_CASES[fam])
        struct_cache[fam] = gnn_mod.compute_structural_features(pedigree, sample_ds["individuals"])
        edge_cache[fam]   = gnn_mod.build_edge_index(pedigree, sample_ds["individuals"])

    summary = {}
    for key in gnn_mod.TEST_KEYS:
        fam = key.split("_")[0]
        log(f"\n{'='*80}\n{key}\n{'='*80}")
        with open(gnn_mod.CACHE_DIR / f"{key}.pkl", "rb") as f:
            ds = pickle.load(f)
        base = gnn_mod.load_dataset(key, struct_cache[fam], edge_cache[fam], device)
        ds["_nf"], ds["_gf"] = base["nf"], base["gf"]
        ei = torch.tensor(edge_cache[fam], device=device)

        q_hat = gnn_q.precompute_qhat(model, ds, key, ei, device)
        individuals = ds["individuals"]
        belief = ds["belief"]
        genes = ds.get("genes", gnn_mod.GENES)
        config = ds["config"]
        policy_dp = ds["policy_dp"]

        a = walk_and_log("GNN", "GNN", frozenset(), q_hat, policy_dp,
                         individuals, belief, genes, config, log)
        b = walk_and_log("DP",  "DP",  frozenset(), q_hat, policy_dp,
                         individuals, belief, genes, config, log)
        summary[key] = {
            "follow_gnn": {"n_match": a["n_match"], "n_total": a["n_total"], "rate": a["rate"]},
            "follow_dp":  {"n_match": b["n_match"], "n_total": b["n_total"], "rate": b["rate"]},
            "steps_follow_gnn": a["steps"],
            "steps_follow_dp":  b["steps"],
        }

    return summary


# ── 2-gene setup ─────────────────────────────────────────────────────────────

def run_2gene(seed, device, log, out_dir):
    tg    = _load_module("two_gene_run_actcmp", EXP_ROOT / "two_gene" / "run.py")
    gnn_q = _load_module("two_gene_gnn_q_actcmp", Q_DIR / "two_gene_gnn_q.py")

    wt = Q_DIR / "results_2gene" / "seed_runs" / f"seed{seed}" / "gnn_q_2gene.pt"
    if not wt.exists():
        raise FileNotFoundError(f"missing weights: {wt}")

    model = gnn_q.GNNQ().to(device)
    model.load_state_dict(torch.load(wt, map_location=device))
    model.eval()
    log(f"loaded {wt}  params={sum(p.numel() for p in model.parameters())}")

    struct_cache, edge_cache = {}, {}
    for fam in tg.TRAIN_FAMILIES + tg.TEST_FAMILIES:
        pedigree = generate_deterministic_pedigree(tg.FAMILY_CASES[fam])
        individuals = pedigree.to_list()
        struct_cache[fam] = tg.compute_structural_features(pedigree, individuals)
        edge_cache[fam]   = tg.build_edge_index(pedigree, individuals)

    summary = {}
    for fam, reg, pre in tg.TEST_CONFIGS:
        key = f"{fam}_{reg}_{pre}_2gene"
        log(f"\n{'='*80}\n{key}\n{'='*80}")
        ds = gnn_q.build_config(fam, reg, pre)
        base = tg.ds_to_tensors(ds, struct_cache[fam], device)
        ds["_nf"], ds["_gf"] = base["nf"], base["gf"]
        ei = torch.tensor(edge_cache[fam], device=device)

        q_hat = gnn_q.precompute_qhat(model, ds, key, ei, device)
        individuals = ds["individuals"]
        belief = ds["belief"]
        genes = ds.get("genes", tg.GENES)
        config = ds["config"]
        policy_dp = ds["policy_dp"]

        a = walk_and_log("GNN", "GNN", frozenset(), q_hat, policy_dp,
                         individuals, belief, genes, config, log)
        b = walk_and_log("DP",  "DP",  frozenset(), q_hat, policy_dp,
                         individuals, belief, genes, config, log)
        summary[key] = {
            "follow_gnn": {"n_match": a["n_match"], "n_total": a["n_total"], "rate": a["rate"]},
            "follow_dp":  {"n_match": b["n_match"], "n_total": b["n_total"], "rate": b["rate"]},
            "steps_follow_gnn": a["steps"],
            "steps_follow_dp":  b["steps"],
        }

    return summary


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--genes", type=int, required=True, choices=[2, 3])
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="cpu")
    args = p.parse_args()

    tag = f"{args.genes}gene"
    out_dir = RESULTS_BASE / tag / f"seed{args.seed}"
    out_dir.mkdir(parents=True, exist_ok=True)
    log_path = out_dir / "action_compare.log"
    log_f = open(log_path, "w")

    def log(msg=""):
        print(msg, flush=True)
        log_f.write(msg + "\n")
        log_f.flush()

    log(f"[action_compare] {datetime.now().isoformat()}  genes={args.genes}  seed={args.seed}")
    log(f"output -> {out_dir}")

    device = torch.device(args.device)
    if args.genes == 3:
        summary = run_3gene(args.seed, device, log, out_dir)
    else:
        summary = run_2gene(args.seed, device, log, out_dir)

    # overall rates
    def _agg(field):
        m = sum(v[field]["n_match"] for v in summary.values())
        t = sum(v[field]["n_total"] for v in summary.values())
        return m, t, (m / t if t else 0.0)

    gm, gt, gr = _agg("follow_gnn")
    dm, dt, dr = _agg("follow_dp")
    log(f"\n{'='*80}\nOVERALL")
    log(f"  follow-GNN traj agreement: {gm}/{gt} = {gr:.3f}")
    log(f"  follow-DP  traj agreement: {dm}/{dt} = {dr:.3f}")

    payload = {
        "genes": args.genes, "seed": args.seed,
        "overall": {
            "follow_gnn": {"n_match": gm, "n_total": gt, "rate": gr},
            "follow_dp":  {"n_match": dm, "n_total": dt, "rate": dr},
        },
        "per_config": {
            k: {"follow_gnn": v["follow_gnn"], "follow_dp": v["follow_dp"]}
            for k, v in summary.items()
        },
        "detail": summary,
    }
    (out_dir / "action_compare.json").write_text(json.dumps(payload, indent=2))
    log(f"saved -> {out_dir/'action_compare.json'}")
    log(f"saved -> {log_path}")
    log_f.close()


if __name__ == "__main__":
    main()
