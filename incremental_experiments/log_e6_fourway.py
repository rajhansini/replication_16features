"""E6 four-way action log: DP (ground truth) vs MYOPIC (Kanix's canonical
myopic_greedy) vs GNN-Q vs MLP-Q, both using the E6 sum-pooling checkpoints
-- unlike E4/E5, MLP-Q's checkpoint changes here too, since the pooling
readout applies to both architectures identically (not GNN-specific like
bidirectional message passing was).

Same walk-and-log structure as log_e4_fourway.py / log_e5_fourway.py.

Outputs:
    results/{2gene|3gene}/seed{S}/e6_fourway.log
    results/{2gene|3gene}/seed{S}/e6_fourway.json

Usage:
    python log_e6_fourway.py --genes 3 --seed 0
    python log_e6_fourway.py --genes 2 --seed 0
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


def _load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


from exputils.eval import _get_entry, _stop_val  # noqa: E402
from genetic_dp.utils.pedigree_generator import generate_deterministic_pedigree  # noqa: E402
from genetic_dp.policy.baselines import myopic_greedy  # noqa: E402
from genetic_dp.exact_dp.utils import GENOTYPE_STATES  # noqa: E402

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


def myopic_pick(state, individuals, belief, genes, config, v_stop_cache=None):
    """Kanix's canonical myopic policy — genetic_dp.policy.baselines.myopic_greedy
    (zero-lookahead: argmax_i immediate r_test(i,s) vs stop)."""
    action, who, _ = myopic_greedy(
        state,
        belief=belief, individuals=individuals, gen_states=GENOTYPE_STATES,
        infer=None,
        a=config.a, b=config.b, c=config.c, delta=config.delta,
        fixed_cost=config.fixed_cost, variable_cost=config.variable_cost,
        genes=genes,
        a_gene=config.a_gene if config.a_gene else None,
        b_gene=config.b_gene if config.b_gene else None,
        c_gene=config.c_gene if config.c_gene else None,
        delta_gene=config.delta_gene if config.delta_gene else None,
        tuple_mode=bool(genes),
    )
    if action == "stop":
        return "STOP", None
    return "TEST", who


def dp_pick(state, policy_dp):
    action, person, _ = policy_dp[state]
    if action == "stop":
        return "STOP", None
    return "TEST", person


def walk_and_log(state0, q_hat_gnn, q_hat_mlp, policy_dp, individuals, belief,
                  genes, config, log, max_steps=20):
    """Walk MYOPIC's most-likely-outcome trajectory, logging DP / MYOPIC /
    GNN / MLP picks together at every visited state."""
    state = state0
    n_total = 0
    n_dp_myo = n_dp_gnn = n_dp_mlp = n_all4 = 0
    rows = []
    log(f"  {'step':<5}{'state':<50}{'DP (truth)':<18}{'MYOPIC':<16}{'GNN':<16}{'MLP':<16}{'agree'}")
    for step in range(max_steps):
        yk, yp = myopic_pick(state, individuals, belief, genes, config)
        dk, dp = dp_pick(state, policy_dp)
        gk, gp = model_pick(state, q_hat_gnn, individuals, belief, genes, config)
        mk, mp = model_pick(state, q_hat_mlp, individuals, belief, genes, config)

        d_str, y_str = fmt_pick(dk, dp), fmt_pick(yk, yp)
        g_str, m_str = fmt_pick(gk, gp), fmt_pick(mk, mp)

        dp_myo = (dk == yk) and (dp == yp)
        dp_gnn = (dk == gk) and (dp == gp)
        dp_mlp = (dk == mk) and (dp == mp)
        all4 = dp_myo and dp_gnn and dp_mlp

        n_total += 1
        n_dp_myo += int(dp_myo); n_dp_gnn += int(dp_gnn); n_dp_mlp += int(dp_mlp)
        n_all4 += int(all4)

        agree = "ALL" if all4 else ("none" if not (dp_myo or dp_gnn or dp_mlp) else "partial")
        log(f"  {step:<5}{fmt_state(state):<50}{d_str:<18}{y_str:<16}{g_str:<16}{m_str:<16}{agree}")
        rows.append({
            "step": step, "state": fmt_state(state),
            "dp": d_str, "myopic": y_str, "gnn": g_str, "mlp": m_str,
            "dp_myopic_match": dp_myo, "dp_gnn_match": dp_gnn, "dp_mlp_match": dp_mlp, "all4_match": all4,
        })

        if yk == "STOP":
            break
        outcome = most_likely_outcome(state, yp, belief, genes)
        nxt = frozenset(state | {(yp, outcome)})
        if nxt not in belief:
            log("  (next state missing from belief cache — stopping walk)")
            break
        state = nxt

    def rate(n):
        return n / n_total if n_total else 0.0

    log(f"  agreement vs DP: myopic={n_dp_myo}/{n_total}={rate(n_dp_myo):.3f}  "
        f"gnn={n_dp_gnn}/{n_total}={rate(n_dp_gnn):.3f}  mlp={n_dp_mlp}/{n_total}={rate(n_dp_mlp):.3f}  "
        f"all4={n_all4}/{n_total}={rate(n_all4):.3f}")
    return {
        "n_total": n_total,
        "n_dp_myopic": n_dp_myo, "n_dp_gnn": n_dp_gnn, "n_dp_mlp": n_dp_mlp, "n_all4": n_all4,
        "rate_dp_myopic": rate(n_dp_myo), "rate_dp_gnn": rate(n_dp_gnn),
        "rate_dp_mlp": rate(n_dp_mlp), "rate_all4": rate(n_all4),
        "steps": rows,
    }


# ── 3-gene setup ─────────────────────────────────────────────────────────────

def run_3gene(seed, device, log):
    gnn_mod = _load_module("gnn_run_e6_3", EXP_ROOT / "gnn" / "run.py")
    mlp_mod = _load_module("mlp_run_e6_3", EXP_ROOT / "mlp" / "run.py")
    gnn_q   = _load_module("gnn_q_e6_sumpool_3", HERE / "e6_train_gnn_q.py")
    mlp_q   = _load_module("mlp_q_e6_sumpool_3", HERE / "e6_train_mlp_q.py")
    from qsa_data import build_qsa_index  # noqa: F401

    gnn_wt = HERE / "results" / "e6_gnn_sumpool" / "seed_runs" / f"seed{seed}" / "gnn_q_e6.pt"
    mlp_wt = HERE / "results" / "e6_mlp_sumpool" / "seed_runs" / f"seed{seed}" / "mlp_q_e6.pt"
    if not gnn_wt.exists():
        raise FileNotFoundError(f"missing E6 GNN weights: {gnn_wt} — run e6_train_gnn_q.py first")
    if not mlp_wt.exists():
        raise FileNotFoundError(f"missing E6 MLP weights: {mlp_wt} — run e6_train_mlp_q.py first")

    gmodel = gnn_q.GNNQBidirSumPool().to(device)
    gmodel.load_state_dict(torch.load(gnn_wt, map_location=device))
    gmodel.eval()
    log(f"loaded {gnn_wt}")

    mmodel = mlp_q.MLPQSumPool().to(device)
    mmodel.load_state_dict(torch.load(mlp_wt, map_location=device))
    mmodel.eval()
    log(f"loaded {mlp_wt}")

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
        log(f"\n{'='*90}\n{key}\n{'='*90}")
        with open(gnn_mod.CACHE_DIR / f"{key}.pkl", "rb") as f:
            ds = pickle.load(f)

        base_g = gnn_mod.load_dataset(key, struct_cache[fam], edge_cache[fam], device)
        ds["_nf"], ds["_gf"] = base_g["nf"], base_g["gf"]
        ei = torch.tensor(edge_cache[fam], device=device)
        q_hat_gnn = gnn_q.precompute_qhat(gmodel, ds, key, ei, device)

        base_m = mlp_mod.load_dataset(key, struct_cache[fam], device)
        ds["_nf"], ds["_gf"] = base_m["nf"], base_m["gf"]
        q_hat_mlp = mlp_q.precompute_qhat(mmodel, ds, key, device)

        individuals = ds["individuals"]
        belief = ds["belief"]
        genes = ds.get("genes", gnn_mod.GENES)
        config = ds["config"]
        policy_dp = ds["policy_dp"]

        summary[key] = walk_and_log(frozenset(), q_hat_gnn, q_hat_mlp, policy_dp,
                                     individuals, belief, genes, config, log)

    return summary


# ── 2-gene setup ─────────────────────────────────────────────────────────────

def run_2gene(seed, device, log):
    tg      = _load_module("two_gene_run_e6_2", EXP_ROOT / "two_gene" / "run.py")
    gnn_q   = _load_module("two_gene_gnn_q_e6_sumpool_2", HERE / "e6_train_two_gene_gnn_q.py")
    mlp_q   = _load_module("two_gene_mlp_q_e6_sumpool_2", HERE / "e6_train_two_gene_mlp_q.py")

    gnn_wt = HERE / "results" / "e6_gnn_sumpool_2gene" / "seed_runs" / f"seed{seed}" / "gnn_q_e6_2gene.pt"
    mlp_wt = HERE / "results" / "e6_mlp_sumpool_2gene" / "seed_runs" / f"seed{seed}" / "mlp_q_e6_2gene.pt"
    if not gnn_wt.exists():
        raise FileNotFoundError(f"missing E6 GNN weights: {gnn_wt} — run e6_train_two_gene_gnn_q.py first")
    if not mlp_wt.exists():
        raise FileNotFoundError(f"missing E6 MLP weights: {mlp_wt} — run e6_train_two_gene_mlp_q.py first")

    gmodel = gnn_q.GNNQBidirSumPool().to(device)
    gmodel.load_state_dict(torch.load(gnn_wt, map_location=device))
    gmodel.eval()
    log(f"loaded {gnn_wt}")

    mmodel = mlp_q.MLPQSumPool().to(device)
    mmodel.load_state_dict(torch.load(mlp_wt, map_location=device))
    mmodel.eval()
    log(f"loaded {mlp_wt}")

    struct_cache, edge_cache = {}, {}
    for fam in tg.TRAIN_FAMILIES + tg.TEST_FAMILIES:
        pedigree = generate_deterministic_pedigree(tg.FAMILY_CASES[fam])
        individuals = pedigree.to_list()
        struct_cache[fam] = tg.compute_structural_features(pedigree, individuals)
        edge_cache[fam]   = tg.build_edge_index(pedigree, individuals)

    summary = {}
    for fam, reg, pre in tg.TEST_CONFIGS:
        key = f"{fam}_{reg}_{pre}_2gene"
        log(f"\n{'='*90}\n{key}\n{'='*90}")

        ds_g = gnn_q.build_config(fam, reg, pre)
        base_g = tg.ds_to_tensors(ds_g, struct_cache[fam], device)
        ds_g["_nf"], ds_g["_gf"] = base_g["nf"], base_g["gf"]
        ei = torch.tensor(edge_cache[fam], device=device)
        q_hat_gnn = gnn_q.precompute_qhat(gmodel, ds_g, key, ei, device)

        ds_m = mlp_q.build_config(fam, reg, pre)
        base_m = tg.ds_to_tensors(ds_m, struct_cache[fam], device)
        ds_m["_nf"], ds_m["_gf"] = base_m["nf"], base_m["gf"]
        q_hat_mlp = mlp_q.precompute_qhat(mmodel, ds_m, key, device)

        individuals = ds_g["individuals"]
        belief = ds_g["belief"]
        genes = ds_g.get("genes", tg.GENES)
        config = ds_g["config"]
        policy_dp = ds_g["policy_dp"]

        summary[key] = walk_and_log(frozenset(), q_hat_gnn, q_hat_mlp, policy_dp,
                                     individuals, belief, genes, config, log)

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
    log_path = out_dir / "e6_fourway.log"
    log_f = open(log_path, "w")

    def log(msg=""):
        print(msg, flush=True)
        log_f.write(msg + "\n")
        log_f.flush()

    log(f"[e6_fourway] {datetime.now().isoformat()}  genes={args.genes}  seed={args.seed}"
        f"  variant=E6 (sum pooling, both GNN and MLP)")

    device = torch.device(args.device)
    if args.genes == 3:
        summary = run_3gene(args.seed, device, log)
    else:
        summary = run_2gene(args.seed, device, log)

    def agg(field, total_field="n_total"):
        n = sum(v[field] for v in summary.values())
        t = sum(v[total_field] for v in summary.values())
        return n, t, (n / t if t else 0.0)

    dm, dt, dr = agg("n_dp_myopic")
    gm, gt, gr = agg("n_dp_gnn")
    mm, mt, mr = agg("n_dp_mlp")
    am, at, ar = agg("n_all4")
    root_dp_myo = sum(1 for v in summary.values() if v["steps"] and v["steps"][0]["dp_myopic_match"])
    root_dp_gnn = sum(1 for v in summary.values() if v["steps"] and v["steps"][0]["dp_gnn_match"])
    root_dp_mlp = sum(1 for v in summary.values() if v["steps"] and v["steps"][0]["dp_mlp_match"])
    root_total  = sum(1 for v in summary.values() if v["steps"])

    log(f"\n{'='*90}\nOVERALL (whole trajectory)")
    log(f"  vs DP: myopic={dm}/{dt}={dr:.3f}  gnn={gm}/{gt}={gr:.3f}  mlp={mm}/{mt}={mr:.3f}  all4={am}/{at}={ar:.3f}")
    log(f"OVERALL (root / first action only)")
    log(f"  vs DP: myopic={root_dp_myo}/{root_total}  gnn={root_dp_gnn}/{root_total}  mlp={root_dp_mlp}/{root_total}")

    payload = {
        "genes": args.genes, "seed": args.seed, "variant": "E6_sumpool",
        "overall_whole_traj": {
            "myopic": {"n_match": dm, "n_total": dt, "rate": dr},
            "gnn":    {"n_match": gm, "n_total": gt, "rate": gr},
            "mlp":    {"n_match": mm, "n_total": mt, "rate": mr},
            "all4":   {"n_match": am, "n_total": at, "rate": ar},
        },
        "overall_root_only": {
            "myopic": {"n_match": root_dp_myo, "n_total": root_total},
            "gnn":    {"n_match": root_dp_gnn, "n_total": root_total},
            "mlp":    {"n_match": root_dp_mlp, "n_total": root_total},
        },
        "per_config": {k: {kk: vv for kk, vv in v.items() if kk != "steps"} for k, v in summary.items()},
        "detail": summary,
    }
    (out_dir / "e6_fourway.json").write_text(json.dumps(payload, indent=2))
    log(f"saved -> {out_dir/'e6_fourway.json'}")
    log(f"saved -> {log_path}")
    log_f.close()


if __name__ == "__main__":
    main()
