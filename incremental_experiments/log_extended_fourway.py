"""Extended-family OOD four-way action log: DP (ground truth) vs MYOPIC
(Kanix's canonical myopic_greedy) vs GNN-Q vs MLP-Q, on the Extended family
(6 people: Grandfather, Grandmother, Father, Uncle, Mother, Child -- adds a
branching Uncle sibling off Father, unlike ThreeGeneration's linear chain).

Extended is defined in shared/data_gen.py::FAMILY_CASES but was never used
anywhere in training or testing -- this is a genuinely out-of-distribution
topology for GNN-Q/MLP-Q, which only ever saw Trio/Nuclear (train) and
ThreeGeneration (test). No retraining: reuses the existing E4 GNN-Q
checkpoints (bidirectional message passing) and E2 MLP-Q checkpoints
(cross-entropy + batching) as-is. Both models pool over people with
mean(dim=1) and have no fixed-N_people dependency, so they can run on a
6-person family unchanged.

2-gene: build_two_gene_dataset builds the belief map + exact DP on the fly
(fast, same machinery already used for every 2-gene test config).
3-gene: build_multigene_dataset does the same for 3 genes -- there is no
cached pickle for Extended at 3 genes (only Trio/Nuclear/ThreeGeneration were
ever precomputed), so this solves exact DP from scratch for a 6-person,
3-gene family. Run as a SLURM job, not interactively.

Outputs:
    results/extended/{2gene|3gene}/seed{S}/extended_fourway.log
    results/extended/{2gene|3gene}/seed{S}/extended_fourway.json

Usage:
    python log_extended_fourway.py --genes 3 --seed 0
    python log_extended_fourway.py --genes 2 --seed 0
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
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


from exputils.eval import _get_entry, _stop_val, q_rollout  # noqa: E402
from genetic_dp.utils.pedigree_generator import generate_deterministic_pedigree  # noqa: E402
from genetic_dp.policy.baselines import myopic_greedy  # noqa: E402
from genetic_dp.policy.myopic import evaluate_myopic_policy  # noqa: E402
from genetic_dp.exact_dp.utils import GENOTYPE_STATES  # noqa: E402
from shared.data_gen import (  # noqa: E402
    FAMILY_CASES, build_two_gene_dataset, build_multigene_dataset,
)

RESULTS_BASE = HERE / "results" / "extended"

REGIMES_2GENE = {
    "LowHigh":    {"GeneA": 0.02, "GeneB": 0.15},
    "MediumEven": {"GeneA": 0.08, "GeneB": 0.08},
    "LowLow":     {"GeneA": 0.02, "GeneB": 0.02},
    "HighHigh":   {"GeneA": 0.15, "GeneB": 0.15},
    "MixedA":     {"GeneA": 0.02, "GeneB": 0.10},
    "MixedB":     {"GeneA": 0.05, "GeneB": 0.12},
}
sys.path.insert(0, str(EXPERIMENTS / "step9_gnn_3gene"))
from build_datasets import ALLELE_FREQS as REGIMES_3GENE  # noqa: E402 -- single source of truth, not a hand copy
PRESETS = ["Base", "Aggressive"]


def ds_to_tensors_generic(ds, struct_feats, genes, device):
    individuals = ds["individuals"]
    n_people    = len(individuals)
    person_idx  = {p: i for i, p in enumerate(individuals)}
    states      = ds["states"]
    N           = len(states)

    X = ds["X"].reshape(N, n_people, 3 * len(genes)).astype(np.float32)
    tested_arr = np.zeros((N, n_people, 1), dtype=np.float32)
    for i, state in enumerate(states):
        for person, _ in state:
            tested_arr[i, person_idx[person], 0] = 1.0
    struct = np.tile(struct_feats[np.newaxis, :, :], (N, 1, 1))
    nf = np.concatenate([X, tested_arr, struct], axis=-1).astype(np.float32)

    config = ds["config"]
    vec = []
    for g in genes:
        vec.append(min(config.a_gene[g].values(), key=abs))
        vec.append(min(config.b_gene[g].values(), key=abs))
        vec.append(list(config.delta_gene[g].values())[0])
    vec.append(config.fixed_cost)
    vec.append(config.variable_cost)
    cv = np.array(vec, dtype=np.float32)
    gf = np.tile(cv[np.newaxis, :], (N, 1)).astype(np.float32)

    return {
        "nf": torch.tensor(nf, device=device),
        "gf": torch.tensor(gf, device=device),
    }


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


def myopic_pick(state, individuals, belief, genes, config):
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


def precompute_qhat(model, ds, edge_index_t, device, is_gnn, batch_size=4096, nmask=None):
    from qsa_data import build_qsa_index
    state_idx, action_idx, _ = build_qsa_index(ds, device=device, cache_key=None)
    individuals = ds["individuals"]
    states = ds["states"]
    nf, gf = ds["_nf"], ds["_gf"]

    model.eval()
    q_hat: dict = {}
    with torch.no_grad():
        for start in range(0, len(state_idx), batch_size):
            end  = min(start + batch_size, len(state_idx))
            s_b  = state_idx[start:end]
            a_b  = action_idx[start:end]
            if is_gnn:
                if nmask is not None:
                    pred = model(nf[s_b], edge_index_t, gf[s_b], a_b, nmask)
                else:
                    pred = model(nf[s_b], edge_index_t, gf[s_b], a_b)
            else:
                pred = model(nf[s_b], gf[s_b], a_b)
            s_b_cpu, a_b_cpu, pred_cpu = s_b.cpu(), a_b.cpu(), pred.cpu()
            for i in range(end - start):
                s = states[s_b_cpu[i].item()]
                p = individuals[a_b_cpu[i].item()]
                q_hat.setdefault(s, {})[p] = pred_cpu[i].item()
    return q_hat


def walk_and_log(state0, q_hat_gnn, q_hat_mlp, policy_dp, individuals, belief,
                  genes, config, log, max_steps=20):
    state = state0
    n_total = 0
    n_dp_myo = n_dp_gnn = n_dp_mlp = n_all4 = 0
    rows = []
    log(f"  {'step':<5}{'state':<60}{'DP (truth)':<18}{'MYOPIC':<16}{'GNN':<16}{'MLP':<16}{'agree'}")
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
        log(f"  {step:<5}{fmt_state(state):<60}{d_str:<18}{y_str:<16}{g_str:<16}{m_str:<16}{agree}")
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


def run(genes_n, seed, device, log, only_regime=None, only_preset=None, variant="e4"):
    tg_or_gnn = _load_module("gnn_run_ext", EXP_ROOT / "gnn" / "run.py")
    compute_structural_features = tg_or_gnn.compute_structural_features
    build_edge_index = tg_or_gnn.build_edge_index

    if genes_n == 2:
        genes = ("GeneA", "GeneB")
        regimes = REGIMES_2GENE
        mlp_q_plain = _load_module("mlp_q_ext2", Q_DIR / "two_gene_mlp_q.py")
        if variant == "e4":
            gnn_wt = HERE / "results" / "e4_gnn_bidir_2gene" / "seed_runs" / f"seed{seed}" / "gnn_q_e4_2gene.pt"
            mlp_wt = ROOT / "fixing_gnn_q" / "results" / "mlp_q_ce_2gene" / "seed_runs" / f"seed{seed}" / "mlp_q_ce_2gene.pt"
            gnn_q_class_src = _load_module("gnn_q_e4_bidir_ext2", HERE / "e4_train_two_gene_gnn_q.py")
            GNNClass, MLPClass = gnn_q_class_src.GNNQBidir, mlp_q_plain.MLPQ
        elif variant == "e6":
            gnn_wt = HERE / "results" / "e6_gnn_sumpool_2gene" / "seed_runs" / f"seed{seed}" / "gnn_q_e6_2gene.pt"
            mlp_wt = HERE / "results" / "e6_mlp_sumpool_2gene" / "seed_runs" / f"seed{seed}" / "mlp_q_e6_2gene.pt"
            gnn_q_class_src = _load_module("gnn_q_e6_ext2", HERE / "e6_train_two_gene_gnn_q.py")
            mlp_q_class_src = _load_module("mlp_q_e6_ext2", HERE / "e6_train_two_gene_mlp_q.py")
            GNNClass, MLPClass = gnn_q_class_src.GNNQBidirSumPool, mlp_q_class_src.MLPQSumPool
        else:  # e7 -- MLP unchanged from E2, same as e4
            gnn_wt = HERE / "results" / "e7_gnn_neighborpool_2gene" / "seed_runs" / f"seed{seed}" / "gnn_q_e7_2gene.pt"
            mlp_wt = ROOT / "fixing_gnn_q" / "results" / "mlp_q_ce_2gene" / "seed_runs" / f"seed{seed}" / "mlp_q_ce_2gene.pt"
            gnn_q_class_src = _load_module("gnn_q_e7_ext2", HERE / "e7_train_two_gene_gnn_q.py")
            GNNClass, MLPClass = gnn_q_class_src.GNNQBidirNeighborPool, mlp_q_plain.MLPQ
    else:
        genes = ("GeneA", "GeneB", "GeneC")
        regimes = REGIMES_3GENE
        mlp_q_plain = _load_module("mlp_q_ext3", Q_DIR / "mlp_q.py")
        if variant == "e4":
            gnn_wt = HERE / "results" / "e4_gnn_bidir" / "seed_runs" / f"seed{seed}" / "gnn_q_e4.pt"
            mlp_wt = ROOT / "fixing_gnn_q" / "results" / "mlp_q_ce" / "seed_runs" / f"seed{seed}" / "mlp_q_ce.pt"
            gnn_q_class_src = _load_module("gnn_q_e4_bidir_ext3", HERE / "e4_train_gnn_q.py")
            GNNClass, MLPClass = gnn_q_class_src.GNNQBidir, mlp_q_plain.MLPQ
        elif variant == "e6":
            gnn_wt = HERE / "results" / "e6_gnn_sumpool" / "seed_runs" / f"seed{seed}" / "gnn_q_e6.pt"
            mlp_wt = HERE / "results" / "e6_mlp_sumpool" / "seed_runs" / f"seed{seed}" / "mlp_q_e6.pt"
            gnn_q_class_src = _load_module("gnn_q_e6_ext3", HERE / "e6_train_gnn_q.py")
            mlp_q_class_src = _load_module("mlp_q_e6_ext3", HERE / "e6_train_mlp_q.py")
            GNNClass, MLPClass = gnn_q_class_src.GNNQBidirSumPool, mlp_q_class_src.MLPQSumPool
        else:  # e7 -- MLP unchanged from E2, same as e4
            gnn_wt = HERE / "results" / "e7_gnn_neighborpool" / "seed_runs" / f"seed{seed}" / "gnn_q_e7.pt"
            mlp_wt = ROOT / "fixing_gnn_q" / "results" / "mlp_q_ce" / "seed_runs" / f"seed{seed}" / "mlp_q_ce.pt"
            gnn_q_class_src = _load_module("gnn_q_e7_ext3", HERE / "e7_train_gnn_q.py")
            GNNClass, MLPClass = gnn_q_class_src.GNNQBidirNeighborPool, mlp_q_plain.MLPQ

    if not gnn_wt.exists():
        raise FileNotFoundError(f"missing {variant} GNN weights: {gnn_wt}")
    if not mlp_wt.exists():
        raise FileNotFoundError(f"missing {variant} MLP weights: {mlp_wt}")

    gmodel = GNNClass().to(device)
    gmodel.load_state_dict(torch.load(gnn_wt, map_location=device))
    gmodel.eval()
    log(f"loaded GNN ({variant}) {gnn_wt}")

    mmodel = MLPClass().to(device)
    mmodel.load_state_dict(torch.load(mlp_wt, map_location=device))
    mmodel.eval()
    log(f"loaded MLP ({variant}) {mlp_wt}")

    pedigree = generate_deterministic_pedigree(FAMILY_CASES["Extended"])
    sample_individuals = pedigree.to_list()
    struct_feats = compute_structural_features(pedigree, sample_individuals)
    edge_index = build_edge_index(pedigree, sample_individuals)
    edge_index_t = torch.tensor(edge_index, device=device)

    nmask = None
    if variant == "e7":
        nmask_np = gnn_q_class_src.build_neighbor_mask(edge_index, len(sample_individuals))
        nmask = torch.tensor(nmask_np, device=device)

    summary = {}
    for reg in regimes:
        if only_regime is not None and reg != only_regime:
            continue
        for pre in PRESETS:
            if only_preset is not None and pre != only_preset:
                continue
            key = f"Extended_{reg}_{pre}_{genes_n}gene"
            log(f"\n{'='*90}\n{key}\n{'='*90}")

            if genes_n == 2:
                ds = build_two_gene_dataset(
                    family_label="Extended", allele_freqs=regimes[reg], preset_label=pre, genes=genes,
                )
            else:
                ds = build_multigene_dataset(
                    family_label="Extended", allele_freqs=regimes[reg], preset_label=pre, genes=genes,
                )
            log(f"  states={len(ds['states']):,}  V*(root)={ds['V_root']:.4f}  V_stop(root)={ds['V_stop_root']:.4f}")

            base = ds_to_tensors_generic(ds, struct_feats, genes, device)
            ds["_nf"], ds["_gf"] = base["nf"], base["gf"]

            q_hat_gnn = precompute_qhat(gmodel, ds, edge_index_t, device, is_gnn=True, nmask=nmask)
            q_hat_mlp = precompute_qhat(mmodel, ds, edge_index_t, device, is_gnn=False)

            individuals = ds["individuals"]
            belief = ds["belief"]
            config = ds["config"]
            policy_dp = ds["policy_dp"]

            ratio2_gnn, L_gnn = q_rollout(q_hat_gnn, ds, log=log, trace=False)
            ratio2_mlp, L_mlp = q_rollout(q_hat_mlp, ds, log=log, trace=False)

            belief_wrapped = {s: e if isinstance(e, tuple) else (e, None) for s, e in belief.items()}
            myopic_result = evaluate_myopic_policy(
                belief=belief_wrapped, individuals=individuals, gen_states=GENOTYPE_STATES,
                infer=None, a=config.a, b=config.b, c=config.c, delta=config.delta,
                fixed_cost=config.fixed_cost, variable_cost=config.variable_cost, genes=genes,
                a_gene=config.a_gene, b_gene=config.b_gene, c_gene=config.c_gene, delta_gene=config.delta_gene,
                state_pool=belief.keys(),
            )
            L_myopic = myopic_result.root_value
            denom = ds["V_root"] - ds["V_stop_root"]
            ratio2_myopic = (ds["V_root"] - L_myopic) / denom if abs(denom) > 1e-12 else 0.0

            log(f"  ratio2: myopic={ratio2_myopic:.4f}  gnn={ratio2_gnn:.4f}  mlp={ratio2_mlp:.4f}")

            summary[key] = walk_and_log(frozenset(), q_hat_gnn, q_hat_mlp, policy_dp,
                                         individuals, belief, genes, config, log)
            summary[key]["ratio2_myopic"] = ratio2_myopic
            summary[key]["ratio2_gnn"] = ratio2_gnn
            summary[key]["ratio2_mlp"] = ratio2_mlp
            summary[key]["V_root"] = ds["V_root"]

    return summary


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--genes", type=int, required=True, choices=[2, 3])
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="cpu")
    p.add_argument("--regime", default=None,
                    help="If set (with --preset), run only this one config and write a per-config "
                         "checkpoint file instead of the full aggregate -- for 3-gene, where a single "
                         "seed's all-12-configs run can exceed the SLURM time limit before finishing.")
    p.add_argument("--preset", default=None, choices=[None, "Base", "Aggressive"])
    p.add_argument("--variant", default="e4", choices=["e4", "e6", "e7"],
                    help="e4 = mean pooling (bidirectional MP + CE), e6 = sum pooling. "
                         "e4 keeps the original filenames; e6 writes alongside with an _e6 suffix "
                         "so neither overwrites the other.")
    args = p.parse_args()

    tag = f"{args.genes}gene"
    out_dir = RESULTS_BASE / tag / f"seed{args.seed}"
    out_dir.mkdir(parents=True, exist_ok=True)
    suffix = "" if args.variant == "e4" else f"_{args.variant}"

    if args.regime is not None:
        assert args.preset is not None, "--regime requires --preset"
        configs_dir = out_dir / f"configs{suffix}"
        configs_dir.mkdir(parents=True, exist_ok=True)
        log_path = configs_dir / f"{args.regime}_{args.preset}.log"
    else:
        log_path = out_dir / f"extended_fourway{suffix}.log"
    log_f = open(log_path, "w")

    def log(msg=""):
        print(msg, flush=True)
        log_f.write(msg + "\n")
        log_f.flush()

    only_suffix = f"  only={args.regime}/{args.preset}" if args.regime else ""
    model_desc = {
        "e4": "GNN=E4 bidir  MLP=E2 ce",
        "e6": "GNN=E6 sumpool  MLP=E6 sumpool",
        "e7": "GNN=E7 neighborpool  MLP=E2 ce",
    }[args.variant]
    log(f"[extended_fourway] {datetime.now().isoformat()}  genes={args.genes}  seed={args.seed}"
        f"  family=Extended (OOD -- never trained on)  {model_desc}{only_suffix}")

    device = torch.device(args.device)
    summary = run(args.genes, args.seed, device, log, only_regime=args.regime, only_preset=args.preset,
                  variant=args.variant)

    if args.regime is not None:
        key = f"Extended_{args.regime}_{args.preset}_{args.genes}gene"
        (configs_dir / f"{args.regime}_{args.preset}.json").write_text(json.dumps(summary[key], indent=2))
        log(f"saved -> {configs_dir / f'{args.regime}_{args.preset}.json'}")
        log_f.close()
        return

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

    avg_ratio2_myopic = float(np.mean([v["ratio2_myopic"] for v in summary.values()]))
    avg_ratio2_gnn = float(np.mean([v["ratio2_gnn"] for v in summary.values()]))
    avg_ratio2_mlp = float(np.mean([v["ratio2_mlp"] for v in summary.values()]))

    log(f"\n{'='*90}\nOVERALL (whole trajectory)")
    log(f"  vs DP: myopic={dm}/{dt}={dr:.3f}  gnn={gm}/{gt}={gr:.3f}  mlp={mm}/{mt}={mr:.3f}  all4={am}/{at}={ar:.3f}")
    log(f"OVERALL (root / first action only)")
    log(f"  vs DP: myopic={root_dp_myo}/{root_total}  gnn={root_dp_gnn}/{root_total}  mlp={root_dp_mlp}/{root_total}")
    log(f"OVERALL avg ratio2: myopic={avg_ratio2_myopic:.4f}  gnn={avg_ratio2_gnn:.4f}  mlp={avg_ratio2_mlp:.4f}")

    payload = {
        "genes": args.genes, "seed": args.seed, "family": "Extended", "variant": args.variant,
        "overall_whole_traj": {
            "myopic": {"n_match": dm, "n_total": dt, "rate": dr},
            "gnn":    {"n_match": gm, "n_total": gt, "rate": gr},
            "mlp":    {"n_match": mm, "n_total": mt, "rate": mr},
            "all4":   {"n_match": am, "n_total": at, "rate": ar},
        },
        "overall_ratio2": {
            "myopic": avg_ratio2_myopic, "gnn": avg_ratio2_gnn, "mlp": avg_ratio2_mlp,
        },
        "overall_root_only": {
            "myopic": {"n_match": root_dp_myo, "n_total": root_total},
            "gnn":    {"n_match": root_dp_gnn, "n_total": root_total},
            "mlp":    {"n_match": root_dp_mlp, "n_total": root_total},
        },
        "per_config": {k: {kk: vv for kk, vv in v.items() if kk != "steps"} for k, v in summary.items()},
        "detail": summary,
    }
    (out_dir / f"extended_fourway{suffix}.json").write_text(json.dumps(payload, indent=2))
    log(f"saved -> {out_dir/f'extended_fourway{suffix}.json'}")
    log(f"saved -> {log_path}")
    log_f.close()


if __name__ == "__main__":
    main()
