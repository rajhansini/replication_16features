"""Three overfitting probes for the 2-gene GNN-Q / MLP-Q models.

Motivation
----------
The original 2-gene "overfitting" signal came from the standard split
(TRAIN=Trio+Nuclear, TEST=ThreeGeneration), where the GNN showed a 3-16x
TRAIN->TEST degradation. The e6/e0 split experiments then showed that gap is
mostly an artifact of WHICH families are in TRAIN: Trio (432 states) and
Nuclear (2,264 states) are the two smallest families, so the training problem
is trivially small, while ThreeGeneration (20,816 states) is ~8x larger than
both combined. Swapping Nuclear -> ThreeGeneration in TRAIN (the "rotate"
split), holding family count and config count fixed, raises TRAIN error ~6x
and collapses the gap to ~1.1x on both E0 and E6.

So these probes all run on the FAIR split (rotate) by default, and ask the
question that actually remains: is there ANY genuine overfitting once the
split is not stacked?

    curve         -- train loss vs held-out validation loss every epoch, plus
                     rollout ratio2 on both sides at checkpoints. Overfitting
                     has a signature: val loss bottoms out and then climbs
                     while train loss keeps falling. If val is monotone to the
                     last epoch, there is no overfitting dynamic to fix and
                     the models are capacity-limited instead.

    randlabel     -- train on SHUFFLED Q targets (targets permuted within each
                     config, so the target distribution is untouched and only
                     the (s,a)->y mapping is destroyed). Establishes the
                     ceiling: if the model cannot drive train MSE toward 0 on
                     random targets, it does not have the capacity to memorize
                     and memorization was never a viable explanation.
                     Compare its train MSE against the `curve` probe's, which
                     is the same setup with real targets.

    configholdout -- hold out whole REGIMES inside the SEEN families, so the
                     model sees the pedigree structure but not those configs.
                     Evaluates three ways:
                       (a) train configs                  (seen)
                       (b) held-out regimes, seen families (unseen config)
                       (c) a fully unseen family           (unseen structure)
                     (a)<<(b) means it memorizes configs. (a)~(b)<<(c) means
                     it generalizes across configs fine and the real failure
                     is transfer across pedigree topology -- a different claim
                     from overfitting, and the one the split results point at.

Reuses E6's model classes and training/eval code UNCHANGED, imported from
e6_train_two_gene_gnn_q.py / e6_train_two_gene_mlp_q.py -- same pattern as
e6_split_experiment.py. Nothing here reimplements the models, the loss, or
the rollout.

Resume: checkpoint.pt every epoch, per-config partial eval files, and a
results.json skip guard, so the whole array is safe to resubmit (every
partition on this cluster has a hard 4h MaxTime).

Usage:
    python overfit_probes_2gene.py --probe curve         --kind gnn --seed 0
    python overfit_probes_2gene.py --probe randlabel     --kind gnn --seed 0
    python overfit_probes_2gene.py --probe configholdout --kind mlp --seed 0
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import random
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import torch

HERE   = Path(__file__).resolve().parent
ROOT   = HERE.parent
Q_DIR  = ROOT / "experiments_after_understanding" / "q_learning"
FIXING = ROOT / "fixing_gnn_q"
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "ground-up-experiments"))
sys.path.insert(0, str(Q_DIR))
sys.path.insert(0, str(FIXING))


def _load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


gnn_mod = _load_module("probe_e6_gnn_mod", HERE / "e6_train_two_gene_gnn_q.py")
mlp_mod = _load_module("probe_e6_mlp_mod", HERE / "e6_train_two_gene_mlp_q.py")
tg = gnn_mod.tg

RESULTS_ROOT = HERE / "results"

# The fair split ("rotate"): TRAIN families include one large family, so TRAIN
# error is not artificially near-zero. See module docstring.
FAIR_TRAIN = ["Trio", "ThreeGeneration"]
FAIR_TEST  = ["Nuclear"]

# For configholdout: whole regimes withheld from the TRAIN families. Held out
# as complete regimes (both presets) so no Base/Aggressive pair leaks a config
# whose partner was trained on.
HOLDOUT_REGIMES = ["MixedA", "MixedB"]


def configs_for(families, regimes=None, presets=None):
    regimes = list(tg.ALLELE_FREQ_REGIMES) if regimes is None else regimes
    presets = list(tg.PRESETS_LIST) if presets is None else presets
    return [(f, r, p) for f in families for r in regimes for p in presets]


def build_struct(families, need_edges):
    struct, edges = {}, {}
    for fam in families:
        ped = tg.generate_deterministic_pedigree(tg.FAMILY_CASES[fam])
        ind = ped.to_list()
        struct[fam] = tg.compute_structural_features(ped, ind)
        if need_edges:
            edges[fam] = tg.build_edge_index(ped, ind)
    return struct, edges


def prepare(mod, configs, struct, dev, ds_cache, log, shuffle_targets=False, seed=0):
    """Build the per-config training tensors, exactly as the E6 trainers do.

    shuffle_targets permutes y within each config (randlabel probe): the target
    distribution is preserved, only the (s,a)->y correspondence is destroyed.
    The permutation is applied AFTER build_qsa_index so its on-disk cache is
    never poisoned with shuffled targets.
    """
    per_config = []
    for i, (fam, reg, pre) in enumerate(configs):
        key = f"{fam}_{reg}_{pre}_2gene"
        ds = mod.build_config(fam, reg, pre)
        base = tg.ds_to_tensors(ds, struct[fam], dev)
        ds["_nf"], ds["_gf"] = base["nf"], base["gf"]
        ds_cache[key] = ds
        s_idx, a_idx, y = mod.build_qsa_index(ds, device=dev, cache_key=key)
        if shuffle_targets:
            g = torch.Generator()
            g.manual_seed(seed * 100003 + i)
            y = y[torch.randperm(y.shape[0], generator=g).to(y.device)]
        k_max = len(ds["individuals"])
        gr, gm, gt = mod.build_state_groups(s_idx, y, k_max)
        per_config.append((fam, {
            "nf": base["nf"], "gf": base["gf"],
            "state_idx": s_idx, "action_idx": a_idx, "y": y,
            "group_rows": gr, "group_mask": gm, "group_target": gt,
        }))
        log(f"    [{key}] {len(ds['states']):,} states -> {len(y):,} (s,a) rows -> {gr.shape[0]:,} groups")
    return per_config


def eval_loss(mod, model, per_config, edge_t, groups_per_batch, lambda_ce, dev, is_gnn):
    """Same combined_loss the trainer optimizes, under no_grad and no shuffling.

    This is the held-out validation loss -- the classic overfitting readout,
    and cheap enough (one forward pass) to run every epoch.
    """
    model.eval()
    tot = totm = totc = 0.0
    n = 0
    with torch.no_grad():
        for fam, d in per_config:
            nf, gf = d["nf"], d["gf"]
            s_i, a_i, y = d["state_idx"], d["action_idx"], d["y"]
            gr, gm, gt = d["group_rows"], d["group_mask"], d["group_target"]
            for start in range(0, gr.shape[0], groups_per_batch):
                rows = gr[start:start + groups_per_batch]
                mask = gm[start:start + groups_per_batch]
                target = gt[start:start + groups_per_batch]
                flat = rows.reshape(-1)
                fm = mask.reshape(-1)
                vr = flat[fm]
                s_b, a_b = s_i[vr], a_i[vr]
                pred_valid = model(nf[s_b], edge_t[fam], gf[s_b], a_b) if is_gnn else model(nf[s_b], gf[s_b], a_b)
                pred = torch.zeros(flat.shape[0], device=dev)
                pred[fm] = pred_valid
                pred = pred.reshape(rows.shape)
                yg = torch.zeros(flat.shape[0], device=dev)
                yg[fm] = y[vr]
                yg = yg.reshape(rows.shape)
                loss, mse_v, ce_v = mod.combined_loss(pred, mask, target, yg, lambda_ce)
                tot += loss.item()
                totm += mse_v
                totc += ce_v
                n += 1
    model.train()
    return tot / n, totm / n, totc / n


def eval_ratio2(mod, model, configs, ds_cache, struct, edges, dev, log, is_gnn, partial_path=None):
    """Rollout ratio2 over `configs`, reusing already-solved datasets from
    ds_cache. Identical to mod.evaluate_rollout on a cache miss. Flushes to
    partial_path per config so a wall-clock kill resumes mid-eval.
    """
    results = {}
    if partial_path is not None and partial_path.exists():
        try:
            results = json.loads(partial_path.read_text())
        except json.JSONDecodeError:
            results = {}
        if results:
            log(f"  [RESUME] {len(results)}/{len(configs)} configs already evaluated")
    for fam, reg, pre in configs:
        key = f"{fam}_{reg}_{pre}_2gene"
        if key in results:
            continue
        if key in ds_cache:
            ds = ds_cache[key]
        else:
            ds = mod.build_config(fam, reg, pre)
            base = tg.ds_to_tensors(ds, struct[fam], dev)
            ds["_nf"], ds["_gf"] = base["nf"], base["gf"]
        if is_gnn:
            q_hat = mod.precompute_qhat(model, ds, key, torch.tensor(edges[fam], device=dev), dev)
        else:
            q_hat = mod.precompute_qhat(model, ds, key, dev)
        ratio2, L = mod.q_rollout(q_hat, ds, log=log, trace=False)
        log(f"  [{key}]  ratio2={ratio2:.4f}  L={L:.4f}  V*={ds['V_root']:.4f}")
        results[key] = {"ratio2": ratio2, "L": L, "V_root": ds["V_root"]}
        if partial_path is not None:
            partial_path.write_text(json.dumps(results, indent=2))
    avg = float(np.mean([r["ratio2"] for r in results.values()]))
    log(f"  avg ratio2 = {avg:.4f}")
    return results, avg


def run(probe, kind, results_dir, epochs, groups_per_batch, lambda_ce, seed, device, ratio_every, log):
    dev = torch.device(device)
    is_gnn = kind == "gnn"
    mod = gnn_mod if is_gnn else mlp_mod
    model_cls = gnn_mod.GNNQBidirSumPool if is_gnn else mlp_mod.MLPQSumPool

    # ---- what each probe trains on and evaluates ----
    if probe == "configholdout":
        train_fams, unseen_fams = FAIR_TRAIN, FAIR_TEST
        seen_regimes = [r for r in tg.ALLELE_FREQ_REGIMES if r not in HOLDOUT_REGIMES]
        train_configs   = configs_for(train_fams, regimes=seen_regimes)
        heldout_configs = configs_for(train_fams, regimes=HOLDOUT_REGIMES)
        unseen_configs  = configs_for(unseen_fams)
        log(f"TRAIN families={train_fams} regimes={seen_regimes} -> {len(train_configs)} configs")
        log(f"HELD-OUT regimes={HOLDOUT_REGIMES} (same families) -> {len(heldout_configs)} configs")
        log(f"UNSEEN family={unseen_fams} -> {len(unseen_configs)} configs")
    else:
        train_fams, unseen_fams = FAIR_TRAIN, FAIR_TEST
        train_configs   = configs_for(train_fams)
        heldout_configs = []
        unseen_configs  = configs_for(unseen_fams)
        log(f"TRAIN families={train_fams} -> {len(train_configs)} configs")
        log(f"VAL/TEST family={unseen_fams} -> {len(unseen_configs)} configs")

    all_fams = sorted(set(train_fams) | set(unseen_fams))
    struct, edges = build_struct(all_fams, need_edges=is_gnn)
    edge_t = {f: torch.tensor(e, device=dev) for f, e in edges.items()} if is_gnn else {}

    model = model_cls().to(dev)
    log(f"Parameters: {sum(p.numel() for p in model.parameters())}")

    ckpt_path = results_dir / "checkpoint.pt"
    ds_cache = {}

    resume_epoch = 0
    if ckpt_path.exists():
        resume_epoch = int(torch.load(ckpt_path, map_location=dev)["epoch"])

    log(f"\n[1] Building TRAIN tensors ({len(train_configs)} configs)"
        + ("  [TARGETS SHUFFLED -- randlabel control]" if probe == "randlabel" else "") + "...")
    per_train = prepare(mod, train_configs, struct, dev, ds_cache, log,
                        shuffle_targets=(probe == "randlabel"), seed=seed)

    per_val = []
    if probe == "curve":
        log(f"\n[1b] Building held-out VALIDATION tensors ({len(unseen_configs)} configs, family {unseen_fams})...")
        per_val = prepare(mod, unseen_configs, struct, dev, ds_cache, log)

    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    start_epoch = 1
    curve = []
    curve_path = results_dir / "curve.json"
    if ckpt_path.exists():
        ck = torch.load(ckpt_path, map_location=dev)
        model.load_state_dict(ck["model_state"])
        opt.load_state_dict(ck["optimizer_state"])
        start_epoch = ck["epoch"] + 1
        if curve_path.exists():
            try:
                curve = json.loads(curve_path.read_text())
            except json.JSONDecodeError:
                curve = []
        log(f"\n[RESUME] checkpoint at epoch {ck['epoch']}, resuming from {start_epoch}")

    log(f"\n[2] Training (epochs {start_epoch}-{epochs})...")
    t0 = time.time()
    for ep in range(start_epoch, epochs + 1):
        tl = tm = tc = 0.0
        for fam, data in per_train:
            if is_gnn:
                l, m, c = mod.train_one_epoch(model, edge_t[fam], data, opt, groups_per_batch, lambda_ce, dev)
            else:
                l, m, c = mod.train_one_epoch(model, data, opt, groups_per_batch, lambda_ce, dev)
            tl += l; tm += m; tc += c
        n = len(per_train)
        point = {"epoch": ep, "train_loss": tl / n, "train_mse": tm / n, "train_ce": tc / n}
        if per_val:
            vl, vm, vc = eval_loss(mod, model, per_val, edge_t, groups_per_batch, lambda_ce, dev, is_gnn)
            point.update({"val_loss": vl, "val_mse": vm, "val_ce": vc})
        curve.append(point)
        if ep % 20 == 0 or ep == 1:
            msg = f"    epoch {ep:4d}  train={point['train_loss']:.5f} (mse={point['train_mse']:.5f})"
            if per_val:
                msg += f"   val={point['val_loss']:.5f} (mse={point['val_mse']:.5f})"
            log(msg)
        torch.save({"epoch": ep, "model_state": model.state_dict(),
                    "optimizer_state": opt.state_dict()}, ckpt_path)
        curve_path.write_text(json.dumps(curve, indent=2))
    log(f"    done in {time.time() - t0:.1f}s")

    # ---- final evaluation, per probe ----
    out = {"probe": probe, "kind": kind, "seed": seed, "epochs": epochs,
           "train_families": train_fams,
           "final_train_loss": curve[-1]["train_loss"] if curve else None,
           "final_train_mse": curve[-1]["train_mse"] if curve else None}

    if probe == "randlabel":
        # No rollout: with shuffled targets ratio2 is meaningless by construction.
        # The whole readout is whether train MSE can be driven toward zero.
        log("\n[3] randlabel: no rollout eval (shuffled targets make ratio2 meaningless). "
            "Readout is final train MSE vs the real-target run.")
    else:
        log(f"\n[3] ratio2 on TRAIN configs ({len(train_configs)})...")
        _, seen_avg = eval_ratio2(mod, model, train_configs, ds_cache, struct, edges, dev, log, is_gnn,
                                  partial_path=results_dir / "eval_seen_partial.json")
        out["ratio2_seen_configs"] = seen_avg
        if heldout_configs:
            log(f"\n[4] ratio2 on HELD-OUT regimes, SEEN families ({len(heldout_configs)})...")
            _, held_avg = eval_ratio2(mod, model, heldout_configs, ds_cache, struct, edges, dev, log, is_gnn,
                                      partial_path=results_dir / "eval_heldout_partial.json")
            out["ratio2_heldout_configs_seen_families"] = held_avg
        log(f"\n[5] ratio2 on UNSEEN family {unseen_fams} ({len(unseen_configs)})...")
        _, unseen_avg = eval_ratio2(mod, model, unseen_configs, ds_cache, struct, edges, dev, log, is_gnn,
                                    partial_path=results_dir / "eval_unseen_partial.json")
        out["ratio2_unseen_family"] = unseen_avg

    (results_dir / "results.json").write_text(json.dumps(out, indent=2))
    log("\n" + json.dumps(out, indent=2))
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--probe", required=True, choices=["curve", "randlabel", "configholdout"])
    p.add_argument("--kind", required=True, choices=["gnn", "mlp"])
    p.add_argument("--device", default="cpu")
    p.add_argument("--epochs", type=int, default=500)
    p.add_argument("--groups_per_batch", type=int, default=512)
    p.add_argument("--lambda_ce", type=float, default=1.0)
    p.add_argument("--ratio_every", type=int, default=0, help="reserved; final-epoch ratio2 only")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--force", action="store_true")
    args = p.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    results_dir = RESULTS_ROOT / f"overfit_probe_{args.probe}" / args.kind / f"seed{args.seed}"
    results_dir.mkdir(parents=True, exist_ok=True)

    done = results_dir / "results.json"
    if done.exists() and not args.force:
        print(f"[SKIP] probe={args.probe} kind={args.kind} seed={args.seed} already complete ({done}).", flush=True)
        return
    if args.force:
        for f in results_dir.glob("*partial*.json"):
            f.unlink()
        for f in (results_dir / "checkpoint.pt", results_dir / "curve.json"):
            f.unlink(missing_ok=True)

    log_f = open(results_dir / "run.log", "a")

    def log(msg=""):
        print(msg, flush=True)
        log_f.write(msg + "\n")
        log_f.flush()

    log(f"\n{'='*70}\n[OVERFIT-PROBE {args.probe} kind={args.kind} seed={args.seed}] "
        f"{datetime.now().isoformat()}  epochs={args.epochs}")
    run(args.probe, args.kind, results_dir, args.epochs, args.groups_per_batch,
        args.lambda_ce, args.seed, args.device, args.ratio_every, log)
    log_f.close()


if __name__ == "__main__":
    main()
