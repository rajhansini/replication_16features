"""Does the 2-gene Q model actually fit Q-values, or only rank actions?

Motivation: the overfit probes found no meaningful overfitting, and the
train/test gaps were small. A natural explanation is that the model never fits
its training data tightly in the first place. This checks that directly on the
TRAIN configs of the fair split, using the trained checkpoints.

Two readouts per config:
  * VALUE fit   -- MSE vs Var(y), i.e. R^2, plus corr(pred, y) and the
                   predicted-vs-true standard deviations. R^2 <= 0 means the
                   model is worse than predicting the global mean.
  * RANKING fit -- fraction of states where argmax_a pred == argmax_a y. This
                   is what the rollout policy actually consumes, and what the
                   CE term in combined_loss optimizes.

If value R^2 is strongly negative while ranking accuracy is well above chance,
the model is carried entirely by the CE ranking term and its value head is
effectively unconstrained -- which also explains the absence of overfitting.

Usage:
    python check_value_fit_2gene.py [--kind gnn|mlp] [--seed 0]
"""
from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
for p in (ROOT, ROOT / "ground-up-experiments", ROOT / "experiments_after_understanding",
          ROOT / "experiments_after_understanding" / "q_learning", ROOT / "fixing_gnn_q"):
    sys.path.insert(0, str(p))


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    sys.modules[name] = m
    spec.loader.exec_module(m)
    return m


gnn_mod = _load("vf_gnn", HERE / "e6_train_two_gene_gnn_q.py")
mlp_mod = _load("vf_mlp", HERE / "e6_train_two_gene_mlp_q.py")
tg = gnn_mod.tg

TRAIN_FAMILIES = ["Trio", "ThreeGeneration"]   # fair-split TRAIN, see overfit_probes_2gene.py
TEST_FAMILIES  = ["Nuclear"]                   # fair-split TEST (never trained on)


def rank_accuracy(pred, y, s_idx):
    """Fraction of states whose best action under pred matches the best under y.

    Ties in y (states where several actions are exactly optimal) count as a
    match if pred picks any of them -- otherwise degenerate states would be
    scored as failures no policy could pass.
    """
    order = torch.argsort(s_idx)
    s_sorted = s_idx[order]
    bounds = torch.nonzero(torch.diff(s_sorted), as_tuple=False).flatten() + 1
    starts = torch.cat([torch.tensor([0]), bounds])
    ends = torch.cat([bounds, torch.tensor([len(s_sorted)])])
    hit = tot = 0
    for a, b in zip(starts.tolist(), ends.tolist()):
        if b - a < 2:
            continue
        idx = order[a:b]
        yv, pv = y[idx], pred[idx]
        best = (yv == yv.max()).nonzero().flatten()
        hit += int(pv.argmax().item() in set(best.tolist()))
        tot += 1
    return (hit / tot if tot else float("nan")), tot


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--kind", default="gnn", choices=["gnn", "mlp"])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--probe", default="curve")
    args = ap.parse_args()

    dev = torch.device("cpu")
    is_gnn = args.kind == "gnn"
    mod = gnn_mod if is_gnn else mlp_mod
    ck_path = HERE / "results" / f"overfit_probe_{args.probe}" / args.kind / f"seed{args.seed}" / "checkpoint.pt"
    ck = torch.load(ck_path, map_location=dev)
    model = (gnn_mod.GNNQBidirSumPool if is_gnn else mlp_mod.MLPQSumPool)().to(dev)
    model.load_state_dict(ck["model_state"])
    model.eval()
    print(f"{args.kind} seed{args.seed}, checkpoint epoch {ck['epoch']} -- TRAIN configs of the fair split\n")
    print(f"{'split config':<44}{'R2':>8}{'corr':>7}{'sd pred/sd y':>14}{'rank acc':>10}{'states':>9}")

    for fam in TRAIN_FAMILIES + TEST_FAMILIES:
        split = "TRAIN" if fam in TRAIN_FAMILIES else "TEST "
        ped = tg.generate_deterministic_pedigree(tg.FAMILY_CASES[fam])
        ind = ped.to_list()
        sc = tg.compute_structural_features(ped, ind)
        ei = torch.tensor(tg.build_edge_index(ped, ind), device=dev) if is_gnn else None
        for reg in tg.ALLELE_FREQ_REGIMES:
            key = f"{fam}_{reg}_Base_2gene"
            ds = mod.build_config(fam, reg, "Base")
            base = tg.ds_to_tensors(ds, sc, dev)
            s_idx, a_idx, y = mod.build_qsa_index(ds, device=dev, cache_key=key)
            with torch.no_grad():
                chunks = []
                for i in range(0, len(y), 4096):
                    s_b, a_b = s_idx[i:i + 4096], a_idx[i:i + 4096]
                    chunks.append(model(base["nf"][s_b], ei, base["gf"][s_b], a_b) if is_gnn
                                  else model(base["nf"][s_b], base["gf"][s_b], a_b))
                pred = torch.cat(chunks)
            var = y.var(unbiased=False).item()
            r2 = 1 - ((pred - y) ** 2).mean().item() / var
            corr = torch.corrcoef(torch.stack([pred, y]))[0, 1].item()
            acc, n_states = rank_accuracy(pred, y, s_idx)
            print(f"{split} {key:<38}{r2:>8.2f}{corr:>7.2f}"
                  f"{pred.std().item()/y.std().item():>13.1f}x{acc:>10.3f}{n_states:>9,}")


if __name__ == "__main__":
    main()
