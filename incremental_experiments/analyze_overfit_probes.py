"""Analysis for the 2-gene overfit probes (overfit_probes_2gene.py).

Reads the probe outputs and prints the tables that actually support a claim,
with two corrections applied that the raw per-probe averages do NOT have:

1. REGIME DIFFICULTY MATCHING. configholdout withholds whole regimes
   (MixedA/MixedB), so its "seen" and "held-out" sets contain DIFFERENT
   regimes. Regime difficulty varies by more than an order of magnitude, so
   comparing those averages directly is meaningless -- the raw numbers make
   held-out configs look 3-4x BETTER than trained ones purely because the
   trained set contains LowLow and the held-out set does not. The comparisons
   here are matched: held-out vs trained-excluding-LowLow for the config-level
   gap, and all-6-regimes vs all-6-regimes for the family-level gap.

2. ratio2 IS UNSTABLE AT SMALL |V*|. ratio2 is a regret normalized by the
   optimal value, so configs whose optimal value is near zero inflate it.
   LowLow (both alleles at 0.02, the rarest variants) has the smallest |V*|
   and the largest ratio2 by far -- but its ABSOLUTE regret is ordinary. The
   |V*| table below is the evidence; treat any ratio2 average that mixes
   regimes as partly a normalization artifact.

Usage:
    python analyze_overfit_probes.py
"""
from __future__ import annotations

import json
import re
import statistics
from pathlib import Path

R = Path(__file__).resolve().parent / "results"
PAT = re.compile(r"(\w+?)_(LowHigh|MediumEven|LowLow|HighHigh|MixedA|MixedB)_(Base|Aggressive)_2gene")
SEEDS = (0, 1, 2)
KINDS = ("gnn", "mlp")


def load(probe, kind, seed, name):
    f = R / f"overfit_probe_{probe}" / kind / f"seed{seed}" / name
    return json.loads(f.read_text()) if f.exists() else None


def mean_r2(dd, keep=None):
    vals = [v["ratio2"] for k, v in dd.items() if keep is None or keep(PAT.match(k).group(2))]
    return statistics.mean(vals) if vals else float("nan")


def hdr(t):
    print(f"\n{'='*78}\n{t}\n{'='*78}")


def curves():
    hdr("PROBE 1 -- curve: does held-out validation loss ever climb? (overfitting signature)")
    print(f"{'kind':<5}{'seed':<6}{'epochs':>8}{'train first->last':>22}{'val min':>10}{'val final':>11}{'rise':>9}")
    for kind in KINDS:
        for s in SEEDS:
            c = load("curve", kind, s, "curve.json")
            if not c or "val_loss" not in c[0]:
                continue
            vl = [p["val_loss"] for p in c]
            tl = [p["train_loss"] for p in c]
            i = min(range(len(vl)), key=lambda j: vl[j])
            print(f"{kind:<5}{s:<6}{len(c):>8}{f'{tl[0]:.4f} -> {tl[-1]:.4f}':>22}"
                  f"{vl[i]:>10.5f}{vl[-1]:>11.5f}{vl[-1]-vl[i]:>+9.5f}")
    print("\n  A rise of a few 1e-3 against a loss of ~0.22 is noise, not overfitting.")


def randlabel():
    hdr("PROBE 2 -- randlabel: can the model fit SHUFFLED targets? (capacity to memorize)")
    print("  NOTE: the optimized loss is mse + lambda*ce. Under shuffling the CE term is")
    print("  unsatisfiable, so the optimizer redirects capacity into MSE -- raw train MSE")
    print("  alone is NOT a valid comparison. Total loss is the honest one.")
    print(f"\n{'kind':<5}{'seed':<6}{'real mse':>11}{'shuf mse':>11}{'real loss':>12}{'shuf loss':>12}{'loss delta':>12}")
    for kind in KINDS:
        for s in SEEDS:
            a, b = load("curve", kind, s, "results.json"), load("randlabel", kind, s, "results.json")
            if not a or not b:
                continue
            print(f"{kind:<5}{s:<6}{a['final_train_mse']:>11.5f}{b['final_train_mse']:>11.5f}"
                  f"{a['final_train_loss']:>12.5f}{b['final_train_loss']:>12.5f}"
                  f"{b['final_train_loss']-a['final_train_loss']:>+12.5f}")


def configholdout():
    hdr("PROBE 3 -- configholdout, RAW (confounded -- shown only to explain the correction)")
    print(f"{'kind':<5}{'seed':<6}{'seen':>9}{'held-out':>10}{'unseen fam':>12}")
    for kind in KINDS:
        for s in SEEDS:
            d = load("configholdout", kind, s, "results.json")
            if not d:
                continue
            print(f"{kind:<5}{s:<6}{d['ratio2_seen_configs']:>9.4f}"
                  f"{d['ratio2_heldout_configs_seen_families']:>10.4f}{d['ratio2_unseen_family']:>12.4f}")
    print("\n  Held-out looks far BETTER than seen. That is an artifact: 'seen' contains LowLow,")
    print("  'held-out' (MixedA/MixedB) does not. Corrected below.")

    hdr("PROBE 3a -- CONFIG-level gap, difficulty-matched (held-out vs trained EXCLUDING LowLow)")
    print(f"{'kind':<5}{'seed':<6}{'trained (no LowLow)':>21}{'held-out':>10}{'ratio':>8}")
    for kind in KINDS:
        for s in SEEDS:
            a, b = load("configholdout", kind, s, "eval_seen_partial.json"), load("configholdout", kind, s, "eval_heldout_partial.json")
            if not a or not b:
                continue
            tr, he = mean_r2(a, lambda r: r != "LowLow"), mean_r2(b)
            print(f"{kind:<5}{s:<6}{tr:>21.4f}{he:>10.4f}{he/tr:>7.2f}x")

    hdr("PROBE 3b -- FAMILY-level gap, regime-matched (all 6 regimes both sides)")
    print(f"{'kind':<5}{'seed':<6}{'train families':>16}{'unseen family':>15}{'ratio':>8}")
    for kind in KINDS:
        for s in SEEDS:
            a, b, c = (load("configholdout", kind, s, n) for n in
                       ("eval_seen_partial.json", "eval_heldout_partial.json", "eval_unseen_partial.json"))
            if not a or not b or not c:
                continue
            tr, un = mean_r2({**a, **b}), mean_r2(c)
            print(f"{kind:<5}{s:<6}{tr:>16.4f}{un:>15.4f}{un/tr:>7.2f}x")


def metric_artifact():
    hdr("METRIC CHECK -- is ratio2 driven by the size of the optimal value?")
    rows = []
    for kind in KINDS:
        for s in SEEDS:
            for n in ("eval_seen_partial.json", "eval_heldout_partial.json", "eval_unseen_partial.json"):
                d = load("configholdout", kind, s, n)
                if not d:
                    continue
                for k, v in d.items():
                    rows.append((PAT.match(k).group(2), kind, abs(v["V_root"]), v["ratio2"], abs(v["L"] - v["V_root"])))
    if not rows:
        return
    print(f"{'regime':<12}{'alleles':>14}{'|V*|':>9}{'ratio2':>10}{'abs regret |L-V*|':>19}")
    freqs = {"LowLow": "0.02 / 0.02", "MixedA": "0.02 / 0.10", "LowHigh": "0.02 / 0.15",
             "MixedB": "0.05 / 0.12", "MediumEven": "0.08 / 0.08", "HighHigh": "0.15 / 0.15"}
    by = {}
    for r, _, v, r2, ab in rows:
        by.setdefault(r, []).append((v, r2, ab))
    for r in sorted(by, key=lambda x: statistics.mean([a for a, _, _ in by[x]])):
        v = by[r]
        print(f"{r:<12}{freqs.get(r,''):>14}{statistics.mean([a for a,_,_ in v]):>9.4f}"
              f"{statistics.mean([b for _,b,_ in v]):>10.4f}{statistics.mean([c for _,_,c in v]):>19.4f}")
    xs = [v for _, _, v, _, _ in rows]
    ys = [r2 for _, _, _, r2, _ in rows]
    inv = [1 / x for x in xs]
    n, my = len(xs), statistics.mean(ys)
    mi = statistics.mean(inv)
    corr = (sum((a - mi) * (y - my) for a, y in zip(inv, ys)) / n) / (statistics.pstdev(inv) * statistics.pstdev(ys))
    print(f"\n  Pearson corr(1/|V*|, ratio2) over {n} config-evals = {corr:+.3f}")
    print("  LowLow's ABSOLUTE regret is ordinary; its ratio2 explodes because |V*| is ~4x smaller.")


if __name__ == "__main__":
    curves()
    randlabel()
    configholdout()
    metric_artifact()
