"""Aggregate the GNN-Q depth-ablation results into a table + plots.

Reads:
    results/rounds{N}/seed_runs/seed{S}/results_gnn.json   (avg_ratio2, per_config)
    results/rounds{N}/seed_runs/seed{S}/run.log            (train MSE curve)

Writes (under results/plots/):
    ratio2_vs_rounds.png       mean test ratio2 vs #rounds (+/- std over seeds)
    train_mse_vs_rounds.png    final-epoch train MSE vs #rounds
    ablation_summary.txt       numeric table
    ablation_summary.json      machine-readable

Run after the SLURM array finishes:
    python aggregate_ablation.py
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

HERE = Path(__file__).resolve().parent
ROUND_CHOICES = [0, 1, 2, 3]
SEEDS = [0, 1, 2]
EPOCH_RE = re.compile(r"^\s*epoch\s+(\d+)\s+mse=([\d.]+)")


def load_ratio2(results_dir: Path) -> dict[int, list[float]]:
    out: dict[int, list[float]] = {r: [] for r in ROUND_CHOICES}
    for r in ROUND_CHOICES:
        for s in SEEDS:
            p = results_dir / f"rounds{r}" / "seed_runs" / f"seed{s}" / "results_gnn.json"
            if not p.exists():
                continue
            data = json.loads(p.read_text())
            if "avg_ratio2" in data:
                out[r].append(data["avg_ratio2"])
    return out


def load_final_train_mse(results_dir: Path) -> dict[int, list[float]]:
    out: dict[int, list[float]] = {r: [] for r in ROUND_CHOICES}
    for r in ROUND_CHOICES:
        for s in SEEDS:
            p = results_dir / f"rounds{r}" / "seed_runs" / f"seed{s}" / "run.log"
            if not p.exists():
                continue
            last = None
            for line in p.read_text().splitlines():
                m = EPOCH_RE.match(line)
                if m:
                    last = float(m.group(2))
            if last is not None:
                out[r].append(last)
    return out


def _mean_std(d: dict[int, list[float]]):
    xs, means, stds, ns = [], [], [], []
    for r in ROUND_CHOICES:
        vals = d.get(r, [])
        if not vals:
            continue
        xs.append(r)
        means.append(float(np.mean(vals)))
        stds.append(float(np.std(vals)))
        ns.append(len(vals))
    return xs, means, stds, ns


def plot_metric_vs_rounds(d, ylabel, title, fname, out_dir, logy=False, ref=None, ref_label=None):
    xs, means, stds, _ = _mean_std(d)
    if not xs:
        return
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.errorbar(xs, means, yerr=stds, marker="o", capsize=4, color="#4c78a8",
                linewidth=2, label="GNN-Q (seed mean +/- std)")
    if ref is not None:
        ax.axhline(ref, color="#e45756", linestyle="--", linewidth=1.5,
                   label=ref_label or "reference")
    ax.set_xticks(ROUND_CHOICES)
    ax.set_xlabel("Message-passing rounds  (0 = no MP, per-node linear lift)")
    ax.set_ylabel(ylabel)
    if logy:
        ax.set_yscale("log")
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / fname, dpi=160)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", default=str(HERE / "results"))
    parser.add_argument("--myopic", type=float, default=0.0269,
                        help="myopic baseline ratio2 for reference line")
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    out_dir = results_dir / "plots"
    out_dir.mkdir(parents=True, exist_ok=True)

    ratio2 = load_ratio2(results_dir)
    mse = load_final_train_mse(results_dir)

    plot_metric_vs_rounds(
        ratio2, "Test ratio2 (lower = better)",
        "GNN-Q depth ablation — generalization",
        "ratio2_vs_rounds.png", out_dir,
        ref=args.myopic, ref_label=f"myopic baseline ({args.myopic:.4f})",
    )
    plot_metric_vs_rounds(
        mse, "Final train MSE (Q* regression)",
        "GNN-Q depth ablation — training fit",
        "train_mse_vs_rounds.png", out_dir, logy=True,
    )

    lines = ["GNN-Q message-passing depth ablation", "=" * 60, ""]
    summary = {}
    for r in ROUND_CHOICES:
        rv = ratio2.get(r, [])
        mv = mse.get(r, [])
        label = "no-MP linear lift" if r == 0 else f"{r} round(s)"
        r2_str = (f"{np.mean(rv):.4f} +/- {np.std(rv):.4f}  (n={len(rv)})"
                  if rv else "pending")
        mse_str = (f"{np.mean(mv):.6f}" if mv else "pending")
        lines.append(f"rounds={r} [{label}]")
        lines.append(f"    test ratio2 : {r2_str}")
        lines.append(f"    train MSE   : {mse_str}")
        lines.append("")
        summary[r] = {
            "label": label,
            "ratio2_vals": rv,
            "ratio2_mean": (float(np.mean(rv)) if rv else None),
            "ratio2_std": (float(np.std(rv)) if rv else None),
            "final_train_mse_vals": mv,
            "final_train_mse_mean": (float(np.mean(mv)) if mv else None),
        }

    (out_dir / "ablation_summary.txt").write_text("\n".join(lines))
    (out_dir / "ablation_summary.json").write_text(json.dumps(summary, indent=2))

    print("\n".join(lines))
    print(f"Wrote plots + summary to {out_dir}")


if __name__ == "__main__":
    main()
