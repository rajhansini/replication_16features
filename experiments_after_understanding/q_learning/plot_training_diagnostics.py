"""Parse Q-track run.log files and plot training MSE + weight diagnostics.

Usage:
    python plot_training_diagnostics.py
    python plot_training_diagnostics.py --results-dir sanity_check_results
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

HERE = Path(__file__).resolve().parent
EXP_ROOT = HERE.parent
sys.path.insert(0, str(EXP_ROOT))
sys.path.insert(0, str(HERE))

from gnn_q import GNNQ  # noqa: E402
from mlp_q import MLPQ  # noqa: E402

EPOCH_RE = re.compile(r"^\s*epoch\s+(\d+)\s+mse=([\d.]+)")
RUN_HDR_RE = re.compile(r"\[(MLP-Q|GNN-Q)\].*?mode=(\w+).*?epochs=(\d+)")


def split_interleaved_curves(pairs: list[tuple[int, float]]) -> tuple[list, list]:
    """Split interleaved (epoch, mse) lines into two monotonic epoch sequences."""
    seq_a: list[tuple[int, float]] = []
    seq_b: list[tuple[int, float]] = []
    for epoch, mse in pairs:
        if not seq_a or epoch > seq_a[-1][0]:
            seq_a.append((epoch, mse))
        elif not seq_b or epoch > seq_b[-1][0]:
            seq_b.append((epoch, mse))
        else:
            # Rare tie-break: higher MSE usually MLP
            if mse >= seq_a[-1][1]:
                seq_b.append((epoch, mse))
            else:
                seq_a.append((epoch, mse))
    return seq_a, seq_b


def parse_training_section(log_text: str, target_epochs: int = 500) -> dict[str, list[tuple[int, float]]]:
    """Extract MLP-Q and GNN-Q train MSE curves from the latest mode=both run."""
    lines = log_text.splitlines()
    # Find last simultaneous both-mode 500-epoch block
    mlp_start = gnn_start = None
    for i, line in enumerate(lines):
        m = RUN_HDR_RE.search(line)
        if not m:
            continue
        tag, mode, epochs = m.group(1), m.group(2), int(m.group(3))
        if mode != "both" or epochs != target_epochs:
            continue
        if tag == "MLP-Q":
            mlp_start = i
        else:
            gnn_start = i

    if mlp_start is None or gnn_start is None:
        return {"MLP-Q": [], "GNN-Q": []}

    start = min(mlp_start, gnn_start)
    # Training window: first [2] Training after headers until gnn_q.pt saved
    train_begin = None
    for i in range(start, len(lines)):
        if "[2] Training" in lines[i]:
            train_begin = i
            break
    if train_begin is None:
        return {"MLP-Q": [], "GNN-Q": []}

    pairs: list[tuple[int, float]] = []
    for line in lines[train_begin:]:
        if "saved ->" in line and "gnn_q.pt" in line:
            break
        m = EPOCH_RE.match(line)
        if m:
            pairs.append((int(m.group(1)), float(m.group(2))))

    seq_a, seq_b = split_interleaved_curves(pairs)
    # MLP starts with higher epoch-1 MSE
    if seq_a and seq_b:
        if seq_a[0][1] >= seq_b[0][1]:
            return {"MLP-Q": seq_a, "GNN-Q": seq_b}
        return {"MLP-Q": seq_b, "GNN-Q": seq_a}
    return {"MLP-Q": seq_a, "GNN-Q": seq_b}


def load_state_dict(path: Path) -> dict:
    obj = torch.load(path, map_location="cpu", weights_only=False)
    if isinstance(obj, dict) and "model_state" in obj:
        return obj["model_state"]
    return obj


def weight_stats(state_dict: dict) -> dict[str, dict[str, float]]:
    out = {}
    for name, t in state_dict.items():
        if not torch.is_floating_point(t):
            continue
        flat = t.detach().float().view(-1)
        out[name] = {
            "norm": flat.norm().item(),
            "mean": flat.mean().item(),
            "std": flat.std(unbiased=False).item(),
            "abs_mean": flat.abs().mean().item(),
            "n": flat.numel(),
        }
    return out


def plot_loss_curves(curves_by_seed: dict[int, dict[str, list]], out_dir: Path, title_suffix: str):
    fig, ax = plt.subplots(figsize=(9, 5.5))
    colors = {"MLP-Q": "#e45756", "GNN-Q": "#4c78a8"}
    for model in ("MLP-Q", "GNN-Q"):
        for seed, curves in sorted(curves_by_seed.items()):
            pts = curves.get(model, [])
            if not pts:
                continue
            xs, ys = zip(*pts)
            ax.plot(xs, ys, color=colors[model], alpha=0.45, linewidth=1.2)
        # mean curve
        all_epochs = sorted({e for c in curves_by_seed.values() for e, _ in c.get(model, [])})
        if not all_epochs:
            continue
        mean_y = []
        for e in all_epochs:
            vals = [mse for c in curves_by_seed.values() for ep, mse in c.get(model, []) if ep == e]
            mean_y.append(np.mean(vals))
        ax.plot(all_epochs, mean_y, color=colors[model], linewidth=2.5, label=f"{model} (seed mean)")

    ax.set_xlabel("Epoch")
    ax.set_ylabel("Train MSE (Q* regression)")
    ax.set_yscale("log")
    ax.set_title(f"Q-track training loss — 500 epochs{title_suffix}")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "train_mse_curves.png", dpi=160)
    plt.close(fig)


def plot_final_mse_bar(curves_by_seed: dict[int, dict[str, list]], out_dir: Path, title_suffix: str):
    models = ["MLP-Q", "GNN-Q"]
    seeds = sorted(curves_by_seed)
    fig, ax = plt.subplots(figsize=(7, 4.5))
    x = np.arange(len(seeds))
    width = 0.35
    for i, model in enumerate(models):
        vals = []
        for seed in seeds:
            pts = curves_by_seed[seed].get(model, [])
            vals.append(pts[-1][1] if pts else np.nan)
        if all(np.isnan(v) for v in vals):
            continue
        ax.bar(x + (i - 0.5) * width, vals, width, label=model)
    ax.set_xticks(x)
    ax.set_xticklabels([f"seed {s}" for s in seeds])
    ax.set_ylabel("Final train MSE (epoch 500)")
    ax.set_yscale("log")
    ax.set_title(f"Final training MSE by seed{title_suffix}")
    ax.legend()
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_dir / "final_train_mse_by_seed.png", dpi=160)
    plt.close(fig)


def plot_weight_norms(seed_dir: Path, seed: int, out_dir: Path):
    mlp_path = seed_dir / "mlp_q.pt"
    gnn_path = seed_dir / "gnn_q.pt"
    if not mlp_path.exists() or not gnn_path.exists():
        return

    mlp_sd = load_state_dict(mlp_path)
    gnn_sd = load_state_dict(gnn_path)
    mlp_stats = weight_stats(mlp_sd)
    gnn_stats = weight_stats(gnn_sd)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), sharey=True)
    for ax, (label, stats) in zip(axes, [("MLP-Q", mlp_stats), ("GNN-Q", gnn_stats)]):
        names = list(stats.keys())
        norms = [stats[n]["norm"] for n in names]
        short = [n.replace(".weight", " W").replace(".bias", " b") for n in names]
        ax.barh(short, norms, color="#72b7b2" if "MLP" in label else "#4c78a8")
        ax.set_title(f"{label} — seed {seed}")
        ax.set_xlabel("L2 norm")
        ax.invert_yaxis()
    fig.suptitle("Final weight L2 norms by parameter tensor")
    fig.tight_layout()
    fig.savefig(out_dir / f"weight_norms_seed{seed}.png", dpi=160)
    plt.close(fig)


def plot_weight_histograms(seed_dir: Path, seed: int, out_dir: Path):
    mlp_sd = load_state_dict(seed_dir / "mlp_q.pt")
    gnn_sd = load_state_dict(seed_dir / "gnn_q.pt")
    mlp_w = torch.cat([t.detach().float().view(-1) for n, t in mlp_sd.items() if "weight" in n])
    gnn_w = torch.cat([t.detach().float().view(-1) for n, t in gnn_sd.items() if "weight" in n])

    fig, ax = plt.subplots(figsize=(8, 4.5))
    bins = np.linspace(
        min(mlp_w.min().item(), gnn_w.min().item()),
        max(mlp_w.max().item(), gnn_w.max().item()),
        40,
    )
    ax.hist(mlp_w.numpy(), bins=bins, alpha=0.55, label=f"MLP-Q ({mlp_w.numel()} weights)", color="#e45756")
    ax.hist(gnn_w.numpy(), bins=bins, alpha=0.55, label=f"GNN-Q ({gnn_w.numel()} weights)", color="#4c78a8")
    ax.set_xlabel("Weight value")
    ax.set_ylabel("Count")
    ax.set_title(f"Final weight distributions — seed {seed}")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / f"weight_hist_seed{seed}.png", dpi=160)
    plt.close(fig)


def plot_cross_seed_weight_drift(results_root: Path, seeds: list[int], out_dir: Path):
    """Compare final MLP/GNN head weight norms across seeds."""
    for model, fname, ctor in [
        ("MLP-Q", "mlp_q.pt", MLPQ),
        ("GNN-Q", "gnn_q.pt", GNNQ),
    ]:
        tensors_by_seed = {}
        for seed in seeds:
            path = results_root / "seed_runs" / f"seed{seed}" / fname
            if not path.exists():
                continue
            sd = load_state_dict(path)
            ref = ctor()
            ref.load_state_dict(sd)
            flat = torch.cat([p.detach().view(-1) for p in ref.parameters()])
            tensors_by_seed[seed] = flat

        if len(tensors_by_seed) < 2:
            continue
        seed_list = sorted(tensors_by_seed)
        ref = tensors_by_seed[seed_list[0]]
        dists = [torch.dist(ref, tensors_by_seed[s]).item() for s in seed_list]

        fig, ax = plt.subplots(figsize=(6, 4))
        ax.bar([f"seed {s}" for s in seed_list], dists, color="#f58518")
        ax.set_ylabel(f"L2 distance to seed {seed_list[0]} weights")
        ax.set_title(f"{model} — cross-seed weight drift")
        fig.tight_layout()
        fig.savefig(out_dir / f"cross_seed_drift_{model.replace('-', '_').lower()}.png", dpi=160)
        plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--results-dir",
        default="results",
        help="Subdir of q_learning/ containing seed_runs/ (default: results)",
    )
    parser.add_argument("--epochs", type=int, default=500)
    args = parser.parse_args()

    results_root = HERE / args.results_dir
    out_dir = results_root / "plots"
    out_dir.mkdir(parents=True, exist_ok=True)

    seed_dirs = sorted((results_root / "seed_runs").glob("seed[0-9]*"))
    seeds = [int(d.name.replace("seed", "")) for d in seed_dirs if d.name.replace("seed", "").isdigit()]
    seeds = [s for s in seeds if s in (0, 1, 2)]

    curves_by_seed: dict[int, dict[str, list]] = {}
    for seed in seeds:
        log_path = results_root / "seed_runs" / f"seed{seed}" / "run.log"
        if not log_path.exists():
            continue
        curves = parse_training_section(log_path.read_text(), target_epochs=args.epochs)
        curves_by_seed[seed] = curves
        plot_weight_norms(results_root / "seed_runs" / f"seed{seed}", seed, out_dir)
        plot_weight_histograms(results_root / "seed_runs" / f"seed{seed}", seed, out_dir)

    if not curves_by_seed:
        print(f"No training curves found under {results_root}")
        return

    suffix = f" ({args.results_dir})"
    plot_loss_curves(curves_by_seed, out_dir, suffix)
    plot_final_mse_bar(curves_by_seed, out_dir, suffix)
    plot_cross_seed_weight_drift(results_root, seeds, out_dir)

    # Text summary
    summary_path = out_dir / "training_summary.txt"
    with summary_path.open("w") as f:
        f.write(f"Q-track training diagnostics — {args.results_dir}\n")
        f.write("=" * 60 + "\n\n")
        for seed, curves in sorted(curves_by_seed.items()):
            f.write(f"seed {seed}:\n")
            for model in ("MLP-Q", "GNN-Q"):
                pts = curves.get(model, [])
                if not pts:
                    f.write(f"  {model}: no data\n")
                    continue
                f.write(f"  {model}: epoch1={pts[0][1]:.6f}  epoch500={pts[-1][1]:.6f}  n_logged={len(pts)}\n")
            f.write("\n")

    print(f"Wrote plots to {out_dir}")
    for p in sorted(out_dir.glob("*.png")):
        print(f"  {p.name}")


if __name__ == "__main__":
    main()
