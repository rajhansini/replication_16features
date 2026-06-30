"""Training loop for MLP and GNN value function approximators."""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset, random_split


def train_model(
    X: np.ndarray,
    Y: np.ndarray,
    model: nn.Module,
    epochs: int = 500,
    lr: float = 1e-3,
    val_frac: float = 0.2,
    batch_size: int = 32,
    device: str = "cpu",
    print_every: int = 50,
    checkpoint_path: Optional[Path] = None,
    checkpoint_every: int = 50,
) -> tuple[nn.Module, dict]:
    """
    Train model to regress Y from X using MSE loss.

    If checkpoint_path is given, saves a checkpoint every checkpoint_every epochs.
    On startup, if the checkpoint exists, resumes from the saved epoch automatically.

    Returns (trained_model, history).
    """
    model = model.to(device)

    # ── Resume from checkpoint if available ───────────────────────────────────
    start_epoch = 1
    history     = {"train_loss": [], "val_loss": []}

    if checkpoint_path is not None and Path(checkpoint_path).exists():
        ckpt = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(ckpt["model_state"])
        start_epoch = ckpt["epoch"] + 1
        history     = ckpt.get("history", history)
        optimizer_state = ckpt.get("optimizer_state")
        print(f"  Resumed from checkpoint at epoch {ckpt['epoch']}/{epochs}")
    else:
        optimizer_state = None

    if start_epoch > epochs:
        print(f"  Already completed {epochs} epochs — skipping training.")
        return model, history

    # ── Data ──────────────────────────────────────────────────────────────────
    X_t = torch.FloatTensor(X).to(device)
    Y_t = torch.FloatTensor(Y).to(device)

    dataset = TensorDataset(X_t, Y_t)
    n_val   = max(1, int(len(dataset) * val_frac))
    n_train = len(dataset) - n_val
    train_ds, val_ds = random_split(dataset, [n_train, n_val])

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader   = DataLoader(val_ds,   batch_size=len(val_ds))

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    if optimizer_state is not None:
        optimizer.load_state_dict(optimizer_state)

    criterion = nn.MSELoss()

    # ── Training loop ─────────────────────────────────────────────────────────
    for epoch in range(start_epoch, epochs + 1):
        model.train()
        train_loss = 0.0
        for xb, yb in train_loader:
            optimizer.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * len(xb)
        train_loss /= n_train

        model.eval()
        with torch.no_grad():
            for xb, yb in val_loader:
                val_loss = criterion(model(xb), yb).item()

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)

        if epoch % print_every == 0:
            print(f"  epoch {epoch:4d}/{epochs}  "
                  f"train_loss={train_loss:.6f}  val_loss={val_loss:.6f}")

        # Save checkpoint periodically
        if checkpoint_path is not None and epoch % checkpoint_every == 0:
            tmp = Path(str(checkpoint_path) + ".tmp")
            torch.save({
                "epoch":          epoch,
                "model_state":    model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "history":        history,
            }, tmp)
            tmp.replace(checkpoint_path)

    return model, history
