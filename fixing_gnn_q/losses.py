"""Combined MSE + cross-entropy loss for Q(s,a) training.

Why: plain MSE on (state, action, Q*) rows only asks "is this number close to
its target" — it never checks whether the predicted argmax within a state
matches the true argmax. At states where the true Q-gap between candidates is
small (e.g. the root state, before any evidence), MSE tolerates errors that
silently flip the greedy policy's pick.

This module reshapes the flat (state_idx, action_idx, y) row layout that
qsa_data.build_qsa_index() produces into per-state groups, padded to the max
number of candidates any state in the config has, so a listwise cross-entropy
loss (softmax over the state's own candidates, target = true best action) can
be computed in one vectorized pass alongside MSE.

    total_loss = mse_loss + lambda_ce * cross_entropy_loss

MSE keeps Q_hat's absolute scale meaningful (needed for the STOP-vs-TEST
comparison against v_stop elsewhere). Cross-entropy is what actually trains
the *policy* — it directly penalizes wrong-argmax predictions, independent of
how small the true value gap between candidates was.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F


def build_state_groups(state_idx: torch.Tensor, y: torch.Tensor, k_max: int):
    """Group (state, action, Q*) rows by state, padded to k_max candidates.

    state_idx must be non-decreasing (true by construction in
    qsa_data.build_qsa_index — rows for a state are emitted contiguously).

    Returns:
        group_rows   : (G, k_max) long  — row index into state_idx/y for each
                        (group, slot); -1 where the group has fewer than
                        k_max candidates.
        group_mask   : (G, k_max) bool  — True where group_rows is valid.
        group_target : (G,)       long  — local slot index of the true best
                        action (argmax Q*) within each group.
    """
    device = state_idx.device
    n_rows = state_idx.shape[0]

    uniq, counts = torch.unique_consecutive(state_idx, return_counts=True)
    n_groups = uniq.shape[0]

    group_of_row = torch.repeat_interleave(torch.arange(n_groups, device=device), counts)
    starts = torch.cumsum(counts, dim=0) - counts
    pos_in_group = torch.arange(n_rows, device=device) - torch.repeat_interleave(starts, counts)

    row_ids = torch.arange(n_rows, device=device)
    group_rows = torch.full((n_groups, k_max), -1, dtype=torch.long, device=device)
    group_rows[group_of_row, pos_in_group] = row_ids

    group_y = torch.full((n_groups, k_max), float("-inf"), device=device)
    group_y[group_of_row, pos_in_group] = y

    group_mask = group_rows >= 0
    group_target = group_y.argmax(dim=1)
    return group_rows, group_mask, group_target


def state_depths(states: list, state_idx: torch.Tensor) -> torch.Tensor:
    """Per-row depth = len(state) = number of people already tested at that
    row's state. Used to build the depth-reweighting term (see depth_weight).
    """
    depth_by_state = torch.tensor([len(s) for s in states], dtype=torch.float32)
    return depth_by_state[state_idx.cpu()].to(state_idx.device)


def depth_weight(depth: torch.Tensor) -> torch.Tensor:
    """Upweight shallow (low-depth) states, root heaviest.

    Root is 1 state out of >1M — under uniform row/state weighting its
    contribution to the total loss is negligible, so gradient descent has
    no incentive to fix it even when it's technically included in a batch.
    This weight makes depth-0 states worth 1.0, depth-1 worth 0.5, etc.,
    so shallow states are not just *included* but *count for more*.
    """
    return 1.0 / (1.0 + depth)


def combined_loss(pred_groups: torch.Tensor, group_mask: torch.Tensor,
                   group_target: torch.Tensor, y_groups: torch.Tensor,
                   lambda_ce: float, group_weight: torch.Tensor | None = None):
    """pred_groups, y_groups: (B, k_max), padded with garbage where ~group_mask.
    group_weight: (B,) optional per-group weight (see depth_weight); None = uniform.

    Returns (total_loss, mse_component, ce_component) — components detached
    floats for logging, total_loss is the tensor to backprop through.
    """
    masked_pred = pred_groups.masked_fill(~group_mask, float("-inf"))

    if group_weight is None:
        ce = F.cross_entropy(masked_pred, group_target)
        mse = F.mse_loss(pred_groups[group_mask], y_groups[group_mask])
    else:
        ce_per_group = F.cross_entropy(masked_pred, group_target, reduction="none")
        ce = (ce_per_group * group_weight).sum() / group_weight.sum()

        row_weight = group_weight.unsqueeze(1).expand_as(pred_groups)[group_mask]
        sq_err = (pred_groups[group_mask] - y_groups[group_mask]) ** 2
        mse = (sq_err * row_weight).sum() / row_weight.sum()

    total = mse + lambda_ce * ce
    return total, mse.item(), ce.item()
