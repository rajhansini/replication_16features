"""Neural network value function approximators."""
from __future__ import annotations

import torch
import torch.nn as nn


class MLPValueNet(nn.Module):
    """
    Flat MLP that maps a belief-state vector to a scalar V*(s) estimate.

    input_dim = n_people * 3  (single gene)
              = n_people * 6  (two genes)
    """

    def __init__(self, input_dim: int, hidden_dims: tuple = (64, 32)):
        super().__init__()
        layers: list[nn.Module] = []
        prev = input_dim
        for h in hidden_dims:
            layers.append(nn.Linear(prev, h))
            layers.append(nn.ReLU())
            prev = h
        layers.append(nn.Linear(prev, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


class PedigreeGNN(nn.Module):
    """
    Simple graph neural network over the pedigree.

    Each person is a node with node_feat_dim features.
    Two rounds of message passing along pedigree edges.
    Global mean pool → MLP head → scalar V*(s).

    node_feat_dim = 3 + 1 = 4  (single gene: 3 probs + is_tested flag)
                 = 6 + 1 = 7  (two genes: 6 probs + is_tested flag)
    """

    def __init__(
        self,
        node_feat_dim: int,
        hidden_dim: int = 32,
        n_rounds: int = 2,
    ):
        super().__init__()
        self.n_rounds = n_rounds

        # Per-round message + update MLPs
        self.msg_layers = nn.ModuleList()
        self.upd_layers = nn.ModuleList()
        in_dim = node_feat_dim
        for _ in range(n_rounds):
            self.msg_layers.append(nn.Sequential(
                nn.Linear(in_dim * 2, hidden_dim), nn.ReLU(),
            ))
            self.upd_layers.append(nn.Sequential(
                nn.Linear(in_dim + hidden_dim, hidden_dim), nn.ReLU(),
            ))
            in_dim = hidden_dim

        # Readout head
        self.head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(
        self,
        node_feats: torch.Tensor,   # (n_nodes, feat_dim)   — single graph
        edge_index: torch.Tensor,   # (2, n_edges)
    ) -> torch.Tensor:
        """Single-graph forward (used during eval). Returns scalar."""
        h   = self.forward_batch(node_feats.unsqueeze(0), edge_index)  # (1,)
        return h.squeeze(0)                                             # scalar

    def _mp_step(self, h, src, dst, msg_fn, upd_fn):
        """One message-passing step. h: (B, N, F). Returns (B, N, F_new)."""
        B, N, _ = h.shape
        # h[:, src, :] gathers src features for all graphs at once
        msg_in = torch.cat([h[:, src, :], h[:, dst, :]], dim=-1)  # (B, E, 2F)
        msgs   = msg_fn(msg_in)                                     # (B, E, H)
        H      = msgs.shape[-1]
        # Scatter-mean into destination nodes
        agg    = torch.zeros(B, N, H, device=h.device)
        idx    = dst.unsqueeze(0).unsqueeze(-1).expand(B, -1, H)
        agg.scatter_add_(1, idx, msgs)
        cnt    = torch.zeros(B, N, 1, device=h.device)
        cnt.scatter_add_(1, dst.unsqueeze(0).unsqueeze(-1).expand(B, -1, 1),
                         torch.ones(B, len(dst), 1, device=h.device))
        agg    = agg / cnt.clamp(min=1.0)
        return upd_fn(torch.cat([h, agg], dim=-1))                 # (B, N, F_new)

    def forward_batch(
        self,
        node_feats: torch.Tensor,   # (B, n_nodes, feat_dim) — same topology batch
        edge_index: torch.Tensor,   # (2, n_edges) — ONE graph's edges (shared topology)
    ) -> torch.Tensor:
        """
        Batched forward for graphs with identical topology.
        All graphs in a dataset share the same edge_index.
        Returns (B,) value estimates.
        """
        h   = node_feats                    # (B, N, F)
        src = edge_index[0]
        dst = edge_index[1]
        for msg_fn, upd_fn in zip(self.msg_layers, self.upd_layers):
            h = self._mp_step(h, src, dst, msg_fn, upd_fn)
        graph_repr = h.mean(dim=1)          # (B, hidden)
        return self.head(graph_repr).squeeze(-1)   # (B,)
