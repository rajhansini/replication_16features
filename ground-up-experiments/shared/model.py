"""Neural network value function approximators."""
from __future__ import annotations

import torch
import torch.nn as nn


class PedigreeGNN_Vstar(nn.Module):
    """
    GNN that predicts V*(s) — one scalar per state — matching the ADP formulation.

    ADP approximates V*(s) with basis functions that encode graph structure.
    This model does the same but learns the approximation end-to-end:

        node_feats (beliefs + is_tested + structural)
            ↓  n_rounds of message passing
        node_embeddings (B, N, hidden_dim)
            ↓  global mean pooling
        graph_embedding (B, hidden_dim)
            ↓  concat cost_vec, MLP head
        V*(s)  (B,)  — one scalar per state

    Action selection is EXTERNAL (eval time):
        Q_hat(s,i) = r_test(i,s) + Σ_g P(g|s,i) * V_hat(s')
        action     = argmax_i Q_hat(s,i) over untested persons

    node_feat_dim   = 3*n_genes + 1 + 3   (13 for 3-gene: beliefs + tested + structural)
    global_feat_dim = 3*n_genes + 2        (11 for 3-gene: cost vector)
    output          : (B,) — V*(s) per state
    """

    def __init__(
        self,
        node_feat_dim: int,
        hidden_dim: int = 32,
        n_rounds: int = 2,
        global_feat_dim: int = 0,
    ):
        super().__init__()
        self.n_rounds        = n_rounds
        self.global_feat_dim = global_feat_dim

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

        # Graph-level head: pooled_embedding + cost_vec → scalar V*(s)
        head_in = hidden_dim + global_feat_dim
        self.head = nn.Sequential(
            nn.Linear(head_in, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
        )

    def _mp_step(self, h, src, dst, msg_fn, upd_fn):
        """One message-passing step. h: (B, N, F). Returns (B, N, F_new)."""
        B, N, _ = h.shape
        msg_in = torch.cat([h[:, src, :], h[:, dst, :]], dim=-1)
        msgs   = msg_fn(msg_in)
        H      = msgs.shape[-1]
        agg    = torch.zeros(B, N, H, device=h.device)
        idx    = dst.unsqueeze(0).unsqueeze(-1).expand(B, -1, H)
        agg.scatter_add_(1, idx, msgs)
        # Sum aggregation (not mean): preserves how many parents sent messages,
        # which distinguishes intermediate nodes (Father, 2 parents) from roots (0 parents).
        return upd_fn(torch.cat([h, agg], dim=-1))

    def forward_batch(
        self,
        node_feats: torch.Tensor,                   # (B, N, feat_dim)
        edge_index: torch.Tensor,                   # (2, E)
        global_feats: torch.Tensor | None = None,   # (B, global_feat_dim)
    ) -> torch.Tensor:
        """
        Returns (B,) — V*(s) for each state in the batch.
        Global sum pooling over node embeddings → graph-level representation.
        Sum (not mean) preserves family-size information: a 5-node ThreeGeneration
        graph produces a proportionally larger pooled vector than a 3-node Trio graph.
        """
        h   = node_feats
        src = edge_index[0]
        dst = edge_index[1]
        for msg_fn, upd_fn in zip(self.msg_layers, self.upd_layers):
            h = self._mp_step(h, src, dst, msg_fn, upd_fn)

        pooled = h.sum(dim=1)                        # (B, hidden_dim) — sum preserves N

        if global_feats is not None:
            pooled = torch.cat([pooled, global_feats], dim=-1)   # (B, hidden+G)

        return self.head(pooled).squeeze(-1)         # (B,)

    def forward(
        self,
        node_feats: torch.Tensor,                   # (N, feat_dim)
        edge_index: torch.Tensor,
        global_feats: torch.Tensor | None = None,   # (global_feat_dim,)
    ) -> torch.Tensor:
        """Single-graph forward. Returns scalar V*(s)."""
        gf = global_feats.unsqueeze(0) if global_feats is not None else None
        return self.forward_batch(node_feats.unsqueeze(0), edge_index, gf).squeeze(0)


class MLPValueNet(nn.Module):
    """Flat MLP: belief-state vector → scalar V*(s)."""

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
    GNN over pedigree with cost conditioning — outputs Q(s, i) per node.

    Previous version predicted a single V*(s) per state and selected actions
    externally. This version predicts Q*(s, i) for each person i: the expected
    value of testing person i from belief state s. The action is now explicit
    in the model output — argmax_i Q(s, i) over untested persons gives the
    optimal person to test.

    node_feat_dim   = 3*n_genes + 1   (7 for 2-gene, 10 for 3-gene)
    global_feat_dim = 3*n_genes + 2   (8 for 2-gene, 11 for 3-gene)
    output          : (B, N) — Q value per state per person
    """

    def __init__(
        self,
        node_feat_dim: int,
        hidden_dim: int = 32,
        n_rounds: int = 2,
        global_feat_dim: int = 0,
    ):
        super().__init__()
        self.n_rounds       = n_rounds
        self.global_feat_dim = global_feat_dim

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

        # Per-node head: node embedding + cost vec → Q(s, i) scalar
        head_in = hidden_dim + global_feat_dim
        self.head = nn.Sequential(
            nn.Linear(head_in, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
        )

    def _mp_step(self, h, src, dst, msg_fn, upd_fn):
        """One message-passing step. h: (B, N, F). Returns (B, N, F_new)."""
        B, N, _ = h.shape
        msg_in = torch.cat([h[:, src, :], h[:, dst, :]], dim=-1)  # (B, E, 2F)
        msgs   = msg_fn(msg_in)                                     # (B, E, H)
        H      = msgs.shape[-1]
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
        node_feats: torch.Tensor,                    # (B, N, feat_dim)
        edge_index: torch.Tensor,                    # (2, E)
        global_feats: torch.Tensor | None = None,    # (B, global_feat_dim)
    ) -> torch.Tensor:
        """
        Batched forward for graphs with identical topology.
        Cost vector is broadcast to each node before the head.
        Returns (B, N) — Q(s, i) for each state s and person i.
        """
        h   = node_feats
        src = edge_index[0]
        dst = edge_index[1]
        for msg_fn, upd_fn in zip(self.msg_layers, self.upd_layers):
            h = self._mp_step(h, src, dst, msg_fn, upd_fn)
        # h: (B, N, hidden_dim)
        if global_feats is not None:
            B, N, _ = h.shape
            gf = global_feats.unsqueeze(1).expand(-1, N, -1)  # (B, N, G)
            h  = torch.cat([h, gf], dim=-1)                   # (B, N, hidden+G)
        return self.head(h).squeeze(-1)                        # (B, N)

    def forward(
        self,
        node_feats: torch.Tensor,                    # (N, feat_dim) — single graph
        edge_index: torch.Tensor,                    # (2, E)
        global_feats: torch.Tensor | None = None,    # (global_feat_dim,)
    ) -> torch.Tensor:
        """Single-graph forward. Returns (N,) — Q(s, i) per person."""
        gf = global_feats.unsqueeze(0) if global_feats is not None else None
        return self.forward_batch(node_feats.unsqueeze(0), edge_index, gf).squeeze(0)
