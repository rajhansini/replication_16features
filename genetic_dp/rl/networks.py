"""
Actor-Critic networks for sequential genetic testing.

Two variants:
  MLPActorCritic  — flat MLP (fast baseline, no graph structure)
  GNNActorCritic  — graph-aware: per-node embeddings → actor logits + pooled value

Both take:
  obs      : (batch, obs_dim) — carrier probs + tested flags from the env
  adj      : (N, N) pedigree adjacency matrix (for GNN only)
  mask     : (batch, N+1) bool — True for valid actions

GNN message-passing is implemented via sparse matrix operations without
torch_geometric. Architecture: 2-layer GraphSAGE (mean aggregation).
"""

from __future__ import annotations

from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _init_weights(module: nn.Module) -> None:
    for m in module.modules():
        if isinstance(m, nn.Linear):
            nn.init.orthogonal_(m.weight, gain=1.0)
            if m.bias is not None:
                nn.init.zeros_(m.bias)


def masked_softmax(logits: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Softmax with -inf masking for invalid actions."""
    logits = logits.clone()
    logits[~mask] = float("-inf")
    return logits


# ---------------------------------------------------------------------------
# MLP Actor-Critic
# ---------------------------------------------------------------------------

class MLPActorCritic(nn.Module):
    """Shared-trunk MLP with separate actor and critic heads.

    Input layout (obs_dim = N*G + N):
      [carrier_probs (N*G)] [tested_flags (N)]
    """

    def __init__(
        self,
        obs_dim: int,
        n_actions: int,
        hidden: List[int] = (256, 128),
    ):
        super().__init__()
        self.obs_dim = obs_dim
        self.n_actions = n_actions

        layers: List[nn.Module] = []
        in_dim = obs_dim
        for h in hidden:
            layers += [nn.Linear(in_dim, h), nn.Tanh()]
            in_dim = h
        self.trunk = nn.Sequential(*layers)

        self.actor_head = nn.Linear(in_dim, n_actions)
        self.critic_head = nn.Linear(in_dim, 1)
        _init_weights(self)

    def forward(
        self,
        obs: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return (logits, value).  logits are already masked (mask optional)."""
        x = self.trunk(obs)
        logits = self.actor_head(x)
        if mask is not None:
            logits = masked_softmax(logits, mask)
        value = self.critic_head(x).squeeze(-1)
        return logits, value

    def act(
        self,
        obs: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        deterministic: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Sample action. Returns (action, log_prob, value)."""
        logits, value = self.forward(obs, mask)
        dist = Categorical(logits=logits)
        action = logits.argmax(-1) if deterministic else dist.sample()
        return action, dist.log_prob(action), value

    def evaluate(
        self,
        obs: torch.Tensor,
        actions: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return (log_probs, values, entropy) for PPO update."""
        logits, value = self.forward(obs, mask)
        dist = Categorical(logits=logits)
        return dist.log_prob(actions), value, dist.entropy()


# ---------------------------------------------------------------------------
# GNN message-passing block (no torch_geometric)
# ---------------------------------------------------------------------------

class GraphSAGELayer(nn.Module):
    """One layer of GraphSAGE with mean aggregation.

    h_v' = LayerNorm(ReLU(W_self * h_v + W_neigh * mean_{u∈N(v)} h_u))
    """

    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.W_self = nn.Linear(in_dim, out_dim, bias=False)
        self.W_neigh = nn.Linear(in_dim, out_dim, bias=False)
        self.norm = nn.LayerNorm(out_dim)
        _init_weights(self)

    def forward(
        self, h: torch.Tensor, adj_norm: torch.Tensor
    ) -> torch.Tensor:
        """
        h         : (N, in_dim)
        adj_norm  : (N, N) row-normalised adjacency (mean aggregation)
        Returns   : (N, out_dim)
        """
        agg = adj_norm @ h  # (N, in_dim) — mean of neighbours
        out = self.W_self(h) + self.W_neigh(agg)
        return self.norm(F.relu(out))


class PedigreeGNN(nn.Module):
    """Two-layer GraphSAGE operating on pedigree graph.

    Node features per individual:
      [carrier_prob (G values), tested_flag, generation_depth_normalised]

    After 2 rounds of message passing, returns:
      node_emb : (batch, N, emb_dim) — per-individual embeddings
      graph_emb: (batch, emb_dim)    — mean-pooled graph embedding
    """

    def __init__(
        self,
        node_in_dim: int,
        emb_dim: int = 64,
        n_layers: int = 2,
    ):
        super().__init__()
        self.layers = nn.ModuleList()
        in_d = node_in_dim
        for _ in range(n_layers):
            self.layers.append(GraphSAGELayer(in_d, emb_dim))
            in_d = emb_dim
        self.emb_dim = emb_dim

    def forward(
        self,
        node_feats: torch.Tensor,
        adj_norm: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        node_feats : (batch, N, node_in_dim)
        adj_norm   : (N, N)  pre-computed normalised adjacency
        """
        batch_size = node_feats.shape[0]
        h = node_feats  # (batch, N, d)

        for layer in self.layers:
            # Apply same adjacency to each element of the batch
            h_out = torch.stack([layer(h[b], adj_norm) for b in range(batch_size)])
            h = h_out  # (batch, N, emb_dim)

        node_emb = h
        graph_emb = h.mean(dim=1)  # (batch, emb_dim)
        return node_emb, graph_emb


def build_pedigree_adj(pedigree, individuals: list) -> torch.Tensor:
    """Row-normalised adjacency matrix for message passing.

    Edges: parent→child and child→parent (undirected).
    Self-loops included to retain own representation.
    Returns (N, N) float tensor.
    """
    N = len(individuals)
    idx = {ind: i for i, ind in enumerate(individuals)}
    A = torch.eye(N)  # self-loops

    for ind in individuals:
        parents = pedigree.get_parents(ind)
        for p in parents:
            if p in idx:
                A[idx[p], idx[ind]] = 1.0  # parent → child
                A[idx[ind], idx[p]] = 1.0  # child → parent (undirected)

    # Row normalise
    deg = A.sum(dim=1, keepdim=True).clamp(min=1.0)
    return A / deg  # (N, N)


def build_generation_depths(pedigree, individuals: list) -> torch.Tensor:
    """Normalised generation depth per individual (0=founder, max→1)."""
    depths = {}

    def depth(ind):
        if ind in depths:
            return depths[ind]
        parents = pedigree.get_parents(ind)
        if not parents:
            depths[ind] = 0
        else:
            depths[ind] = 1 + max(depth(p) for p in parents)
        return depths[ind]

    for ind in individuals:
        depth(ind)

    max_d = max(depths.values()) or 1
    return torch.tensor(
        [depths[ind] / max_d for ind in individuals], dtype=torch.float32
    )


# ---------------------------------------------------------------------------
# GNN Actor-Critic
# ---------------------------------------------------------------------------

class GNNActorCritic(nn.Module):
    """Actor-Critic using pedigree GNN for state representation.

    Architecture:
      1. Build node features from obs vector (carrier probs + tested flags + gen depth)
      2. Two-layer GraphSAGE → node_emb (N, emb_dim), graph_emb (emb_dim)
      3. Actor: MLP(node_emb[i], graph_emb) → logit for action i (test individual i)
                + a stop-action head on graph_emb
      4. Critic: MLP(graph_emb) → scalar value
    """

    def __init__(
        self,
        n_individuals: int,
        n_genes: int,
        pedigree,
        emb_dim: int = 64,
        gnn_layers: int = 2,
        critic_hidden: int = 128,
    ):
        super().__init__()
        self.N = n_individuals
        self.G = n_genes
        self.n_actions = n_individuals + 1  # N tests + stop

        # Node features: G carrier probs + 1 tested flag + 1 gen depth = G+2
        node_in_dim = n_genes + 2

        self.gnn = PedigreeGNN(node_in_dim, emb_dim=emb_dim, n_layers=gnn_layers)

        # Per-node action head (for test-individual actions)
        self.node_actor = nn.Sequential(
            nn.Linear(emb_dim + emb_dim, emb_dim),
            nn.Tanh(),
            nn.Linear(emb_dim, 1),
        )

        # Stop action head
        self.stop_actor = nn.Sequential(
            nn.Linear(emb_dim, emb_dim // 2),
            nn.Tanh(),
            nn.Linear(emb_dim // 2, 1),
        )

        # Critic
        self.critic = nn.Sequential(
            nn.Linear(emb_dim, critic_hidden),
            nn.Tanh(),
            nn.Linear(critic_hidden, 1),
        )

        _init_weights(self)

    def _build_node_feats(
        self,
        obs: torch.Tensor,
        gen_depths: torch.Tensor,
    ) -> torch.Tensor:
        """Extract per-node feature matrix from flat obs.

        obs layout: [carrier_probs (N*G), tested_flags (N)]
        Returns (batch, N, G+2)
        """
        batch = obs.shape[0]
        N, G = self.N, self.G
        carrier = obs[:, : N * G].reshape(batch, N, G)     # (batch, N, G)
        tested = obs[:, N * G : N * G + N].unsqueeze(-1)   # (batch, N, 1)
        gen_d = gen_depths.unsqueeze(0).unsqueeze(-1).expand(batch, N, 1)  # (batch, N, 1)
        return torch.cat([carrier, tested, gen_d], dim=-1)  # (batch, N, G+2)

    def forward(
        self,
        obs: torch.Tensor,
        adj_norm: torch.Tensor,
        gen_depths: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Returns (logits, value)."""
        batch = obs.shape[0]
        node_feats = self._build_node_feats(obs, gen_depths)           # (batch, N, G+2)
        node_emb, graph_emb = self.gnn(node_feats, adj_norm)           # (batch,N,emb), (batch,emb)

        # Per-individual test logits
        graph_expanded = graph_emb.unsqueeze(1).expand(-1, self.N, -1) # (batch, N, emb)
        node_input = torch.cat([node_emb, graph_expanded], dim=-1)     # (batch, N, 2*emb)
        test_logits = self.node_actor(node_input).squeeze(-1)           # (batch, N)

        # Stop logit
        stop_logit = self.stop_actor(graph_emb)  # (batch, 1)

        logits = torch.cat([test_logits, stop_logit], dim=-1)  # (batch, N+1)
        if mask is not None:
            logits = masked_softmax(logits, mask)

        value = self.critic(graph_emb).squeeze(-1)  # (batch,)
        return logits, value

    def act(
        self,
        obs: torch.Tensor,
        adj_norm: torch.Tensor,
        gen_depths: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        deterministic: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        logits, value = self.forward(obs, adj_norm, gen_depths, mask)
        dist = Categorical(logits=logits)
        action = logits.argmax(-1) if deterministic else dist.sample()
        return action, dist.log_prob(action), value

    def evaluate(
        self,
        obs: torch.Tensor,
        actions: torch.Tensor,
        adj_norm: torch.Tensor,
        gen_depths: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        logits, value = self.forward(obs, adj_norm, gen_depths, mask)
        dist = Categorical(logits=logits)
        return dist.log_prob(actions), value, dist.entropy()
