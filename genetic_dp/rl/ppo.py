"""
PPO (Proximal Policy Optimisation) — pure PyTorch implementation.

Works with both MLPActorCritic and GNNActorCritic via a unified interface.
GNN variant passes adj_norm and gen_depths through kwargs.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from .networks import MLPActorCritic, GNNActorCritic


# ---------------------------------------------------------------------------
# Rollout buffer
# ---------------------------------------------------------------------------

@dataclass
class RolloutBuffer:
    """Stores one batch of rollout experience."""
    obs: List[np.ndarray] = field(default_factory=list)
    actions: List[int] = field(default_factory=list)
    log_probs: List[float] = field(default_factory=list)
    rewards: List[float] = field(default_factory=list)
    values: List[float] = field(default_factory=list)
    dones: List[bool] = field(default_factory=list)
    masks: List[np.ndarray] = field(default_factory=list)

    def clear(self) -> None:
        self.obs.clear(); self.actions.clear(); self.log_probs.clear()
        self.rewards.clear(); self.values.clear(); self.dones.clear()
        self.masks.clear()

    def __len__(self) -> int:
        return len(self.actions)

    def compute_returns_advantages(
        self,
        last_value: float,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """GAE-λ returns and advantages."""
        n = len(self.rewards)
        returns = np.zeros(n, dtype=np.float32)
        advantages = np.zeros(n, dtype=np.float32)
        values = np.array(self.values + [last_value], dtype=np.float32)
        dones = np.array(self.dones, dtype=np.float32)

        gae = 0.0
        for t in reversed(range(n)):
            delta = self.rewards[t] + gamma * values[t + 1] * (1 - dones[t]) - values[t]
            gae = delta + gamma * gae_lambda * (1 - dones[t]) * gae
            advantages[t] = gae
            returns[t] = advantages[t] + values[t]

        return returns, advantages


# ---------------------------------------------------------------------------
# PPO Agent
# ---------------------------------------------------------------------------

class PPOAgent:
    """PPO trainer for sequential genetic testing.

    Parameters
    ----------
    policy       : MLPActorCritic or GNNActorCritic.
    lr           : Learning rate.
    clip_ratio   : PPO clip ε.
    value_coef   : Coefficient for value loss.
    entropy_coef : Entropy bonus coefficient.
    max_grad_norm: Gradient clipping norm.
    n_epochs     : PPO update epochs per rollout.
    batch_size   : Mini-batch size for PPO update.
    gamma, gae_lambda: Return / advantage computation.
    gnn_kwargs   : If using GNNActorCritic, pass adj_norm and gen_depths here.
    """

    def __init__(
        self,
        policy: nn.Module,
        lr: float = 3e-4,
        clip_ratio: float = 0.2,
        value_coef: float = 0.5,
        entropy_coef: float = 0.01,
        max_grad_norm: float = 0.5,
        n_epochs: int = 4,
        batch_size: int = 64,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        gnn_kwargs: Optional[Dict[str, Any]] = None,
        device: str = "cpu",
    ):
        self.policy = policy.to(device)
        self.device = torch.device(device)
        self.clip_ratio = clip_ratio
        self.value_coef = value_coef
        self.entropy_coef = entropy_coef
        self.max_grad_norm = max_grad_norm
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.gnn_kwargs = gnn_kwargs  # {adj_norm: tensor, gen_depths: tensor}

        self.optimizer = optim.Adam(policy.parameters(), lr=lr)
        self._buffer = RolloutBuffer()

        # Training metrics (filled during train())
        self.metrics: Dict[str, List[float]] = {
            "episode_reward": [],
            "policy_loss": [],
            "value_loss": [],
            "entropy": [],
            "approx_kl": [],
        }

    # ------------------------------------------------------------------
    # Rollout collection
    # ------------------------------------------------------------------

    @torch.no_grad()
    def collect_rollout(
        self,
        env,
        n_steps: int = 512,
        n_envs: int = 1,
    ) -> Dict[str, float]:
        """Collect `n_steps` environment steps. Returns episode stats."""
        self.policy.eval()
        self._buffer.clear()
        ep_rewards: List[float] = []
        ep_r = 0.0
        obs = env.reset()

        for _ in range(n_steps):
            obs_t = torch.tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
            mask_t = torch.tensor(env.action_mask(), dtype=torch.bool, device=self.device).unsqueeze(0)

            if isinstance(self.policy, GNNActorCritic) and self.gnn_kwargs:
                action, log_prob, value = self.policy.act(
                    obs_t, mask=mask_t,
                    adj_norm=self.gnn_kwargs["adj_norm"].to(self.device),
                    gen_depths=self.gnn_kwargs["gen_depths"].to(self.device),
                )
            else:
                action, log_prob, value = self.policy.act(obs_t, mask=mask_t)

            a = int(action.item())
            # Capture mask before step — post-step mask blocks the taken action,
            # which would give -inf log_prob during the PPO update.
            pre_step_mask = env.action_mask().copy()
            next_obs, reward, done, _ = env.step(a)
            ep_r += reward

            self._buffer.obs.append(obs.copy())
            self._buffer.actions.append(a)
            self._buffer.log_probs.append(float(log_prob.item()))
            self._buffer.rewards.append(float(reward))
            self._buffer.values.append(float(value.item()))
            self._buffer.dones.append(done)
            self._buffer.masks.append(pre_step_mask)

            obs = next_obs
            if done:
                ep_rewards.append(ep_r)
                ep_r = 0.0
                obs = env.reset()

        # Bootstrap last value
        obs_t = torch.tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
        if isinstance(self.policy, GNNActorCritic) and self.gnn_kwargs:
            _, last_value = self.policy.forward(
                obs_t,
                self.gnn_kwargs["adj_norm"].to(self.device),
                self.gnn_kwargs["gen_depths"].to(self.device),
            )
        else:
            _, last_value = self.policy.forward(obs_t)
        last_value = float(last_value.item())

        returns, advantages = self._buffer.compute_returns_advantages(
            last_value, self.gamma, self.gae_lambda
        )
        self._buffer._returns = returns
        self._buffer._advantages = advantages

        return {
            "mean_ep_reward": float(np.mean(ep_rewards)) if ep_rewards else 0.0,
            "n_episodes": len(ep_rewards),
        }

    # ------------------------------------------------------------------
    # PPO update
    # ------------------------------------------------------------------

    def update(self) -> Dict[str, float]:
        """Run PPO update on the current buffer. Returns loss stats."""
        self.policy.train()
        buf = self._buffer
        n = len(buf)

        obs_arr = np.array(buf.obs, dtype=np.float32)
        act_arr = np.array(buf.actions, dtype=np.int64)
        old_lp_arr = np.array(buf.log_probs, dtype=np.float32)
        ret_arr = buf._returns
        adv_arr = buf._advantages
        mask_arr = np.array(buf.masks, dtype=bool)

        # Normalise advantages
        adv_arr = (adv_arr - adv_arr.mean()) / (adv_arr.std() + 1e-8)

        total_policy_loss = 0.0
        total_value_loss = 0.0
        total_entropy = 0.0
        total_kl = 0.0
        n_updates = 0

        for _ in range(self.n_epochs):
            indices = np.random.permutation(n)
            for start in range(0, n, self.batch_size):
                idx = indices[start: start + self.batch_size]

                obs_b = torch.tensor(obs_arr[idx], device=self.device)
                act_b = torch.tensor(act_arr[idx], device=self.device)
                old_lp_b = torch.tensor(old_lp_arr[idx], device=self.device)
                ret_b = torch.tensor(ret_arr[idx], dtype=torch.float32, device=self.device)
                adv_b = torch.tensor(adv_arr[idx], dtype=torch.float32, device=self.device)
                mask_b = torch.tensor(mask_arr[idx], device=self.device)

                if isinstance(self.policy, GNNActorCritic) and self.gnn_kwargs:
                    new_lp, value, entropy = self.policy.evaluate(
                        obs_b, act_b, mask=mask_b,
                        adj_norm=self.gnn_kwargs["adj_norm"].to(self.device),
                        gen_depths=self.gnn_kwargs["gen_depths"].to(self.device),
                    )
                else:
                    new_lp, value, entropy = self.policy.evaluate(obs_b, act_b, mask_b)

                ratio = torch.exp(new_lp - old_lp_b)
                surr1 = ratio * adv_b
                surr2 = torch.clamp(ratio, 1 - self.clip_ratio, 1 + self.clip_ratio) * adv_b
                policy_loss = -torch.min(surr1, surr2).mean()
                value_loss = F.mse_loss(value, ret_b)
                entropy_loss = -entropy.mean()

                loss = policy_loss + self.value_coef * value_loss + self.entropy_coef * entropy_loss

                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
                self.optimizer.step()

                with torch.no_grad():
                    kl_vals = old_lp_b - new_lp
                    finite = kl_vals[torch.isfinite(kl_vals)]
                    kl = finite.mean().item() if len(finite) > 0 else 0.0

                total_policy_loss += policy_loss.item()
                total_value_loss += value_loss.item()
                total_entropy += (-entropy_loss).item()
                total_kl += kl
                n_updates += 1

        stats = {
            "policy_loss": total_policy_loss / max(n_updates, 1),
            "value_loss": total_value_loss / max(n_updates, 1),
            "entropy": total_entropy / max(n_updates, 1),
            "approx_kl": total_kl / max(n_updates, 1),
        }
        return stats

    # ------------------------------------------------------------------
    # Top-level train loop
    # ------------------------------------------------------------------

    def train(
        self,
        env,
        total_timesteps: int = 50_000,
        rollout_steps: int = 512,
        log_every: int = 10,
        eval_env=None,
        eval_every: int = 20,
    ) -> Dict[str, List[float]]:
        """Train for `total_timesteps` environment steps.

        Returns log dict with per-iteration metrics.
        """
        log: Dict[str, List] = {
            "iteration": [], "timesteps": [], "mean_ep_reward": [],
            "policy_loss": [], "value_loss": [], "entropy": [], "approx_kl": [],
        }

        timesteps = 0
        iteration = 0
        t_start = time.time()

        while timesteps < total_timesteps:
            rollout_stats = self.collect_rollout(env, n_steps=rollout_steps)
            update_stats = self.update()
            timesteps += rollout_steps
            iteration += 1

            if iteration % log_every == 0:
                elapsed = time.time() - t_start
                fps = timesteps / elapsed
                print(
                    f"[{timesteps:>7d}/{total_timesteps}] "
                    f"ep_reward={rollout_stats['mean_ep_reward']:>8.4f}  "
                    f"policy_loss={update_stats['policy_loss']:>7.4f}  "
                    f"value_loss={update_stats['value_loss']:>7.4f}  "
                    f"entropy={update_stats['entropy']:>6.4f}  "
                    f"fps={fps:>5.0f}"
                )
                log["iteration"].append(iteration)
                log["timesteps"].append(timesteps)
                log["mean_ep_reward"].append(rollout_stats["mean_ep_reward"])
                log["policy_loss"].append(update_stats["policy_loss"])
                log["value_loss"].append(update_stats["value_loss"])
                log["entropy"].append(update_stats["entropy"])
                log["approx_kl"].append(update_stats["approx_kl"])

        return log

    # ------------------------------------------------------------------
    # Evaluation
    # ------------------------------------------------------------------

    @torch.no_grad()
    def evaluate_policy(
        self,
        env,
        n_episodes: int = 100,
        deterministic: bool = True,
    ) -> Dict[str, float]:
        """Monte-Carlo evaluation. Returns mean/std episode reward."""
        self.policy.eval()
        rewards: List[float] = []

        for _ in range(n_episodes):
            obs = env.reset()
            ep_r = 0.0
            while not env.done:
                obs_t = torch.tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
                mask_t = torch.tensor(env.action_mask(), dtype=torch.bool, device=self.device).unsqueeze(0)

                if isinstance(self.policy, GNNActorCritic) and self.gnn_kwargs:
                    action, _, _ = self.policy.act(
                        obs_t, mask=mask_t,
                        adj_norm=self.gnn_kwargs["adj_norm"].to(self.device),
                        gen_depths=self.gnn_kwargs["gen_depths"].to(self.device),
                        deterministic=deterministic,
                    )
                else:
                    action, _, _ = self.policy.act(obs_t, mask=mask_t, deterministic=deterministic)

                obs, reward, done, _ = env.step(int(action.item()))
                ep_r += reward
            rewards.append(ep_r)

        return {
            "mean_reward": float(np.mean(rewards)),
            "std_reward": float(np.std(rewards)),
            "min_reward": float(np.min(rewards)),
            "max_reward": float(np.max(rewards)),
        }
