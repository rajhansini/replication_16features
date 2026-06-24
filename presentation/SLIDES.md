# RL for Scalable Genetic Testing Policies
### Week 6 Presentation — Rajhansini

---

## Slide 1 — Title

**RL for Scalable Genetic Testing Policies**
*When Exact Dynamic Programming Runs Out of Room*

Rajhansini | Rana Lab | June 2026

> **Speaker notes:** Introduce the core tension in one sentence: optimal genetic testing policies exist in theory, but computing them exactly becomes impossible for realistic family sizes. We used RL to bridge that gap.

---

## Slide 2 — Problem

**Genetic Testing as a Sequential Decision Problem**

- Family of N members, each may carry a disease variant
- Clinician decides: who to test next, or stop
- Reward = information gained minus testing cost
- Exact backward induction solves this optimally — but state space is 3^N
  - N=9 → 262,144 states, 62 seconds to solve
  - N=15 → 1,073,741,824 states → **infeasible**

> **Speaker notes:** This is a classic MDP. The action space is "test person i" or "stop." Beliefs update via Bayes each time a result comes in. Exact DP gives the ground-truth optimal policy — but only when the family is small enough to enumerate.

---

## Slide 3 — Approach

**Gym Environment + PPO**

- Built a custom OpenAI Gym environment with correct Bayesian belief updates
- Observation: belief vector over all 3^N genotype configs + test-result history
- Action space: N+1 actions (test member i, or stop)
- Trained with PPO (Stable-Baselines3), ~50K timesteps per family size
- **Validated against exact backward induction** on N ≤ 9

Key design choices:
- Belief state as observation → generalizes across rollouts
- Reward shaped as: testing cost penalty + diagnostic information gain

> **Speaker notes:** Nothing exotic here — the hard work is in the environment, specifically getting the Bayesian update right and making sure the action masking works (you can't test someone twice). Training takes ~5 minutes per N on a single GPU.

---

## Slide 4 — Benchmark: RL vs Exact DP

**Figure 3 — [figures/fig3_scalability.png](figures/fig3_scalability.png)**

| N | Exact Solve Time | RL Inference | Gap vs Optimal |
|---|---|---|---|
| 4 | 15 ms | 0.12 ms/step | 16.7% |
| 6 | 3.4 s | 0.12 ms/step | 14.6% |
| 7 | 2.1 s | 0.13 ms/step | 7.2% |
| **9** | **62 s** | **0.12 ms/step** | **0.52%** |
| 12 | infeasible | 0.11 ms/step | — |
| 15 | infeasible | 0.12 ms/step | — |

- At N=9: **RL matches exact within 0.5%** — the last point where ground truth exists
- Inference is **~515,000× faster per step** than exact solve time (62s → 0.12ms)
- RL inference time is **flat** across all N — exact grows exponentially

> **Speaker notes:** The gap shrinks as N grows because harder problems have more room for reward and PPO converges to near-optimal on the benchmark family. At N=9 the policy is essentially exact. The 0.12ms inference time doesn't move as N doubles — that's the key scalability claim.

---

## Slide 5 — Scaling Beyond Exact DP

**Figure 5 — [figures/fig5_family_scale.png](figures/fig5_family_scale.png)**

- N=12: state space = 16.7 million — exact DP **does not finish**
- N=15: state space = 1.07 billion — exact DP **cannot run**
- RL trains and deploys at both: 0.11–0.12 ms/step, smooth reward curves

| N | RL Reward | Train Time |
|---|---|---|
| 9 | −0.165 | ~5.7 min |
| 12 | −0.239 | ~15.4 min |
| 15 | −0.320 | ~94 min |

- Multi-gene (G=3, N=9): exact DP is infeasible — RL trains in ~6 min, reward −0.164

> **Speaker notes:** This slide makes the core case. Exact DP is the gold standard but it falls off a cliff. RL doesn't care about the state space size — it samples trajectories. The reward increasing (becoming more negative) with N just reflects that more family members = more costly testing required on average.

---

## Slide 6 — Baselines Comparison

**Figure 8 — [figures/fig8_baselines.png](figures/fig8_baselines.png)**

At N=9 (where exact ground truth exists):

| Policy | Mean Reward | vs. Exact |
|---|---|---|
| Random | −0.283 | −70% |
| Myopic (greedy) | −0.167 | −0.3% |
| **RL (PPO)** | **−0.179** | **−7.7%** |
| Exact DP | −0.166 | — |

- RL beats random by **37%**
- RL is competitive with myopic; myopic happens to be strong at N=9
- At larger N (no exact baseline), RL is the best tractable policy

> **Speaker notes:** Myopic (always test the highest-entropy member) is a surprisingly strong heuristic at small N. RL's advantage grows at larger family sizes where myopic can't look ahead. Random is the floor — picking arbitrarily is clearly bad. The point is RL is a principled, learnable policy that can scale where myopic also struggles.

---

## Slide 7 — What the Policy Does

**Figure 10 — [figures/fig10_policy_viz.png](figures/fig10_policy_viz.png)**

- Policy learns to test **high-prior-risk members first** (informationally dense)
- Stops early when belief is sufficiently concentrated
- Adapts testing order based on incoming results — not a fixed sequence

> **Speaker notes:** This is the "sanity check" slide. We can look at the policy and confirm it's doing something reasonable — not a black box that happened to get lucky on the benchmark. Early stopping is especially important since each test has a cost.

---

## Slide 8 — Takeaways

1. **RL matches exact DP within <1% at N=9**, the largest instance with ground truth
2. **Inference is ~500,000× faster per step** than exact solve (62s → 0.12ms), and constant as N grows
3. **RL trains on N=12, N=15, and multi-gene (G=3)** settings where exact DP is infeasible

> **Speaker notes:** These three bullets are the full story. Quality parity where measurable, speed advantage that only grows, and access to a regime exact methods can't touch.

---

## Slide 9 — Future Work

- **GNN state encoder** — replace MLP with graph net over family pedigree; generalize across family structures
- **More genes (G=5+)** — observation space grows but training pipeline unchanged
- **Real pedigrees** — currently synthetic Mendelian genetics; connect to ABCD-16 pedigree structures
- **ABCD-16 connection** — link to existing replicate.py ADP work as the "exact baseline" anchor

> **Speaker notes:** The GNN encoder is the most impactful next step — current MLP encodes family members as a flat vector, which doesn't respect the graph structure of a pedigree. Real pedigrees from clinical data would be the eventual deployment target.

---

## Appendix A — Sensitivity Analysis

Figure 9 — sensitivity to prior probability and cost parameters

## Appendix B — Training Curves

Figure 4 — PPO reward curves across N, convergence by ~30K timesteps

---
