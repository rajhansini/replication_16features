# Speaker Notes — RL Genetic Testing Presentation

Open slides: `presentation/index.html` in Chrome/Firefox (needs network once for Reveal.js CDN).

**2-minute pitch (memorize this):**
> Exact dynamic programming gives optimal genetic testing policies but explodes past ~9 family members — 62 seconds at N=9, infeasible beyond that. We built a Gym environment with validated Bayesian belief updates, trained PPO, and showed it tracks exact optimality within 1% on solvable instances. RL inference stays at 0.12 milliseconds per step while exact solve time grows exponentially. We trained policies for families up to N=15 and multi-gene pedigrees where exact methods cannot run, beating random and myopic baselines.

---

## Slide 1 — Title (~15 sec)
- Set up: sequential genetic testing MDP, RL as scalable alternative to exact DP.

## Slide 2 — The Bottleneck (~45 sec)
- Decision: test next untested individual or stop.
- Belief state = carrier probabilities; updates follow Mendelian rules.
- Exact backward induction optimal but state space is 3^N genotypes.
- **Key number:** 62 seconds to solve N=9 exactly; N=12+ infeasible.
- Point at fig1: exponential growth in exact solve time.

## Slide 3 — Our Approach (~45 sec)
- Walk through the pipeline left to right.
- Emphasize validation: env beliefs match hand-computed posteriors.
- RL inference ~0.12 ms/step — doesn't grow with N at deployment time.

## Slide 4 — Week 3 Benchmark (~60 sec)
- Table: gap shrinks as N grows — **0.5% at N=9**.
- fig3: the killer slide — exact time shoots up, RL inference flat.
- "500× faster inference than exact solve at N=9."

## Slide 5 — Convergence (~30 sec)
- fig2: gap curve — small families harder (less structure to learn from).
- fig4: PPO training curves stabilize within ~50K steps.

## Slide 6 — Scaling Past Exact (~60 sec)
- **Main result for scalability story.**
- N=12: 16.8M state bound, RL reward −0.239 — no ground truth but policy exists.
- N=15: 1.07 billion states — exact DP cannot run, RL trains in ~94 min.
- fig5 left panel: RL reward tracks exact where known, continues beyond.

## Slide 7 — Multi-Gene (~30 sec)
- Observation dimension grows with genes; RL doesn't degrade G=1→3.
- Sets up future work on 5–20 genes.

## Slide 8 — Baselines (~45 sec)
- RL crushes random (60–80% gap reduction).
- Beats myopic at N=4,6,7; at N=9 myopic is slightly better (−0.167 vs −0.179) but both near exact.
- Myopic over-tests at N=7 (6.1 vs 5.2 tests).

## Slide 9 — Sensitivity (~30 sec)
- Robust across allele freqs 0.05–0.30.
- Best match at freq=0.15 (0.5% gap).

## Slide 10 — Policy Viz (~45 sec)
- fig10: RL tests parents/high-info individuals first.
- Stops earlier on average than myopic (6.05 vs 6.70 tests).
- "The policy learned something structurally sensible."

## Slide 11 — Takeaways (~30 sec)
- Read the three boxes. Pause after each.

## Slide 12 — Future Work (~20 sec)
- GNN encoder, more genes, link to ABCD-16 ADP work in same repo.
- Skip details unless asked.

## Slide 13 — Thank You
- Open for questions. Backup: `artifacts/` has all raw JSON + figures.

---

## Anticipated questions

**Q: Why is gap worse at small N?**
A: Less training signal / shorter episodes; also fewer timesteps relative to state diversity. At N=9 the problem structure is richer and PPO has more to learn from.

**Q: How do you know N=15 policy is good without exact V*?**
A: We don't have a certificate — that's the point. We compare to myopic/random on those sizes and show RL trains stably. Future: ADP upper bounds from the Gurobi pipeline.

**Q: Why PPO not DQN?**
A: Continuous-ish belief observations, variable episode length, action space scales with N — PPO handles this naturally with the Gym interface.

**Q: GNN?**
A: Planned — MLP in `genetic_dp/rl/networks.py` is swappable; Kanix said GNN is secondary to proving scalability first.
