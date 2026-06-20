# The ABCD 16-Feature Model — Definitions and How It Runs

Reference for the ABCD-16 model reproduced by this package: the ADP value
function with the fixed `ABCD16_DIRECT` feature bank (all 16 features, used on
every family). Content here is condensed from the project's own design docs,
not invented.

## How the model is run

The ADP value function adds a linear feature correction to a base value:

```
Phi(s) = base(s) + sum_{j in A/D} beta_j * f_j(s) + sum_{k in B/C} gamma_k * g_k(s)
```

For one benchmark family, the run does, in order:

1. Run the recursive **myopic** base-policy diagnostics — compute the myopic
   action `a_M(s)`, the myopic stop indicator `z_stop(s)`, and the myopic
   residual / test-vs-stop margin `rho(s)`.
2. Compute the posterior / topology / state summaries each feature needs.
3. Evaluate all **16** ABCD feature values `f(s)`, `g(s)` on every materialized
   belief state.
4. Solve **one** ADP/ALP for the base variables and all 16 feature coefficients
   jointly — with the full fixed 16-feature bank on every state.

The myopic diagnostics are required because several features are functions of
`a_M`, `z_stop`, and `rho`.

**Feature normalization.** A raw feature `h_j(s)` does not enter the LP directly.
It is first root-centered, `h_j°(s) = h_j(s) − h_j(s0)`, then divided by a frozen
scale `scale_j = max_u |h_j(u) − h_j(s0)|` over the state pool (features with
`scale_j < 1e-12` are skipped). Thus the LP feature `~h_j(s0) = 0`: a feature
cannot improve the root certificate by a constant offset — it must act through
successor-state effects inside the Bellman constraints.

## Notation

For belief state `s`: `U(s)` = untested people; `F(s)` = the active unresolved
**frontier** (untested people still Bellman-relevant — affecting stop reward,
test value, or posterior propagation). For person `i`:

- `c_i(s)` — any-positive **carrier probability**, `1 − prod_k (1 − c_{i,k}(s))`.
- `u_i(s)` — posterior **uncertainty mass** (e.g. variance `c(1−c)` or entropy).
- `b_i` — topology-only **bridge / articulation** score (how much `i` connects
  pedigree branches); `d_i` — structural **depth** score; `D_i(s)` — downstream
  descendant weight.

Every feature is computed **only** from `s`, the pedigree topology, posterior
marginals, and the parameter setting `η` (allele frequencies, reward
coefficients, fixed/variable costs). No feature uses the exact optimum `V*`, its
policy `π*`, exact Bellman slack, exact policy labels, or myopic value *levels*.

## A/D block — 13 myopic / structural features

| feature | definition | meaning |
|---|---|---|
| `frontier_carrier_mass` | `sum_{i in F} c_i(s)` | total posterior carrier probability over the active frontier |
| `frontier_carrier_max` | `max_{i in F} c_i(s)` | largest single-person carrier probability on the frontier |
| `frontier_carrier_variance` | `Var_{i in F} c_i(s)` | concentration vs diffusion of frontier carrier risk |
| `bridge_depth_mass` | `sum_{i in F} u_i(s) b_i d_i` | unresolved uncertainty on bridge-like people, weighted by depth |
| `descendant_bridge_mass` | `sum_{i in F} u_i(s) b_i D_i(s)` | bridge uncertainty connected to downstream descendants |
| `sibling_breadth` | `sum_B 1{|B_U|>=2} sum_{i in B_U} u_i(s)` | unresolved width across sibling blocks (≥2 untested sibs) |
| `collateral_block_count` | `sum_C 1{C ∩ U(s) != ∅}` | number of active collateral / side-branch blocks (uncle/aunt/cousin motifs) |
| `myopic_tests` | `1{a_M(s) = test}` | the myopic base policy tests (vs stops) at `s` |
| `stop_test_margin` | `rho(s)` | myopic test-vs-stop value gap (myopic Bellman residual) at `s` |
| `boundary_state_indicator` | `1{0 < rho(s) <= 1e-3}` | state sits on the myopic stop/test decision boundary |
| `best_second_best_test_margin` | `rho(s) * (1 − z_stop(s))` | the margin restricted to non-stopping states |
| `myopic_stop_gate_pressure` | `z_stop(s) * (1 + rho(s))` | stop-gated pressure — active where the myopic policy stops |
| `cost_adjusted_continuation_margin` | `rho(s) / (1 + stage)` | stop/test margin discounted by stage depth |

## B/C block — 3 regime features

The `_honest` suffix is part of each feature's name. Like every feature here, these
are computed only from the belief state, pedigree topology, and parameters.

| feature | meaning |
|---|---|
| `all_untested_carrier_variance_honest` | carrier-probability variance over **all** untested people `U(s)` (not just the frontier) — diffuse vs concentrated risk across the whole untested set |
| `allele_asymmetry_high_gene_GeneA_carrier_depth_mass_honest` | allele-asymmetry-gated, depth-weighted carrier mass on the high-frequency gene (GeneA); active when allele frequencies are asymmetric |
| `collateral_active_parent_pair_block_count_honest` | count of active collateral **parent-pair** blocks (side-branch parent pairs with untested members) |

## Why "ABCD"

The 16 split into an **A/D** myopic/structural block (13) and a **B/C**
regime block (3). The canonical accepted A/B/C/D block is 14 features;
the two extras observed in the universe are `sibling_breadth` and
`frontier_carrier_variance`, giving the fixed **16**. The ABCD-16 model uses the
entire fixed 16-feature universe on every family.

## Provenance

- Frozen bank (the 16 feature names, `selection_mode`, tolerance):
  `documentation/abcd16_direct_feature_bank_20260528.json` (bundled in this package).
- Feature computation: `genetic_dp/optimisation/myopic_adp.py::build_state_features`.
- Formulas, normalization, and the input contract are condensed from the project's
  feature-definition design docs.
