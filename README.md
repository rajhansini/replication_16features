# ABCD16 — Standalone Replication Package

This directory reproduces the **ABCD-16** results from
`abcd16_stage_chain_ratio2_ratio3_report_20260601.md` and audits the reproduced
`ratio2`/`ratio3` against the published numbers.

The ABCD-16 model is the ADP value function that fixes all **16** ABCD features
(13 myopic/structural + 3 regime) and uses them on every family. Feature bank:
`ABCD16_DIRECT`, `selection_mode = fixed_all_features`.

It is **self-contained**: it reads and writes nothing outside this folder. Copy
the folder anywhere and it still runs.

## How to run

Requirements: Python 3.11+ with `gurobipy` (Gurobi 12), `numpy`, and `tqdm`
(`pip install -r requirements.txt`). Gurobi must be licensed — it is the
exact-DP/ADP dual LP solver; a free academic license works. Run from inside this
directory:

```bash
python replicate.py
```

- Runs the 4 sampled rows (~5 min each; ~20 min total on this machine).
- Re-solves each family from scratch — exact DP **and** the fixed-16 ADP —
  and compares the reproduced `ratio2`/`ratio3` to the embedded expected values
  to **1e-6 absolute**, printing the per-row signed diff.
- Writes `output/replication_results.json` (verdict + per-row numbers). Exit code
  is non-zero if any row misses tolerance.
- **From-scratch guarantee:** the exact-DP cache is memory-only
  (`EXACT_DP_CACHE_IN_MEMORY_ONLY=1`), so no result is ever persisted to disk or
  replayed across runs — every invocation recomputes from scratch and logs
  `Exact DP cache miss`. Any incidental scratch stays under `output/.cache/`.

Run a single row with `--only`, e.g.
`python replicate.py --only ThreeGeneration_LowHigh_Aggressive`.

### Reproducing all 102 families — step by step

The 4-row default is a fast audit; the package also contains the **complete**
102-family universe (original8 + local40 + phase6_54) and can reproduce every
one of the report's 102 rows.

1. **Install dependencies** (one time):
   ```bash
   pip install -r requirements.txt        # gurobipy, numpy, tqdm
   ```
2. **License Gurobi.** Gurobi is the LP solver. A free
   [academic license](https://www.gurobi.com/academia/academic-program-and-licenses/)
   works; run `grbgetkey <key>` once to activate it. Verify with
   `python -c "import gurobipy; print(gurobipy.Model())"`.
3. **Confirm the run is set up** (instant, no solving) — loads all 102 families
   and checks each builds from the bundled manifests and has an embedded expected
   value:
   ```bash
   python replicate.py --full102 --plan
   ```
4. **Run the full replication** (re-solves all 102 from scratch — **hours** of
   compute; each family ~1–11 min):
   ```bash
   python replicate.py --full102
   ```

`--full102` audits each reproduced `ratio2`/`ratio3` against `full102_expected.json`
(the 102 expected values, copied verbatim from the report) to 1e-6 and writes the
results to `output/full102_results.json` (see *Output* below). The 102 settings
come from three bundled manifests under `documentation/` and `artifacts/`.

## Output — where results go

Everything is written under `output/` inside this directory (nothing leaves the
folder):

- **Console** — a live per-family line as each solves: `[PASS]/[FAIL]`, the
  expected vs reproduced `ratio2`/`ratio3`, the signed diff, and solve time; then
  a final `VERDICT: PASS/FAIL`.
- **`output/full102_results.json`** (or `output/replication_results.json` for the
  4-row sample) — the machine-readable audit: overall `verdict`, the `tolerance_abs`,
  and one record per family with `expected`, `reproduced`, `diff` (signed, per
  ratio), `intermediate` (`V_star`, `V_stop`, `U_adp_phi`, `L_policy`), `pass`, and
  `seconds`.
- The process exit code is non-zero if any family misses tolerance.

During a solve the engine writes large scratch (per-family belief snapshots,
a debug LP dump) under `output/.cache/` and the folder root, but the run
**clears it after every family**, so the package never accumulates large files —
a full 102-family run holds at most one family's scratch at a time. All scratch
is git-ignored. The committed code + data are all under ~0.3 MB.

### What the run does, step by step

For each sampled row, `replicate.py`:

1. builds the exact `Setting` (family topology + reward preset + allele
   frequencies + costs) from the copied manifests via the original
   `load_cases(suite)` loader — identical inputs to the report;
2. applies `direct16_env()` — the locked configuration that pins the ABCD-16
   model (the `fixed_all_16` feature bank);
3. calls the shared solver (`_run_setting` → `run_and_compare_solvers`), which
   computes the exact optimum `V*`, the stop-only value `V_stop`, the fixed-16
   ADP upper bound `U`, and the extracted-policy value `L`;
4. forms `ratio2`/`ratio3` and checks them against the report.

## The model: 16 ABCD features

Feature bank `ABCD16_DIRECT` (`documentation/abcd16_direct_feature_bank_20260528.json`):
`selection_mode = fixed_all_features`, `certificate_tolerance = 5e-7`. All 16
features are used on every family (the bank is fixed). They are computed per
belief-state in
`genetic_dp/optimisation/myopic_adp.py` (`build_state_features`). The 16 split
into an **A/D** myopic/structural block (13) and a **B/C** regime-residual block
(3) — this is the "ABCD" in the name.

> **See [FEATURES.md](FEATURES.md)** for the full per-feature formulas, the
> feature normalization, the input contract, and the step-by-step
> run algorithm (myopic diagnostics → one ADP/ALP solve with
> all 16 features) — condensed from the project design docs and cited. The two
> tables below are the quick summary.

**A/D block — 13 myopic / structural features** — summaries of the carrier-belief
over untested individuals and the pedigree "bridge" structure, plus the myopic
test-vs-stop margin:

| feature | meaning |
|---|---|
| `frontier_carrier_mass` | total posterior carrier probability over untested individuals |
| `frontier_carrier_max` | largest single-individual carrier probability among untested |
| `frontier_carrier_variance` | variance of carrier probabilities across untested individuals |
| `bridge_depth_mass` | carrier mass weighted by pedigree bridge depth |
| `descendant_bridge_mass` | carrier mass carried on descendant-bridge individuals |
| `sibling_breadth` | breadth of the sibling structure at the frontier |
| `collateral_block_count` | number of active collateral (non-lineal) blocks |
| `myopic_tests` | indicator that the myopic action is "test" |
| `stop_test_margin` | myopic Bellman residual (myopic test-vs-stop value gap) |
| `boundary_state_indicator` | indicator that the stop/test margin is near-zero (0–1e-3) |
| `best_second_best_test_margin` | stop/test margin restricted to non-stopping states |
| `myopic_stop_gate_pressure` | stop-gated pressure term `1{stop}·(1+margin)` |
| `cost_adjusted_continuation_margin` | stop/test margin scaled by stage depth |

**B/C block — 3 regime ("honest") features** — residual basis features evaluated
over all untested individuals:

| feature | meaning |
|---|---|
| `all_untested_carrier_variance_honest` | carrier-probability variance over all untested individuals |
| `allele_asymmetry_high_gene_GeneA_carrier_depth_mass_honest` | allele-asymmetry-gated carrier depth mass on the high-frequency gene (GeneA) |
| `collateral_active_parent_pair_block_count_honest` | count of active collateral parent-pair blocks |

### Metrics (lower is better for both)

With `V*` = exact optimum, `V_stop` = stop-only value, `U` = ADP upper bound,
`L` = extracted-policy value, and `denom = V* − V_stop`:

- **`ratio3`** = `(U − V*) / denom` — ADP **certificate** (upper-bound) quality.
- **`ratio2`** = `(V* − L) / denom` — deployed-**policy** (lower-bound) quality.

## Sampled families and expected values

Drawn from the report (both pedigree topologies, both
presets, both allele regimes, and both tails — best-behaved and worst-case):

| suite | row_id | expected `ratio2` | expected `ratio3` |
|---|---|---:|---:|
| original8 | `Extended_LowHigh_Base` | 0.10068959165 | 0.26795140506 |
| original8 | `ThreeGeneration_LowHigh_Aggressive` | 0.05132042772 | 0.10682960329 |
| original8 | `Extended_MediumEven_Base` | 0.15896943584 | 0.18823080255 |
| phase6_54 | `Extended_LowHigh_Base_fixed_0p015_variable_0p030` | 0.42325420934 | 0.39321674745 |

Expected values are embedded in `expected.json`, copied verbatim from the published report.

**Verified (2026-06-19).**
- All 4 sample rows re-solved from scratch and reproduced both `ratio2` and
  `ratio3` with diff `+0.00e+00` (exact to every digit, far inside the 1e-6 gate)
  — verdict **PASS**. Per-row solve times: 44–673 s.
- Run from a `/tmp` copy with `PYTHONPATH` cleared, fully detached from this repo,
  with identical results — confirming it touches no outside files.
- `--full102 --plan` confirms all **102** rows (8 + 40 + 54) build from the
  bundled manifests and have embedded expected values.
- A local40 row (`local_anchor_ThreeGeneration_LowHigh_Aggressive`) re-solved via
  the local40 manifest also reproduced at diff `+0.00e+00`, confirming the
  full-102 path is correct.

## Layout

```
replication_16features/
├── replicate.py          # entry point: re-solve rows + audit vs report (--full102, --plan, --only)
├── expected.json         # embedded ground-truth ratio2/ratio3 — 4-row sample
├── full102_expected.json # embedded ground-truth ratio2/ratio3 — all 102 rows
├── README.md
├── FEATURES.md           # the 16 ABCD features + the run algorithm (cited)
├── genetic_dp/           # ADP solver package
├── scripts/              # family loader + the ADP solver/model modules
├── documentation/        # ABCD16 feature bank, original8 + local40 settings manifests
├── artifacts/.../        # phase6_54 row manifest (extended_54row_manifest.csv)
└── output/               # results + solver scratch (created on run; gitignored)
```

## Provenance

- Source report: `documentation/abcd16_stage_chain_ratio2_ratio3_report_20260601.md`
- Expected `ratio2`/`ratio3` values: the published ABCD16 report (embedded in `expected.json` / `full102_expected.json`).
