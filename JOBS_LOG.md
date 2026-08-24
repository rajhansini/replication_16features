# SLURM job log

Last updated: 2026-08-24 (updated every time a new job is submitted or a
tracked job's status changes materially). Only jobs from this investigation
are listed -- `squeue -u $(whoami)` also shows unrelated jobs (nlp_rq*,
rarm7_*, tex_*, etc.) from other projects, not touched or tracked here.

To check status yourself: `squeue -u $(whoami) | grep <jobname>` or
`sacct -j <jobid> --format=JobID,State,Elapsed -n`

## Currently running / pending

Full re-derivation of every number in the E0-E9 PI briefing artifact, so that
no figure on that page predates the 3-gene allele-frequency fix (2026-08-15/16)
or the E0-E9 retrain (2026-08-22).

| Job ID | Name | Purpose | Status |
|---|---|---|---|
| 2206182 | fourway_rerun | Four-way action logs (DP vs myopic vs GNN-Q vs MLP-Q; root action + whole trajectory) for all 8 rungs x {3-gene, 2-gene} x 3 seeds, 48 tasks | **COMPLETE 08-24, 48/48, 0 errors.** Was RUNNING as of 08-24 03:00. Sat in `JobHeldUser` from submission (08-23 17:59) until released on 08-24 -- it had never started, so until it lands every root-agreement / whole-trajectory figure is still the stale 07-25 -> 08-10 data. 24/48 done (all 2-gene, ~55s each); the 24 3-gene tasks are the slow half but historical runs top out at ~28 min, comfortably inside the 4h wall, so this array needs no resume logic. |

### 2026-08-24: the queue was holding, not running

`2206182` never ran because it was `JobHeldUser` with `Priority=0` -- along with
22 other jobs of this user's (`mnca_*`, `tex_*`, `canary_mem12`). All 23 were
released on 08-24; `JobHeldUser` is now zero. The 3 `nlp_rq7_*` jobs are
`JobHeldAdmin` and cannot be released from a user account.

**Check for this first when a submitted job shows no progress.** `squeue` shows
it as PENDING, which reads like "waiting for nodes" -- the distinguishing signal
is `Reason=JobHeldUser` and `Priority=0` in `scontrol show job <id>`.


### Verified: the 3-gene dataset cache IS the corrected data

Checked before submitting, because regenerating logs against stale inputs would
just produce new wrong numbers. `ground-up-experiments/step9_gnn_3gene/results/cache`
holds 36 pickles, and the mtimes line up exactly with the bug:

| Regime | Files | Dated |
|---|---|---|
| HighHigh, MixedA, MixedB | 6 each | **08-15** (regenerated -- these are the three regimes whose GeneA/GeneB frequencies were wrong) |
| LowHigh, MediumEven, LowLow | 6 each | 06-30 / 07-01 (untouched -- these were always correct) |

So the fourway rerun reads corrected 3-gene data. `log_*_fourway.py` also calls
`genetic_dp.policy.baselines.myopic_greedy` directly, so its myopic rows are
canonical by construction.

### Resume logic (added 2026-08-23)

Every partition on this cluster (`general`, `peanut-cpu`, `threedle-*`) has a
hard **4h MaxTime** -- there is no longer limit to ask for. The leave-one-out
GNN folds do not fit in 4h, so they can only finish by checkpointing and
requeueing across several 4h slots. `submit_e6_split_experiment.sh` +
`incremental_experiments/e6_split_experiment.py` do that, and
`submit_overfit_probes.sh` + `incremental_experiments/overfit_probes_2gene.py`
use the same pattern:

- `#SBATCH --signal=B:USR1@300` fires 5 min before the wall clock; the trap
  calls `scontrol requeue` so the task restarts instead of dying.
- Training resumes from `checkpoint.pt` (written every epoch, already existed).
- TRAIN-side eval resumes from `eval_train_partial.json` (flushed per config).
- A finished `(exp,kind,seed)` exits in <1s via a `results_e6.json` skip guard,
  so **the whole 0-29 array can be resubmitted at any time and only does the
  work that is actually missing**. `--force` redoes one from scratch.

### 2206182 result: every 3-gene action-agreement figure moved (2026-08-24)

Standard families (TRAIN = Trio + Nuclear, TEST = ThreeGeneration), all 12
configs (6 regimes x 2 presets), averaged over seeds 0/1/2. Old = the committed
07-25 -> 08-10 files, New = regenerated against corrected data + retrained
checkpoints.

**2-gene: bit-identical at all 8 rungs**, root and whole-trajectory alike. That
is now the *fourth* independent confirmation that 2-gene was untouched by the
3-gene allele-frequency fix (the others: myopic ratio2, the 2-gene Extended
rerun, and the E0-E9 retrain).

**3-gene: changed at every rung.** Whole-trajectory agreement vs DP:

| Rung | GNN old -> new | MLP old -> new |
|---|---|---|
| E0 | 0.486 -> **0.314** | 0.443 -> 0.517 |
| E1 | 0.462 -> 0.527 | 0.557 -> 0.502 |
| E2 | 0.681 -> **0.729** | 0.533 -> 0.725 |
| E4 | 0.633 -> 0.705 | 0.533 -> 0.725 |
| E5 | 0.590 -> 0.623 | 0.533 -> 0.725 |
| E6 | 0.567 -> 0.628 | 0.710 -> **0.565** |
| E7 | 0.581 -> 0.623 | 0.533 -> 0.725 |
| E9 | 0.395 -> 0.522 | 0.533 -> 0.725 |

Myopic moved 0.557 -> 0.565, consistent with its ratio2 going 0.1052 -> 0.1090.
E0's GNN is the big loser (0.486 -> 0.314); E6's MLP drops hard (0.710 -> 0.565,
and root 2.0/12 -> 0.3/12). E2/E4/E5/E7/E9 share one MLP number because they all
reuse the E2 CE MLP checkpoint -- expected, and true in the old data too.

**Every root-agreement and whole-trajectory figure in the 3-gene half of the PI
briefing is therefore stale and needs replacing from these files.**

### 3-gene Extended does not fit in one slot (2026-08-24)

All nine 3-gene Extended tasks in jobs 2206213 / 2206261 hit the 4h wall and
produced **nothing**. Two compounding causes:

1. Each of the 12 configs solves exact DP over **9,190,992 states from
   scratch** -- there is no cached pickle for Extended at 3 genes, only for
   Trio / Nuclear / ThreeGeneration. The logs show each task finished only
   4-6 of 12 configs before being killed.
2. `log_extended_fourway.py` wrote its JSON **only after all 12 configs
   finished**. So a task that solved 5 configs and then hit the wall threw all
   five away. That is the part that made the timeout total rather than partial.

Fixed by giving the whole-seed path the same resume machinery the split
experiments already use:

- Every config is flushed to `extended/{genes}gene/seed{N}/partial{suffix}/
  {regime}_{preset}.json` the instant it is solved, and a resumed task skips
  what it already has. The log is appended to, not truncated.
- `submit_extended_fourway.sh` and `submit_extended_fourway_e6e7_rerun.sh` now
  carry `--signal=B:USR1@300` + a `scontrol requeue` trap, same as
  `submit_e6_split_experiment.sh`.
- **Partials are keyed to the SLURM job id.** `scontrol requeue` preserves the
  job id, a fresh `sbatch` does not -- so a requeue resumes, but a new
  submission *wipes* the partials rather than inheriting them. Without that
  guard a resume could silently fold a config solved against superseded data or
  superseded checkpoints into a fresh aggregate, which is exactly the failure
  this entire regeneration pass exists to undo. `--force` re-solves from
  scratch.

**Verified end-to-end on a compute node (jobs 2207508 / 2207509, 2-gene so it
is cheap, 2026-08-24).** The test kills a run mid-flight, restarts it under the
same job id, then simulates a fresh submission:

| Property | Result |
|---|---|
| Killed at the simulated wall, partials survive | **3/12 configs** checkpointed (LowHigh_Base, LowHigh_Aggressive, MediumEven_Base); final JSON correctly still the old 08-23 one. Under the old code this was **0**. |
| Requeue resumes rather than restarts | slot 2 logged `[resume] 3/12 configs already solved in an earlier slot of this job`, skipped those three, solved the remaining nine, wrote the final JSON |
| A new submission does not inherit partials | 12 partials on disk, **0 inherited** -- the job-id key held |
| **Resume does not change the answer** | the resumed seed1 output is **identical** to the known-good pre-test copy (whole-trajectory, root-only, per-config and ratio2 blocks all equal). seed0, run clean, likewise identical. |

One expected side effect: a resumed run's `.log` legitimately contains the
earlier slot's config sections plus a header per slot, so a resumed log has
more than 12 config blocks. The `.json` is unaffected and is what every
analysis script reads -- but do not count config blocks in a resumed `.log`.

Note the per-config mode (`--regime X --preset Y`, driven by
`submit_extended_3gene_perconfig{,_e6,_e7}.sh`, 36 tasks each) was always
immune to this -- one config per task, own JSON written immediately. It remains
the alternative if the resume path is ever in doubt.

**Not yet resubmitted.** 3-gene Extended (e4/e6/e7 x 3 seeds) is still stale
07-26 / 08-09 data, and re-running it is ~108 config-solves either way.


## Finished

| Job ID | Name | Purpose | Result |
|---|---|---|---|
| 2206212 | verify_myopic_TRUE | Myopic ratio2 reference, 24 tasks | **COMPLETE, 24/24.** 3-gene 0.1052 -> 0.1090; 2-gene 0.2303 bit-identical. Already folded into the artifact and commit 07f979d. |
| 2206213 | extended_fourway (e4) | Extended/OOD four-way, variant e4, {2,3}-gene x 3 seeds, 6 tasks | **PARTIAL: 2-gene 3/3 COMPLETE, 3-gene 3/3 TIMEOUT** at the 4h wall. See "3-gene Extended does not fit in one slot" below. The 2-gene results came back bit-identical to the committed ones apart from a cosmetic `variant` label -- a third independent confirmation that 2-gene was untouched by the 3-gene fix, alongside the myopic 2-gene result. |
| 2206261 | ext_e6e7_rerun | Extended/OOD four-way, variants E6+E7, {2,3}-gene x 3 seeds, 12 tasks | **PARTIAL: 2-gene 6/6 COMPLETE, 3-gene 6/6 TIMEOUT** at the 4h wall. Same cause. So all 2-gene Extended rows are now post-correction; all **3-gene** Extended rows (e4/e6/e7) are still the stale 07-26 / 08-09 data. |
| 2204396 (artifact) | — | 3-gene ratio2 figures in the PI briefing artifact replaced from the corrected retrain | DONE 2026-08-23. Per-rung tables, master table, all 12 per-config rows, and every vs-previous-rung delta. Four directional claims flipped (E1 batching no longer a uniform win; E4 bidirectional MP now regresses at 3-gene; E7/E9 now better than E4; E6 MLP 0.214 -> 0.821) and the prose was rewritten to match. 2-gene verified unchanged and correct. |
| 2204396 | ovfit_probe | Overfit probes: curve / randlabel / configholdout x GNN+MLP x 3 seeds, 500 epochs, on the FAIR rotate split (TRAIN=Trio+ThreeGeneration, TEST=Nuclear) | **COMPLETE, 18/18, 0 errors.** Three results: (1) real but shallow overfitting -- GNN val loss bottoms at ep150-250 then drifts up only ~2-3%; MLP shows none. (2) Neither model can memorize shuffled targets -- shuffled-target R2 sits at mean-predictor level (GNN -0.16 to -0.44), and total loss is HIGHER under shuffling. (3) Config-level and family-level generalization gaps are the same size (~1.7x vs ~1.6x), so neither config memorization nor a specific topology-transfer failure is supported. Analysis: `analyze_overfit_probes.py`. |
| 2204343 | ovfit_smoke | 2-epoch smoke of all three overfit probes on the real partition | COMPLETE, 4/4, exit 0. All probes emitted their expected fields; `configholdout` produced all three ratio2 numbers. seed99 output dirs deleted so they do not pollute the results tree. |
| 2202164 | e6_split | E6 2-gene split experiments: rotate/add/loo_trio/loo_nuclear/loo_threegen x GNN+MLP x 3 seeds (30 tasks) | **COMPLETE, 30/30, 0 errors.** The last LOO GNN fold beat the 4h wall by minutes, so the resume machinery never had to fire. Headline: the standard-split "overfitting" is mostly a split artifact -- swapping Nuclear -> ThreeGeneration in TRAIN (same 2 families, same 24 configs) raises GNN TRAIN error ~6x and collapses the gap from 8.3x/3.1x (E0/E6) to 1.3x/1.1x. Confirmed on both E0 and E6, so not rung-specific. |
| 2204292 | e6_split (resume) | Resubmit of the 0-29 array with resume logic, `--dependency=afterany:2202164` | CANCELLED (scancel) -- 2202164 finished all 30 tasks on its own, so every task would have been a <1s no-op. Cancelled rather than let it hold queue slots under `QOSMaxJobsPerUserLimit`. The resume machinery it added stays in the scripts. |
| 2204260 | overfit2g_eval | Standard-split (TRAIN=Trio+Nuclear, TEST=ThreeGeneration) TRAIN-vs-TEST overfit check, 2-gene, E0/E1/E2/E4/E6, 3 seeds | COMPLETE, 3/3, 24s. Result: the overfit gap is a **GNN** phenomenon, not an MLP one -- every GNN rung degrades 3-16x from TRAIN to TEST, every MLP rung is ~1x (E1/E2 MLP even slightly negative). E6 has the SMALLEST GNN gap (3.1x vs E0's 8.3x). |
| 2199861-2199871 | gnn_q_seeds / mlp_q_seeds / e1_gnn_train / e1_mlp_train / gnn_q_ce_train / mlp_q_ce_train / e4_gnn_bidir_train / e5_gnn_rounds3_train / e6_sumpool_train / e7_neighborpool_train / e9_combopool_train | Retrain E0-E9 (3-gene) on corrected allele-frequency data | ALL COMPLETE, 108/108 sub-tasks, 0 errors. E6 is the best rung (avg TEST ratio2 = 0.056 vs E0's 0.058, 3-seed avg). |
| 2201157 | (unnamed) | E0 2-gene split experiments: rotate + add x GNN+MLP x 3 seeds (12 tasks) | COMPLETE, 12/12, 0 errors. Not used further -- E0 isn't the rung that matters, superseded by the E6 version (2202164). |
| 2202143 | e6_smoke | Diagnostic smoke test for e6_split_experiment.py on the real peanut-cpu partition (after the same script looked pathologically slow on the shared login node) | COMPLETE, 38s total. Confirmed the "slowness" was login-node CPU throttling, not a bug -- unblocked the real 2202164 submission. |

## ADP ground truth (separate track, supplementary comparison only -- not part of the E0-E9 GNN/MLP investigation)

| Job ID | Name | Purpose | Result |
|---|---|---|---|
| (original 96-config array, ~job 2182xxx) | adp_ground_truth | Kanix's real ADP solver (dual_dp) on all 96 configs | 72/96 done: all 2-gene (48/48), 3-gene Trio+Nuclear (24/24). |
| 2199700 | adp_gt_fix_3g | Retry 3-gene ThreeGeneration (12 configs) after raising memory | **ALL 12 FAILED** -- not the OOM/pickle bug, a different real error: `RuntimeError: Row generation did not converge ... state_truncation` from Kanix's own `dual_dp.py` (only scanned 110k/1,054,528 states before giving up). This is an algorithmic limit of his ADP solver at this state-space size, not an infra bug -- have not touched his convergence logic, flagged not fixed. |
| 2201156 | smoke_adp_patch | Smoke test of the belief-snapshot pickle-fix on 3-gene Extended_LowHigh_Base | **TIMED OUT** (not completed) -- Extended (9.19M states) needs a longer time limit / bigger job than a smoke test allows. 3-gene Extended ADP remains 0/12, unresolved. |

### Bottom line on ADP: 3-gene ThreeGeneration and Extended (24/96 configs) are NOT done and need a real decision (raise convergence tolerance? accept partial coverage? different approach?) before pursuing further -- not silently retried.
