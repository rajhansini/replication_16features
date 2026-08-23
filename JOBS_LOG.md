# SLURM job log

Last updated: 2026-08-23 (updated every time a new job is submitted or a
tracked job's status changes materially). Only jobs from this investigation
are listed -- `squeue -u $(whoami)` also shows unrelated jobs (nlp_rq*,
rarm7_*, tex_*, etc.) from other projects, not touched or tracked here.

To check status yourself: `squeue -u $(whoami) | grep <jobname>` or
`sacct -j <jobid> --format=JobID,State,Elapsed -n`

## Currently running / pending

| Job ID | Name | Purpose | Status (as of last update) |
|---|---|---|---|
| 2204396 | ovfit_probe | Full overfit-probe array: curve / randlabel / configholdout x GNN+MLP x 3 seeds, 500 epochs, 18 tasks. All on the FAIR rotate split (TRAIN=Trio+ThreeGeneration, TEST=Nuclear) | SUBMITTED. Smoke-tested first (2204343, 4/4 clean). Reference cost: the comparable e6_split rotate GNN tasks ran ~22min, MLP ~4min; `curve` adds a per-epoch validation forward pass on top. |

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

## Finished

| Job ID | Name | Purpose | Result |
|---|---|---|---|
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
