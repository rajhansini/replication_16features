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
| 2202164 | e6_split | E6 2-gene split experiments: rotate/add/loo_trio/loo_nuclear/loo_threegen x GNN+MLP x 3 seeds (30 tasks) | 22/30 done, 0 errors. Remaining 8 = GNN folds of loo_trio/loo_nuclear/loo_threegen (all train on Extended, heaviest combo), at epoch 400-480 of 500 with ~50min of a 4h wall left. Expected to TIME OUT -- this job predates the resume logic, so it will be picked up by 2204292 below. No work is lost: checkpoint.pt is written every epoch. |
| 2204292 | e6_split (resume) | Same 0-29 array, resubmitted with resume logic, `--dependency=afterany:2202164` | PENDING (Dependency) -- starts the moment 2202164 fully ends. The 22 finished tasks exit in <1s via the skip guard; the 8 unfinished ones resume from their last checkpointed epoch. |
| 2204260 | overfit2g_eval | Standard-split (Trio+Nuclear train / ThreeGeneration test) TRAIN-vs-TEST overfit check, now extended to include E6 alongside E0/E1/E2/E4 | PENDING -- `QOSMaxJobsPerUserLimit`, queued behind the e6_split jobs. Expected fast (~30s) once running, based on job 2202143's timing. |

### Resume logic (added 2026-08-23)

Every partition on this cluster (`general`, `peanut-cpu`, `threedle-*`) has a
hard **4h MaxTime** -- there is no longer limit to ask for. The leave-one-out
GNN folds do not fit in 4h, so they can only finish by checkpointing and
requeueing across several 4h slots. `submit_e6_split_experiment.sh` +
`incremental_experiments/e6_split_experiment.py` now do that:

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
