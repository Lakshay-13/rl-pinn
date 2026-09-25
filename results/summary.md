# Archived result summary

These values summarize the completed local run reports used for the reported schedule comparison. The public release keeps the compact numeric summary and complete run configuration; raw checkpoints, logs, and per-run JSON arrays are excluded.

Input hashes and the numeric evaluation definition are recorded in [provenance.json](provenance.json).

## Standard endpoint potential RMSE

Potential RMSE is recomputed from each saved `best` potential array against the reference curve on the stored 100-point uniform grid over `phi = [0, 1.8]`. Schedules 1–4 and 6 use their final saved cycle report. Schedules 5, 7, and 8 use the separate 100K inference report. The non-localized reference uses its final cycle report.

| Run | Replicates | Memory targets by cycle | Replicate RMSE | Mean | Median |
|---|---:|---|---|---:|---:|
| Non-localized Flexplore reference | 1 | 10K/20K/20K/30K/30K | 1.621 | 1.621 | 1.621 |
| Schedule 1 | 3 | 10K/20K/20K/30K/30K | 0.770, 1.746, 1.623 | 1.379 | 1.623 |
| Schedule 2 | 3 | 20K/25K/25K/35K/35K | 1.973, 0.761, 1.626 | 1.454 | 1.626 |
| Schedule 3 | 3 | 10K/20K/30K/40K/50K | 1.246, 1.802, 1.639 | 1.563 | 1.639 |
| Schedule 4 | 3 | 10K/20K/20K/30K/30K/30K | 1.845, 1.726, 1.665 | 1.746 | 1.726 |
| Schedule 5 | 1 | 20K/40K/40K/60K/60K | 1.661 | 1.661 | 1.661 |
| Schedule 6 | 1 | 100K/200K/200K/300K/300K | 2.964 | 2.964 | 2.964 |
| Schedule 7 | 3 | 5K/10K/10K/20K/30K | 1.446, 1.142, 1.882 | 1.490 | 1.446 |
| Schedule 8 | 3 | 30K/30K/30K | 10.165, 1.440, 1.889 | 4.498 | 1.889 |

The strongest individual endpoints are Schedule 2 replicate 2 (0.761), Schedule 1 replicate 1 (0.770), and Schedule 7 replicate 2 (1.142). Schedule 1 has the lowest mean; Schedule 7 has the lowest median among the localized schedules. The Schedule 8 mean is dominated by its first replicate, so its median is also shown.

## Inference extensions

| Checkpoint | Inference epochs | Potential RMSE |
|---|---:|---:|
| Schedule 1 replicate 1 | 200K | 1.099 |
| Schedule 1 replicate 1 | 500K | 1.233 |
| Schedule 2 replicate 2 | 200K | 2.171 |
| Schedule 2 replicate 2 | 500K | 2.066 |
| Schedule 5 replicate 1 | 100K | 1.661 |
| Schedule 5 replicate 1 | 200K | 1.825 |
| Schedule 5 replicate 1 | 300K | 1.869 |

These longer rollouts are evaluations of the same saved controllers, not additional training replicates. Their errors need not improve with a longer inference budget.

## Interpretation and provenance

- The loss field in each render report is named `best_unweighted`. The evaluation path does not explicitly reset the equation scales before calculating that value, so it is retained as a reported optimization diagnostic and is not treated here as a confirmed scale-independent residual comparison.
- RMSE is computed directly from the saved potential and reference arrays; it is not minimized across controller cycles. The “best” potential is the solver-selected state tracked by the run.
- Replicate counts are three for Schedules 1–4, 7, and 8; one for Schedules 5 and 6 and the reference. The archived campaigns did not record fixed random seeds.
- The numeric values were checked against the archived `render_report.json` arrays before release. Those raw reports, logs, and checkpoints are not included. Use `configs/schedules.json` and `./reproduce.sh` to generate a new run archive under the ignored `repro_runs/` directory.
