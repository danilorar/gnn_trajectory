# Experiments

One row per training run. **Add a row for every new run** (including pod runs, while they are still running).
Metrics: best-epoch **val** target minADE₆ (plan on for ego-plan runs). Test numbers do not go here
(test is looked at once, at the end; see CLAUDE.md).
Checkpoints: `checkpoints/<run>/v2_seed<N>_best.pth` + `.json`; logs: `outputs/train_v2_<run>_seed<N>.log`.

| Run | Date | Change vs previous | Key flags | Best / stop epoch | Val minADE₆ [m] | Verdict |
|---|---|---|---|---|---|---|
| v1 GRU baseline | 2026-10-19 | GATv2 + GRU decoder (branch feature/trajectory-prediction) | see README | n/a | n/a (only test numbers recorded) | Reference model from the v1 report |
| v2_check | 2026-11-05 | First v2 scene transformer, 5-epoch check | `--epochs 5` | 5 / – | 1.852 | Pipeline works; still improving |
| v2_anchor (k*) | 2026-11-05 | Anchor-based mode assignment | `--assign anchor` | 25 / 38 | 1.840 | Much worse than predicted assignment; dropped |
| angle anchors (stopped) | 2026-11-05 | Angle-based anchors | log: `train_v2_angle_anchors_seed42_stopped.log` | stopped early | – (2.377 at ep 5) | Dead mode (<3%); stopped |
| v2_predicted (k*), seed 42 | 2026-11-05 | Predicted mode assignment, 60 epochs | `--assign predicted` | 30 / – | 1.222 | Baseline v2; overfits after ep 30 |
| v2_predicted, seed 123 | 2026-11-05 | Same, seed 123 | | 28 / 38 | 1.220 | Seed spread |
| v2_predicted, seed 2026 | 2026-11-05 | Same, seed 2026 | | 30 / 40 | 1.252 | Seed spread: 1.231 ± 0.015 over 3 seeds |
| v2_topology | 2026-11-05 | + lane topology in attention bias | `--topology` | 33 / 43 | 1.226 | No gain (within seed spread) |
| v2_scale25 | 2026-11-06 | Data scaling: 25% of train | train_fraction 0.25 | 56 / – | 1.278 | Data-scaling point (first attempt cut by power-off) |
| v2_scale50 | 2026-11-06 | Data scaling: 50% of train | train_fraction 0.5 | 39 / 49 | 1.255 | Data-scaling point |
| v2_multi_check | 2026-11-05 | Block 1: multi-agent prediction, 5-epoch check | `--predict-agents 16 --epochs 5` | 5 / – | 1.621 | Check only |
| (a) | ? | **Not found in repo**: fill in | ? | ? | ? | ? |
| v2_b12 (b) | 2026-11-06 | Blocks 1+2: 16 agents + ego-plan token | `--predict-agents 16 --ego-plan` (plan_drop 0.5) | 22 / 32 | 1.367 | Target accuracy drops vs single-agent v2 (1.22) |
| v2_b123 (b) | 2026-11-06 | Blocks 1+2+3: + soft scores | `... --soft-scores` | 19 / 29 | 1.350 | Soft scores ≈ no change |
| v2_c (c) | 2026-11-06 | Target-weighted loss | `--predict-agents 16 --ego-plan --target-split` | 28 / – | 1.265 | Recovers most target accuracy |
| v2_d (d) interrupted | 2026-11-06 | + plan in each agent's frame, plan dropout 0.2 | `... --plan-queries --plan-drop 0.2` | 28 / – (interrupted) | – (no val in manifest) | Interrupted; rerun below |
| v2_d_rerun (d) | 2026-11-06 | Rerun of (d) | `--predict-agents 16 --ego-plan --target-split --plan-queries --plan-drop 0.2` | 40 / 55 | 1.243 (plan off 1.281) | Current reference model (follower test: `outputs/run_d_rerun/`) |
| e32 | running (pod) | (d) with 32 predicted agents | (d) flags + `--predict-agents 32` | – | 1.351 at ep 25 | Running; behind (d) |
| e63 | running (pod) | (d) with 63 predicted agents (all slots but ego) | (d) flags + `--predict-agents 63` | – | 1.266 at ep 23 | Running; close to (d) |

Notes
- e32 / e63 exist to raise planner coverage: with 16 slots, busy scenes (~22 nearby agents) get only ~68% world-model
  coverage; the rest falls back to constant velocity.
- "All agents minADE₆" is not comparable across different `--predict-agents` values (different agent sets).
  Compare runs on target minADE₆.
