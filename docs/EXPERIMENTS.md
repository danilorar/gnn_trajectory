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
| v2_b123 (a) | 2026-11-06 | Blocks 1+2+3: 16 agents + ego-plan token + soft scores | `--predict-agents 16 --ego-plan --soft-scores` (plan_drop 0.5) | 19 / 29 | 1.350 | Soft scores: minADE₆ ≈ (b), but top-1 ADE worse (3.50 vs 3.36); dropped |
| v2_b12 (b) | 2026-11-06 | Blocks 1+2, no soft scores | `--predict-agents 16 --ego-plan` (plan_drop 0.5) | 22 / 32 | 1.367 | Target accuracy drops vs single-agent v2 (1.22) |
| v2_c (c) | 2026-11-06 | Target-weighted loss | `--predict-agents 16 --ego-plan --target-split` | 28 / – (interrupted at ep 33, laptop suspend) | 1.265 | Recovers most target accuracy; plan has no effect (follower test) |
| v2_d (d) interrupted | 2026-11-06 | + plan in each agent's frame, plan dropout 0.2 | `... --plan-queries --plan-drop 0.2` | 28 / – (interrupted at ep 31, display freeze) | 1.251 (plan off 1.296; from the log) | World model v1 (used in the mini planner development); rerun below |
| v2_d_rerun (d) | 2026-11-06 | Rerun of (d) | `--predict-agents 16 --ego-plan --target-split --plan-queries --plan-drop 0.2` | 40 / 55 | 1.243 (plan off 1.281) | Current reference model (follower test: `outputs/run_d_rerun/`) |
| e32 | 2026-11-06 (pod) | (d) with 32 predicted agents | (d) flags + `--predict-agents 32` | 36 / 51 | 1.222 | Moving top-1 ADE 2.520 (16 shared slots); follower direction ✓ |
| e63 | 2026-11-06 (pod) | (d) with 63 predicted agents (all slots but ego) | (d) flags + `--predict-agents 63` | 26 / 41 | 1.228 | Moving top-1 ADE 2.606 (16 shared slots); follower test pending |
| **Model selection** | 2026-11-06 | Val, rule pre-stated (Amendment 1 (2)) | `train_model.py --evaluate-only --eval-slots 16` | – | – | **d_rerun chosen**: best val moving-agent top-1 ADE (2.462 vs e32 2.520, e63 2.606); d_rerun and e32 pass the follower direction check, e63 pending (see Next steps) |

Notes
- Follower test (val, 20 scenes, 6 s travel real / brake / accel): d_rerun 30.7 / 25.3 / 36.7 m, e32 29.5 / 23.3 / 35.5 m
  (`outputs/run_d_rerun/`, `outputs/e32/`). Direction is right, but with brake-to-stop (ego 3.4 m, follower ~12 m behind) both
  still predict ~23–25 m of travel: the reaction is too weak to be physically plausible.
- e32 / e63 exist to raise planner coverage: with 16 slots, busy scenes (~22 nearby agents) get only ~68% world-model
  coverage; the rest falls back to constant velocity.
- "All agents minADE₆" is not comparable across different `--predict-agents` values (different agent sets).
  Compare runs on target minADE₆.

## Closed-loop evaluations
Planner weights frozen on the 10 mini dev scenes (col 500, progress 1, comfort 0.1, offset 0.5, veto early risk > 0.3
within 2 s). WM = v2_d_rerun unless noted. Results: `outputs/sim/<run>/report.md`.

| Run | Date | Scenes | Setting | Status |
|---|---|---|---|---|
| mini dev runs | 2026-11-06 | 10 mini | v2_d (ep 28), then v2_d_rerun | Development set (weights chosen here) |
| test150 (v1) | 2026-11-06 | 146 test | MetaDrive ego 4.52 x 1.85 m, stopped-ego collisions at fault | Superseded (classifier false positives) |
| test150_v2 | 2026-11-06 | 146 test | Zoe ego 4.08 x 1.73 m, ego < 0.5 m/s not at fault, near-miss metrics | Superseded: order-dependent (MetaDrive physics state across resets; 1 of 15 re-rendered scenes changed) |
| test150_v3 | 2026-11-06 | 146 test | v2 + fresh env per scene (deterministic), initial-overlap class, at-fault diagnosis | Current non-reactive result |

## Pre-registration: closed-loop test with reactive agents
Written 2026-11-06, before any reactive run on the test scenes. Nothing below may change after the first reactive
test run; deviations are reported as such.

**Question.** With agents that react to the ego (instead of log replay), does the world-model planner (WM) cause fewer
at-fault collisions than the constant-velocity planner (CV), without less progress?

**Agents.** The log-path IDM agents of `docs/reactive_agents_design.md` (vehicles keep their logged path, speed =
min(log tracking, IDM)), not MetaDrive's built-in `reactive_traffic` (too narrow: only vehicles spawning behind the ego).
Their parameters are tuned and frozen on the 10 mini dev scenes using the design doc's validation (a) and (b), before the
test run. If validation fails its targets (median along-track error < 0.2 m and p95 < 1 m at 8 s with the ego driving
the log; the follower brakes without collision), the test run does not start and this is reported.

**Fixed for the test run.** Scenes: `sim/scenarios/nuscenes_test150` (146 = official nuScenes val minus the 4 dev mini
scenes). Planner, cost weights, veto, plan grid, tracker: as in test150_v3. WM checkpoint: see Amendment 1, (2). Ego:
Zoe size. Fresh MetaDrive env per scene. One run per planner (CV, WM) plus the expert (logged ego, agents reactive).

**Collision classes** (as in test150_v3): initial overlap (first contact within 0.5 s); not at fault (the other object
hits the ego from behind, or the ego is below 0.5 m/s); at fault (all others).

**Primary endpoint.** At-fault collisions per scene, paired WM − CV over the 146 scenes: mean difference with a
bootstrap 95% CI (10,000 resamples over scenes, seed 0) and a two-sided Wilcoxon signed-rank p-value.

**Secondary endpoints** (same paired analysis): progress ratio (scenes where the logged ego moves), not-at-fault
collisions per scene, min distance, min TTC (capped at 5 s), share of steps with distance < 1 m. Also reported, not
tested: off-road, max |a| and |jerk|, per-map breakdown, WM coverage, planning time, and the at-fault diagnosis (no safe
plan / cost picked a risky plan / prediction miss).

**Decision rules.**
- "WM is safer" only if the primary CI lies entirely below 0 and progress is not worse (progress CI lower bound > −0.02).
- If the primary CI contains 0, the result is "no detectable difference", not "equal": with ~5–10 at-fault events per
  planner the test can only detect large effects; the event counts are reported with it.
- Secondary endpoints are reported with their CIs but are not used to claim the primary result.
- The reactive vs replay comparison (same planner, test150_v3 vs reactive) is descriptive: rear-ended collisions are
  expected to drop; at-fault counts are compared per planner with the same paired analysis.

**Reported regardless of outcome.** The full per-scene table (JSON), the 5 worst scenes per planner with GIFs, and a
random spot-check of 5 at-fault and 5 not-at-fault events with frames.

### Amendment 1 (2026-11-06, made before any reactive run)
Made before any reactive run on any scene (mini or test). The non-reactive test results test150_v2 had been seen when
it was written: there, the close-call share was the only significant paired difference (WM − CV −2.4 pp, 95% CI
[−4.3, −0.7], Wilcoxon p = 0.02). This amendment overrides the corresponding parts above.

**(1) Two co-primary endpoints, Bonferroni-corrected.**
- At-fault collisions per scene, and the close-call share (share of steps with ego–agent box distance < 1 m), both
  paired WM − CV over the 146 scenes.
- Each is tested at α = 0.025: a bootstrap **97.5%** CI (10,000 resamples over scenes, seed 0) and a two-sided Wilcoxon
  signed-rank p-value compared with 0.025.
- Decision: "WM is safer" if **either** 97.5% CI lies entirely below 0 **and** progress is not worse by more than 0.02
  (lower bound of the progress 95% CI > −0.02). Otherwise "no detectable difference".
- Secondary endpoints are unchanged and are no longer co-primary: progress ratio, not-at-fault collisions per scene, min
  distance, min TTC.

**(2) Checkpoint = the final model chosen on val, fixed before the reactive run.** It is not necessarily v2_d_rerun.
The candidates are the multi-agent ego-plan models in the run table, including e32 and e63 when they finish. Only val
(open loop) and the mini dev scenes (closed loop) may be used, never test. The rule is applied in order:
1. **Follower test passed** (open-loop `--follower-test`, 20 val scenes). The braking plan must lower the follower's
   predicted 6 s travel below the real plan's, and the accelerating plan must raise it above the real plan's.
2. Among those, the **best val moving-agent top-1 ADE**.
3. Tie-break: **higher coverage** (more predicted agent slots). Within 0.02 m counts as a tie.

The chosen checkpoint, the numbers behind the choice, and the date are recorded here before the reactive run starts.

The pass criterion in step 1 is written out here so that it is fixed. Earlier, the follower test was only read by eye.

## Next steps (handoff, 2026-11-06)
1. Run the e63 follower test (it was queued behind a sim GPU job), on val, one GPU job at a time:
   `.venv/bin/python scripts/run_turn_angle_analysis.py --model v2 --checkpoint checkpoints/v2_e63/v2_seed42_best.pth --follower-test outputs/e63`
   If brake < real < accel for the follower's 6 s travel, change "e63 pending" to ✓ in the e63 and Model selection
   rows. If it fails, e63 drops out; the choice (d_rerun) does not change either way.
2. Record the chosen checkpoint in Amendment 1 (2) before any reactive run (d_rerun, numbers in the Model selection row).
3. Uncommitted: the new test rule in `CLAUDE.md` (the user decides when to commit it).
4. Open: the follower reaction is too weak (see Notes). Consider this before relying on the WM for followers.
