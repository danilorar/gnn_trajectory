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
| e63 | 2026-11-06 (pod) | (d) with 63 predicted agents (all slots but ego) | (d) flags + `--predict-agents 63` | 26 / 41 | 1.228 | Moving top-1 ADE 2.606 (16 shared slots); follower direction ✓ |
| **Model selection** | 2026-11-06 | Val, rule pre-stated (Amendment 1 (2)) | `train_model.py --evaluate-only --eval-slots 16` | – | – | **d_rerun chosen**: best val moving-agent top-1 ADE (2.462 vs e32 2.520, e63 2.606); all pass the follower direction check |
| v2_d_seed123 | 2026-11-07 (pod) | (d) rerun, seed 123 | d_rerun flags + `--seed 123` (no `--eval-test`; torch 2.8.0+cu128) | 34 / 49 | 1.232 (plan off 1.260) | Variance seeds for d_rerun; not used for selection |
| v2_d_seed2026 | 2026-11-07 (pod) | (d) rerun, seed 2026 | d_rerun flags + `--seed 2026` (no `--eval-test`; torch 2.8.0+cu128) | 21 / 36 | 1.241 (plan off 1.271) | Variance seeds for d_rerun; not used for selection. 3 seeds (42/123/2026): 1.239 ± 0.006 (plan off 1.271 ± 0.011) |

Notes
- Follower test (val, 20 scenes, 6 s travel real / brake / accel): d_rerun 30.7 / 25.3 / 36.7 m, e32 29.5 / 23.3 / 35.5 m,
  e63 29.8 / 20.7 / 34.7 m (`outputs/run_d_rerun/`, `outputs/e32/`, `outputs/e63/`). Direction is right, but with brake-to-stop (ego 3.4 m, follower ~12 m behind) all
  still predict ~21–25 m of travel: the reaction is too weak to be physically plausible.
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
| test150_v3 | 2026-11-06 | 146 test | v2 + fresh env per scene (deterministic), initial-overlap class, at-fault diagnosis | Superseded by v4 (footprint changed) |
| test150_v4 | 2026-11-07 | 146 test | Final replay: covering footprints, frozen setup | At fault CV 1 / WM 0; rear-ended 14 / 16; close-call WM − CV −4.7 pp [−7.6, −2.1] |
| test150_reactive | 2026-11-07 | 146 test | **Pre-registered reactive run** (two-state rule) | **Decision: WM is safer** (close-call 97.5% CI [−8.0, −2.1] pp, p = 1e-4; at fault −0.007 [−0.027, 0.000], p = 0.32; progress CI [−0.005, +0.006]). Report: `outputs/sim/test150_final_report.md` |
| dev100 | 2026-11-06 | 100 val-split scenes (`sim/scene_lists/dev100.json`) | footprint = circles covering each box (real sizes), tracking error | Planner development / crash diagnosis (never test). At fault CV 3 / WM 1, all 'no safe plan existed' (risk first seen 0.4–0.8 s before); 0 'zero predicted risk'; tracking p95 0.24 m |
| reactive_dev (stage 1) | 2026-11-07 | mini + dev100 (never test) | `--reactive idm`: log-path IDM agents; IDM tuned on dev100 (3 attempts; final s0 1.0, T 0.8, a_max 3.0, b 3.0, v0 x1.5) | Validation (a) fails (8 s median 0.65 m / p95 3.2 m; tracking law lags, not IDM), (b) passes (0/20 vs replay 18/20); dev100 reactive: rear-ended 0.23 -> 0.02 (CV), 0.24 -> 0.01 (WM); WM close-call -5.1 pp (p 3e-5). Report: `outputs/sim/reactive_dev/report.md`. Pre-registered reactive test NOT run (needs (a) to pass first) |
| reactive_dev (deviation rule) | 2026-11-07 | mini + dev100 (never test) | IDM only when the leader deviates > 0.3 m from its log; design-doc IDM | Failed on mini: 8 s p95 9.2 m (start-up transients > 0.3 m trigger chain reactions); replaced by the two-state rule |
| reactive_dev (two-state rule) | 2026-11-07 | mini + dev100 (never test) | REPLAY exactly until a deviating leader (off-log ego / REACTIVE agent) with a_IDM < a_ff; REACTIVE = min(a_track, a_IDM), 1 s smoothed feed-forward clipped to ±3; back after 1 s, \|Δs\| < 0.3 m, \|Δv\| < 0.3 m/s; design-doc IDM | Mini: (a) 0.00 m, 0 new overlaps, expert 0 at fault; (b) pass; (d) ego 10% slower: 0 rear-ends, 0 new overlaps, max decel 3.4 m/s². dev100: (a) 0.00 m, 0 new overlaps, expert 0 at fault; (b) 0/20 (replay 18/20); (d): 1 rear-end by a non-eligible creeping car (by design), 2 new overlaps explained (replayed motorcycle merge; two reactive cars merging; stage-1 limitation), max decel at b_max in 13 / 100 scenes; meets the (d) targets recorded 2026-11-07. Reactive test waits for the user's go |

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

**Recorded 2026-11-06, before any reactive run:** chosen checkpoint = `checkpoints/v2_d_rerun/v2_seed42_best.pth`.
Val selection by the pre-stated rule: all three candidates pass the follower test (brake < real < accel); val
moving-agent top-1 ADE on the 16 shared slots is d_rerun 2.462, e32 2.520, e63 2.606 m, so d_rerun is chosen (step 2;
no tie). This checkpoint is fixed for the reactive run.

The pass criterion in step 1 is written out here so that it is fixed. Earlier, the follower test was only read by eye.


### Validation targets for check (d) (recorded 2026-11-07, before any reactive test result)
Check (d): the logged ego driven 10% slower along its logged route, agents in `--reactive idm` mode, mini + dev100.
- 0 rear-ends by an **eligible** reactive agent that had the ego as its leader.
- 0 new agent–agent overlaps (not in the log), or each one explained and shown not to be caused by the reaction rule.
- Max agent deceleration ≤ 6 m/s² (b_max) is allowed. The number of scenes that hit the cap is reported as a limitation.

## Next steps (handoff, 2026-11-07, after the final test)
1. Done: final test (replay v4 + pre-registered reactive run). Decision: WM is safer, on the close-call co-primary
   endpoint; at fault shows no detectable difference (1 event). Report: `outputs/sim/test150_final_report.md`.
2. Done after the test: `LogPathAgent.eligible` now also requires the ScenarioNet type VEHICLE (no simulation change).
3. Open: reactive stage 2 (swept ego footprint, merge anticipation); the WM's follower reaction is too weak; coverage
   beyond 16 slots.
