# Reactive agents, stage 1: validation on the dev scenes (mini + dev100; never test)

Date: 2026-11-07 (overnight run). Code: `sim/closed_loop.py` (`ReactiveTraffic`, `LogPathAgent`), `scripts/sim_check.py reactive-check` and `--reactive idm`.

## Summary

- **(b) Sim follower test: passes.** With reactive agents the follower never hits the braking ego (dev100: 0 of 20 cases; log replay: 18 of 20).
- **(a) Expert drives the log: fails** the design-doc targets (median < 0.2 m and p95 < 1 m at 8 s, 0 new agent–agent collisions), also after the 3 allowed tuning attempts on dev100: final 8 s error median 0.65 m, p95 3.2 m; 20 new agent–agent overlaps.
  The median is set by the speed-tracking law, not by IDM: IDM binds in only 9% of agent-steps, and the error is already 0.15 m after 1 s. `a_track = k_s·Δs + k_v·Δv` has no feed-forward of the logged acceleration, so agents lag accelerating logs. Fixing this is a design change (outside the IDM parameters) and is left for your decision.
- **(c) Planners on dev100 in reactive mode:** rear-end collisions almost disappear (CV 0.23 → 0.02 / scene, WM 0.24 → 0.01). At-fault collisions are unchanged (CV 3, WM 1; all 'no safe plan existed'). WM vs CV: close-call share −5.1 pp [−8.1, −2.6] (p = 3e-5), min distance +0.16 m [+0.09, +0.24]; at fault −0.02 [−0.05, 0.00] (p = 0.16); progress +0.015 [−0.001, +0.033].
- **Caveat from (a):** because agents drift from their logs, the logged expert now has 3 at-fault collisions in reactive mode (0 in replay), and there are ~20 new agent–agent overlaps per 100 scenes. The reactive results are usable for rear-end behaviour, but not yet as a faithful log reproduction.

## Decisions implemented (from the brief)

- Every moving logged vehicle keeps its logged path (polyline of its longest valid run, 0.5 m resampling); speed `a = clip(min(a_track, a_IDM), −b_max, a_max)`, b_max = 6 m/s². Pedestrians, cyclists, parked (< 2 m displacement or max speed < 0.5 m/s), reversing and short (< 2 s or < 5 m) vehicles replay.
- Leader = nearest object physically on the path ahead (any footprint corner within w/2 + 0.3 m of the path, window [s, s + H], H = max(30 m, vT + 20 m)), including the ego by its current state; no swept-footprint anticipation (stage 2), agents never see the ego's plan.
- Synchronous update; mutual leaders: the agent that reaches the conflict point earlier in its log keeps priority. An agent that would spawn overlapping the ego replays until clear. At the end of its path it stops, and it is removed once its log has ended.
- New at-fault rule in reactive mode: a hit from behind by a reactive agent within 2 s after the ego entered that agent's path = at fault ('unsafe cut-in'). It did not trigger on dev100.
- Not implemented from the design doc (stage 1 simplifications): the deadlock backstop (both stuck > 3 s), explicit snap-back to replay with hysteresis (convergence comes from `a_track` only), cut-in validation case.

## (a) Expert drives the log: agent position error vs log

| set / IDM parameters | 1 s median | 3 s median | 5 s median | 8 s median | 8 s p95 | agents > 1 m at 8 s | IDM binding | new agent–agent collisions | max ego standoff |
|---|---|---|---|---|---|---|---|---|---|
| dev100, design doc: s0 2.0, T 1.2, a_max 1.5, b 2.0, v0 x1.1 | 0.16 | 0.72 | 0.89 | 1.04 | 12.32 | 51% | 39% | 26 (log: 17) | 13.8 s |
| dev100, 1: a_max 3.0, T 0.8, s0 1.5, b 3.0 | 0.15 | 0.62 | 0.80 | 0.66 | 4.69 | 38% | 17% | 21 (log: 18) | 13.0 s |
| dev100, 2: a_max 3.0, T 0.5, s0 1.0, b 3.0, v0 x1.5 | 0.15 | 0.64 | 0.77 | 0.66 | 2.94 | 37% | 7% | 19 (log: 18) | 13.0 s |
| dev100, 3 (final): a_max 3.0, T 0.8, s0 1.0, b 3.0, v0 x1.5 | 0.15 | 0.63 | 0.76 | 0.65 | 3.21 | 36% | 9% | 20 (log: 18) | 13.0 s |
| mini, design doc | 0.26 | 1.42 | 1.75 | 1.40 | 12.46 | 60% | 41% | 5 (log: 1) | 4.9 s |
| mini, final (attempt 3) | 0.18 | 0.82 | 0.94 | 0.70 | 1.95 | 39% | 10% | 4 (log: 1) | 7.1 s |

Errors in metres (Euclidean; along-track is the same within a few cm, since the path is kept exactly). Target: 8 s median < 0.2 m and p95 < 1 m, 0 new agent–agent collisions. 'log' = overlaps already present in the log. Standoff = longest time a reactive agent waited behind a stopped ego.

Tuning log (IDM parameters only, on dev100; stopped after 3 attempts):
1. a_max 1.5 → 3.0 (logged accelerations reach ~3 m/s²), T 1.2 → 0.8 s, s0 2.0 → 1.5 m, b 2.0 → 3.0: queues no longer brake below the logged gaps (binding 39% → 17%); p95 12.3 → 4.7 m; median 1.04 → 0.66 m.
2. T 0.5 s, s0 1.0 m, v0 = 1.5 × logged max speed: binding 7%, p95 2.9 m, median unchanged (0.66 m), but the follower test fails once (1 of 20).
3. T 0.8 s, s0 1.0 m, v0 × 1.5 (final): p95 3.2 m, median 0.65 m, follower test 0 of 20. Kept as the default (`closed_loop.IDM`).

## (b) Sim follower test (ego brakes to a stop at −4 m/s²; follower 2–15 m behind in the same lane)

| set / parameters | cases | collisions reactive | collisions replay | min gap (median) | follower peak decel (median) | follower stays behind |
|---|---|---|---|---|---|---|
| dev100, design doc: s0 2.0, T 1.2, a_max 1.5, b 2.0, v0 x1.1 | 20 | 0 | 18 | 1.97 m | 2.9 m/s² | 18 / 20 |
| dev100, 1: a_max 3.0, T 0.8, s0 1.5, b 3.0 | 20 | 0 | 18 | 1.49 m | 3.3 m/s² | 18 / 20 |
| dev100, 2: a_max 3.0, T 0.5, s0 1.0, b 3.0, v0 x1.5 | 20 | 1 | 18 | 0.98 m | 3.6 m/s² | 19 / 20 |
| dev100, 3 (final): a_max 3.0, T 0.8, s0 1.0, b 3.0, v0 x1.5 | 20 | 0 | 18 | 1.00 m | 3.6 m/s² | 19 / 20 |
| mini, final | 1 | 0 | 1 | 0.99 m | 3.7 m/s² | 1 / 1 |

Mini has only 1 qualifying case. Reaction delay is reported as 0 s in the JSON: the follower is already decelerating at t0 in most cases (car-following), so the metric is not informative here. Cut-in case: not run (stage 2).

## (c) CV and WM planners on dev100: replay vs reactive (final IDM parameters)

| | at-fault / scene | rear-ended / scene | progress | close-call share | min distance | min TTC |
|---|---|---|---|---|---|---|
| CV planner, replay | 0.030 | 0.230 | 0.927 | 14.3% | 1.06 m | 2.53 s |
| CV planner, reactive | 0.030 | 0.020 | 0.920 | 8.4% | 1.30 m | 2.75 s |
| WM planner, replay | 0.010 | 0.240 | 0.950 | 9.9% | 1.19 m | 2.61 s |
| WM planner, reactive | 0.010 | 0.010 | 0.935 | 3.3% | 1.46 m | 2.70 s |
| expert (log), replay | 0.000 | 0.000 | 1.000 | 8.1% | 1.53 m | 3.25 s |
| expert (log), reactive | 0.030 | 0.000 | 1.000 | 8.5% | 1.51 m | 3.15 s |

Paired WM − CV, reactive dev100 (100 scenes): at fault −0.020 [−0.050, 0.000] (Wilcoxon p = 0.157); close-call share −5.1 pp [−8.1, −2.6] (p = 3.0e-5); progress +0.015 [−0.001, +0.033] (p = 0.017); min distance +0.16 m [+0.09, +0.24] (p = 1e-4). Full report: `outputs/sim/reactive_dev/dev100_reactive/report.md`.

At-fault events in reactive mode (all 'no safe plan existed', risk first seen 0.3–0.8 s before): CV scene-0032 (motorcycle), scene-0262 (pedestrian), scene-0263 (trailer); WM scene-0032 (motorcycle). Expert at fault (artefacts of the agents' drift from their logs): scene-0005 (bus, car), scene-0159 (truck).

## Next steps (for your decision)

1. Add a feed-forward of the logged acceleration to `a_track` (and/or raise k_v), then re-run (a): expected to remove most of the median error. This is a change to the speed law, not IDM tuning.
2. Stage 2: swept-footprint anticipation of the ego (cut-in case), deadlock backstop, snap-back hysteresis.
3. Only after (a) passes: freeze the parameters and run the pre-registered reactive test (docs/EXPERIMENTS.md). No test run was started.


## Update 2026-11-07: feed-forward + snap-back + timing fix (approved changes), rerun of (a) and (b)

Changes: a_track = a_log + k_s Δs + k_v Δv (a_log from the 2 Hz positions); snap back to exact replay after 1 s
without a leader within H and |Δs| < 0.3 m, |Δv| < 0.3 m/s. Bug fixed on the way: the agents were tracked one sim
step out of phase with the log (≈ v · 0.1 s error), which caused most of the overnight median error. IDM parameters
unchanged (attempt 3; note they were tuned while the timing bug was present).

| set | 1 / 3 / 5 / 8 s median [m] | 8 s p95 [m] | agents > 1 m at 8 s | IDM binding | new agent–agent overlaps (log) | expert at fault | (b) collisions reactive / replay |
|---|---|---|---|---|---|---|---|
| mini | 0.10 / 0.17 / 0.10 / 0.04 | 2.18 | 17% | 11% | 3 (1) | 0 | 0 / 1 (1 case) |
| dev100 | 0.03 / 0.06 / 0.04 / 0.03 | 2.70 | 10% | 10% | 20 (18) | 3 (scene-0005 bus, car; scene-0159 truck) | 0 / 18 (20 cases) |

**Verdict: (a) still fails** (median target met, p95 < 1 m and 0 new agent–agent overlaps not met; the expert is not
~0 at fault on dev100). (b) passes. **No test run was started.** The tail comes from the ~10% of agent-steps where
IDM binds (leader interactions), with parameters chosen before the timing fix.
