# Reactive agents: design (no code yet)

Goal: replace pure log replay with nuPlan-style reactive vehicles, so that the rear-end collisions in our
closed-loop results stop being log artefacts and the follower behaviour becomes testable in sim.

## 1. What MetaDrive's `reactive_traffic` does today
Sources: `sim/metadrive/metadrive/manager/scenario_traffic_manager.py`, `policy/idm_policy.py`.
- **Who becomes reactive:** decided once at spawn (`spawn_vehicle`). Only vehicles that (i) are behind the ego
  (`heading_dist < -1 m`), (ii) lie within 15 m laterally, (iii) head within 90° of the ego, (iv) move > 5 m over
  their valid window, and (v) are not static (position std > 3 m). Everything else replays, including every
  vehicle ahead of the ego and every crossing vehicle. Pedestrians and cyclists always replay.
- **Path:** `TrajectoryIDMPolicy` follows the logged positions as a `PointLane` (width 2 m) with PID steering, so
  lateral drift is possible. When it reaches the end of the path, it is despawned. It never switches back to replay.
- **Leader:** found among lidar-detected objects within 20 m (`IDM_MAX_DIST`): the nearest object ahead
  along the path that has a bounding-box corner on the 2 m path. The ego counts, but only once it is physically
  on the path: there is no anticipation of a crossing or merging ego.
- **IDM:** s₀ = 10 m, T = 1.5 s, a = 1 m/s², b = 5 m/s², δ = 10, fixed v₀ = 40 km/h (not the logged speed). Speed control
  runs every 5th step per vehicle (batched); in between it repeats the last acceleration.
- **Consequence:** vehicles that are delayed or crossing never react, and vehicles that do react drift from their log
  even without an interaction (wrong v₀, 10 m jam gap). It is too narrow for our use and too far from the log.

## 2. Proposed design
Every logged **vehicle** that moves is reactive. It keeps its logged path exactly and only its speed changes.
Pedestrians, cyclists, cones and parked vehicles replay, but they still act as leaders.

**Kinematics (no steering controller).** Each agent stores its arclength `s` on its logged polyline (built from the
valid positions) and a speed `v`. Each step: `v ← clip(v + a·dt, 0, v_max)`, `s ← s + v·dt`. Position and heading are taken
from the path at `s` and set directly (`set_position`, `set_heading_theta`, `set_velocity`), like
`ReplayTrafficParticipantPolicy`. This gives zero lateral error by construction, and `s` is integrated, never re-projected, so loops and U-turns are safe.

**Speed law: track the log, and brake for leaders.**
`a = min(a_track, a_IDM)`, capped to [−b_max, a_max].
- `a_track = k_s·(s_log(t) − s) + k_v·(v_log(t) − v)` with k_s = 0.5 s⁻², k_v = 1.0 s⁻¹. Here `s_log(t)` and `v_log(t)`
  come from the 2 Hz positions (not from the interpolated 10 Hz data; see CLAUDE.md pitfalls).
- `a_IDM = a_max·[1 − (v/v₀)^δ − (s*/gap)²]`, with `s* = s₀ + v·T + v·Δv / (2√(a_max·b))`.

| Param | Value | Note |
|---|---|---|
| s₀ | 2.0 m | jam gap (MetaDrive's 10 m is far above urban queue gaps) |
| T | 1.2 s | time headway |
| a_max | 1.5 m/s² | also caps catch-up after a delay |
| b | 2.0 m/s² | comfortable decel in s* |
| b_max | 6.0 m/s² | hard cap on the applied braking |
| δ | 4 | standard |
| v₀ | max(logged max speed × 1.1, 3 m/s) | speed limit if map data has it (nuScenes lanes mostly do not) |

**A deviation from the brief, flagged for your decision:** pure IDM with v₀ = logged max speed would accelerate every
vehicle to its maximum speed and drive it through logged stops (nuScenes has no traffic-light states). The
`min(a_track, a_IDM)` form keeps the stops at red lights and stop signs, reproduces the log exactly when nothing is in the
way, and still lets IDM take over when a leader is closer than the log implies. Pure IDM can stay available as an option.

**Leader search (per agent, every step).** Look at objects (the ego, vehicles, pedestrians, cyclists, cones) within
`H = max(30 m, v·T + 20 m)` ahead. Project the 4 footprint corners onto the agent's path, restricted to the window `[s, s+H]`.
An object is on the path if any corner has |lateral| < w_self/2 + 0.3 m. Then gap = min corner arclength − s − L_self/2,
and the leader speed = the object's velocity projected on the path tangent at that point. A crossing ego therefore counts as a slow
obstacle on the path (Δv large, so the agent brakes), which is the conservative behaviour we want.
- **Ego merging or crossing:** in addition, sweep the ego's footprint along a constant-velocity extrapolation for the next 1.0 s (option: its committed plan, see the open question).
  If the swept footprint enters the path ahead, treat it as a leader at its first entry point with speed 0. This avoids
  braking only once the ego is already on the path.
- **Update order:** compute every acceleration from the previous step's state, then integrate them all together, so
  there is no dependence on agent order.

## 3. Edge cases
- **Parked in the log:** displacement over the track < 2 m, or max logged speed < 0.5 m/s, means replay (static
  obstacle). Also replay vehicles that reverse in the log (heading differs from the motion direction by more than 90°) and tracks
  shorter than 2 s or 5 m.
- **Starts outside the map or appears mid-scene:** the path is the logged polyline, so no map lanes are needed. Spawn
  when the track becomes valid (as now). If the spawn footprint overlaps another object, keep replaying it until it is clear.
  For gaps in a track, use the longest valid segment (`get_max_valid_indicis`).
- **U-turns and loops:** handled by the integrated `s` plus the windowed leader projection. An object on the
  return leg within H counts as genuinely in the path.
- **End of path:** despawn when `s` reaches the path end. If the agent is delayed past the end of the log, stop it at the path
  end (speed 0) instead of despawning it in the middle of the road.
- **Mutual leaders (junction deadlock):** if agents A and B are each other's leaders, the one that passes the conflict
  point first in the log keeps priority and ignores the other. Backstop: if both have been at v < 0.1 m/s for more than 3 s, the
  log-earlier agent goes. The ego is never ignored, and an agent/ego standoff for more than 10 s is logged as a deadlock
  metric (not hidden).
- **Switch back to replay:** implicit. With `a_track` active, an agent with no leader converges to its log. Snap to
  exact replay once |s − s_log| < 0.3 m and |v − v_log| < 0.3 m/s and there has been no leader within H for 1 s (hysteresis). Leave
  replay again as soon as a leader appears.

## 4. Validation plan (on the 10 mini dev scenes only; parameters frozen before test150)
**(a) Ego drives the log** (expert replay). For every reactive agent, report
the along-track error |s − s_log| and the Euclidean error at 1, 3, 5 and 8 s and over the episode, plus the share of agents
with error > 1 m and the share of steps where `a_IDM` was binding. Target: median < 0.2 m, p95 < 1 m at 8 s,
0 agent–agent collisions beyond those in the log. Cases that fail are logged gaps < s₀ + vT, which tells us how to tune s₀ and T.

**(b) Follower test in sim:** scenes with a vehicle in the same lane behind the ego within 15 m (same selection as the
open-loop follower test). The ego replays until t₀, then brakes to a stop at −4 m/s². Check: the follower brakes (report reaction
delay, peak deceleration and minimum gap), no collision, and the follower stays behind the ego. Also run a
cut-in case (the ego merges into an agent's path) to exercise the swept-footprint leader. The control is the same run with log
replay, which should rear-end.

Then re-run `sim_check.py plan` on the mini scenes with reactive agents and compare the rear-end and at-fault collision counts with replay.

## 5. Effort and files
About 2–3 days: speed law and kinematics, 0.5 d; leader projection, swept ego and deadlock rule, 1 d; validation (a)/(b),
0.5 d; tuning on mini, 0.5 d.
- `sim/closed_loop.py`: `LogPathIDMAgent` (path, `s`, speed law, leader search) and a `ReactiveTraffic` step
  that updates all agents together (~250 lines). No new file, in line with the few-scripts preference.
- MetaDrive hook without editing the clone: a subclass of `ScenarioTrafficManager` that assigns our policy in
  `spawn_vehicle`, registered by a small `ScenarioEnv` subclass (overriding `setup_engine`), plus `reactive_traffic=False`.
- `scripts/sim_check.py`: `--reactive {off,metadrive,idm}` (today's `--reactive` becomes `metadrive`); `follow` gets
  `--reactive` for validation (a) and a `--brake-at` option for (b).
- `docs/CODE_GUIDE.md`: one section. No model or planner changes.

Open question: should the ego's planned trajectory (the swept check) be visible to agents? It is a mild "agents read the planner's
mind" effect. The alternative is a constant-velocity extrapolation of the ego, which is more realistic and is the default I'd suggest.
