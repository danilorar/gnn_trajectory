# Final closed-loop test (146 test scenes): replay v4 and the pre-registered reactive run

Date 2026-11-07. Frozen setup: v2_d_rerun (epoch 40), cost weights / veto frozen on the mini dev scenes, covering-circle footprints, Zoe-size ego, fresh MetaDrive env per scene. Reactive agents: two-state REPLAY / REACTIVE rule, design-doc IDM parameters, validated on mini + dev100 (`outputs/sim/reactive_dev/report.md`). Full reports: `outputs/sim/test150_v4/report.md`, `outputs/sim/test150_reactive/report.md`.

## Pre-registered decision (reactive run)

| co-primary endpoint | mean WM − CV | 97.5% CI (bootstrap, seed 0) | Wilcoxon p (α = 0.025) |
|---|---|---|---|
| at-fault collisions / scene | −0.007 | [−0.027, 0.000] | 0.317 (not significant) |
| close-call share (steps < 1 m) | −4.7 pp | [−8.0, −2.1] pp | 1.0e-4 (significant) |

Progress WM − CV: 95% CI [−0.005, +0.006], not worse by more than 0.02.

**Decision: WM is safer** (the close-call share 97.5% CI lies below 0, and progress is not worse). The at-fault endpoint alone shows no detectable difference: there is 1 at-fault event in total (CV 1, WM 0), far too few to test.

## Results

| | at fault | rear-ended | initial overlap | progress | close-call share | min distance | min TTC | max \|a\| |
|---|---|---|---|---|---|---|---|---|
| CV, replay v4 | 1 | 14 | 3 | 0.921 | 10.7% | 1.63 m | 3.03 s | 3.89 |
| WM, replay v4 | 0 | 16 | 3 | 0.923 | 6.0% | 1.71 m | 3.12 s | 4.15 |
| expert, replay v4 | 1 | 2 | 3 | 1.000 | 8.1% | 1.82 m | 3.42 s | 2.48 |
| CV, reactive | 1 | 2 | 3 | 0.917 | 8.3% | 1.77 m | 3.14 s | 3.68 |
| WM, reactive | 0 | 2 | 3 | 0.917 | 3.6% | 1.87 m | 3.26 s | 3.98 |
| expert, reactive | 1 | 2 | 3 | 1.000 | 8.1% | 1.83 m | 3.42 s | 2.48 |

Counts are over the 146 scenes; the other metrics are means over scenes. Paired WM − CV in replay v4 (secondary): close-call −4.7 pp [−7.6, −2.1] (p = 4e-4), min distance +0.08 m [+0.02, +0.14], progress +0.003 [−0.002, +0.008]. Reactive (secondary): min distance +0.10 m [+0.04, +0.16] (p = 6e-4), progress 0.000 [−0.005, +0.006], rear-ended 0.000.
The only at-fault events: CV scene-0332 (pedestrian, 'no safe plan existed'; risk seen 0.8 s before), and the expert's scene-0914 (pedestrian; a log overlap).

## What changed versus v3 (non-reactive, older footprint)

- At-fault collisions dropped from CV 8 / WM 6 (v3) to CV 1 / WM 0. The covering-circle footprints, which are part of the frozen setup chosen on the dev scenes, account for this.
- Reactive agents remove the rear-ends caused by log replay (CV 14 → 2, WM 16 → 2, against the expert's 2, which are in the log itself).

## Limitations

- **Reactive agents, stage 1:** motorcycles and bicycles (ScenarioNet type CYCLIST) and pedestrians replay. There is no crossing or merge anticipation and no swept ego footprint. Agents brake at the 6 m/s² cap in some scenes (13 of 100 dev100 scenes in check (d)).
- **Few at-fault events:** with 1 at-fault event in 146 scenes, the at-fault endpoint cannot show a difference. The 'safer' decision rests on the close-call co-primary endpoint.
- **Promoted endpoint:** the close-call endpoint was made co-primary after seeing the non-reactive test150_v2 result. This is disclosed in Amendment 1.
- **Model coverage:** 16 predicted slots (about 85% coverage on test); the remaining nearby agents fall back to constant velocity.

