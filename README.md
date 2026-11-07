# A trajectory-prediction world model for closed-loop planning (nuScenes)

A scene-transformer world model (v2) that predicts 6 future modes over 6 s for the target and its 15 nearest agents,
conditioned on a candidate ego plan, from 2 s of history and vectorised lane segments on
[nuScenes](https://www.nuscenes.org/nuscenes).

The model is used inside a sampling-MPC planner in [MetaDrive](https://github.com/metadriverse/metadrive)
(scenes converted with [ScenarioNet](https://github.com/metadriverse/scenarionet)). Every 0.5 s, 35 candidate plans
are scored on collision risk under the predicted futures, progress, comfort and lateral offset. The planner is
compared with the same planner using constant-velocity predictions (CV), on 146 held-out nuScenes scenes. It is
tested in two settings: other agents replay the log, or they react to the ego (log-path IDM agents).

Settings were chosen on val and the dev scenes only, and test was looked at once at the end; the analysis was
pre-registered in [docs/EXPERIMENTS.md](docs/EXPERIMENTS.md).

<p align="center">
  <img src="assets/closed_loop_cv_vs_wm.gif" width="960" alt="The same nuScenes scene planned twice: with constant-velocity predictions the ego hits a pedestrian on a personal mobility device; with world-model predictions it picks a plan without collision">
</p>

*The same scene (nuScenes mini scene-1094, a development scene) with reactive agents, planned twice. Every 0.5 s the
planner scores 35 candidate plans (thin blue) against the predicted futures of nearby agents (red: vehicles, orange:
pedestrians; opacity = probability) and drives the cheapest one (thick green).
**Left:** with constant-velocity predictions the conflict shows up only 0.2 s ahead, when every plan is already risky, and the ego hits the rider (5.7 s).
**Right:** the world model predicts the crossing, so the chosen plan avoids it.*

<p align="center">
  <img src="assets/prediction_examples_val.png" width="960" alt="Eight val examples of six predicted modes against the ground truth, two per turn-angle bin">
</p>

*Open-loop predictions on val, 2 examples per turn-angle bin. Green: ground truth; orange: the 6 predicted modes (the
number is the mode probability; solid = top mode); blue: 2 s of history.*

<p align="center">
  <img src="assets/follower_test_val.png" width="960" alt="Follower test: predicted futures of the car behind the ego under the real, braking and accelerating ego plans">
</p>

*The predictions depend on the ego's plan. The car behind the ego is predicted to slow down when the ego plans to brake
(red) and to speed up when the ego plans to accelerate (blue). The reaction is weaker than it should be (see the
limitations).*

## Results

**Open loop (val, target minADE₆):** 1.239 ± 0.006 m over 3 seeds (42 / 123 / 2026; 1.271 ± 0.011 with the ego plan
off).

**Closed loop (146 test scenes, frozen setup):**

| Planner | Agents | At-fault collisions | Rear-ended | Progress | Close-call share (< 1 m) | Min distance |
|---|---|---|---|---|---|---|
| CV | replay (v4) | 1 | 14 | 0.921 | 10.7% | 1.63 m |
| World model | replay (v4) | 0 | 16 | 0.923 | 6.0% | 1.71 m |
| CV | reactive | 1 | 2 | 0.917 | 8.3% | 1.77 m |
| World model | reactive | 0 | 2 | 0.917 | 3.6% | 1.87 m |
| Logged expert | reactive | 1 | 2 | 1.000 | 8.1% | 1.83 m |

**Pre-registered decision (reactive run):**

| Co-primary endpoint | WM − CV | 97.5% CI | Wilcoxon p |
|---|---|---|---|
| At-fault collisions / scene | −0.007 | [−0.027, 0.000] | 0.32 |
| Close-call share | −4.7 pp | [−8.0, −2.1] pp | 1e-4 |

Progress: WM − CV = 0.000, 95% CI [−0.005, +0.006].

The world-model planner is **safer** by the pre-registered rule: the close-call CI lies below 0, and progress is not
worse. With 1 at-fault event in 146 scenes, the at-fault endpoint alone shows no detectable difference.

The full analysis is in `outputs/sim/test150_final_report.md`, and every run is listed in
[docs/EXPERIMENTS.md](docs/EXPERIMENTS.md).

## Limitations

- **Few at-fault events.** With 1 event, the at-fault endpoint cannot show a difference; the "safer" result rests on
  the close-call endpoint.
- **A promoted endpoint.** The close-call endpoint was made co-primary after a non-reactive test run had been seen.
  This is disclosed in Amendment 1 of docs/EXPERIMENTS.md.
- **Limited coverage.** The model predicts 16 agent slots, covering about 85% of the nearby agents on test; the rest
  fall back to constant velocity.
- **Weak reaction to the plan.** When the ego brakes to a stop, the follower's predicted travel drops by only about
  5 m (open-loop follower test).
- **Stage-1 reactive agents:**
  - Motorcycles, bicycles and pedestrians replay.
  - There is no crossing, merge or ego anticipation.
  - Agents react only to leaders that deviate from their logs.
  - Some brake at the 6 m/s² cap.
- **Frozen planner settings.** Cost weights were tuned on 10 mini scenes only.
- **No sensors or perception.** Agents come from the nuScenes annotations; the ego is the Renault Zoe size.

## Reproduce

**1. Data.** Put the nuScenes v1.0-trainval metadata (no sensor blobs needed) and the map expansion v1.3 in `data/`,
with `data/v1.0-trainval/` and `data/maps/expansion/`. For the simulator dev scenes, also add the `v1.0-mini` tables
in `data/nuscenes_mini/v1.0-mini/` and symlink the maps:

```bash
ln -s ../maps data/nuscenes_mini/maps
```

**2. Environments.** `.venv` is for training and open-loop evaluation, `.venv_sim` for the simulator.

```bash
python3.12 -m venv .venv && .venv/bin/pip install -r requirements.txt scikit-learn scipy openpyxl
uv venv --python 3.10 .venv_sim
git clone --depth 1 https://github.com/metadriverse/metadrive.git sim/metadrive
git clone --depth 1 https://github.com/metadriverse/scenarionet.git sim/scenarionet
VIRTUAL_ENV=.venv_sim uv pip install -e sim/metadrive -e sim/scenarionet "nuscenes-devkit>=1.1.10" ijson
VIRTUAL_ENV=.venv_sim uv pip install torch --index-url https://download.pytorch.org/whl/cu132
.venv_sim/bin/python -m metadrive.pull_asset
```

**3. Dataset** (about 20 min). This writes `data/processed_v2/`.

```bash
.venv/bin/python scripts/build_processed_dataset.py --lane-tokens --workers 6
```

**4. Train the world model** (val-based checkpointing; test is only scored with `--eval-test`).

```bash
.venv/bin/python scripts/train_model.py --model v2 --predict-agents 16 --ego-plan --target-split \
    --plan-queries --plan-drop 0.2 --assign predicted --epochs 60 --patience 15 --patience-start 30 \
    --batch-size 64 --lr 0.001 --seed 42 --checkpoint-dir checkpoints/v2_d_rerun
```

**5. Validation evaluation.** `--evaluate-only` needs the same model flags as training.

```bash
.venv/bin/python scripts/train_model.py --model v2 --predict-agents 16 --ego-plan --target-split \
    --plan-queries --plan-drop 0.2 --evaluate-only --checkpoint checkpoints/v2_d_rerun/v2_seed42_best.pth
.venv/bin/python scripts/run_turn_angle_analysis.py --model v2 \
    --checkpoint checkpoints/v2_d_rerun/v2_seed42_best.pth --follower-test outputs/run_d_rerun
```

**6. Closed loop.** First convert the scenes. The mini scenes are the development set; test150 is the 146 test
scenes.

```bash
.venv_sim/bin/python scripts/sim_check.py convert --scenes 10
.venv_sim/bin/python scripts/sim_check.py convert --set test150 --scenes 146
```

Then run the planners: a dev check on the mini scenes, then the final test (CV, world model and expert), with log
replay and with reactive agents.

```bash
.venv_sim/bin/python scripts/sim_check.py evaluate --data sim/scenarios/nuscenes_mini --out outputs/sim/mini_check
.venv_sim/bin/python scripts/sim_check.py evaluate --data sim/scenarios/nuscenes_test150 --out outputs/sim/test150_v4
.venv_sim/bin/python scripts/sim_check.py evaluate --data sim/scenarios/nuscenes_test150 --reactive idm \
    --out outputs/sim/test150_reactive
```

Each run writes a `report.md` with the tables above, the paired statistics, GIFs of the worst scenes and a spot-check
of the collision classification. Run the test commands only for the final evaluation.

## Repository

- `utils/`: dataset, lane tokens and the v2 model (`models_v2.py`).
- `scripts/`: data building, training, open-loop analysis, and the simulator (`sim_check.py`).
- `sim/`: the adapter from simulator state to model inputs, the planner, and closed-loop evaluation with reactive
  agents.

File-by-file details are in [docs/CODE_GUIDE.md](docs/CODE_GUIDE.md). The reactive-agent design is in
[docs/reactive_agents_design.md](docs/reactive_agents_design.md).

The v1 GATv2–GRU model, a final project for SSY340 Deep Machine Learning at Chalmers, is on branch
`feature/trajectory-prediction`.
