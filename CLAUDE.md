# CLAUDE.md

## Project
Trajectory prediction on nuScenes, used as an object-level world model for closed-loop
sampling-MPC planning (MetaDrive + ScenarioNet). v1 = GNN (branch feature/trajectory-prediction),
v2 = scene transformer (branch v2-transformer). See docs/CODE_GUIDE.md for file-by-file details.

## Environments
- `.venv` (Python 3.12, torch 2.15 cu132): data building, training, open-loop eval.
- `.venv_sim` (Python 3.10, torch 2.14 cu132): MetaDrive/ScenarioNet, closed-loop planning.
- RunPod pod (torch 2.8): long training runs only, via the pod agent in ~/Desktop/wm-pod.

## Key commands
- Train v2: `python scripts/train_model.py --model v2 ...` (see CODE_GUIDE for flags)
- Turn bins / CV baseline: `python scripts/run_turn_angle_analysis.py --model v2 ...`
- Simulator: `.venv_sim/bin/python scripts/sim_check.py {match|follow|plan|evaluate} ...`

## Data splits — never break these
- train = model training; val = model selection and all tuning; test = final numbers only.
- Never evaluate on the test split (open or closed loop) unless the user explicitly asks.
- Closed-loop test set = sim/scenarios/nuscenes_test150 (146 scenes). The 10 mini scenes are
  the planner development set; cost weights are tuned only there and then frozen.

## Rules
- Never commit or upload nuScenes data, images or processed samples (data/ is git-ignored).
- Ask before: deleting files, downloads > 5 GB, anything that costs money (pods, APIs).
- Launch every training run with `systemd-inhibit` (the laptop must not sleep mid-run).
- Run GPU jobs on the laptop sequentially, one at a time.
- Commit after each completed step, with a message that says what changed and why.
- Keep long commands short and with a timeout; run long jobs in the background with a log.
- When a result changes a decision, report it before continuing.

## Known pitfalls
- In log replay, other agents do not react: report rear-end collisions separately from at-fault ones.
- ScenarioNet drops lane-connector polygons: the adapter restores them (add_connector_polygons).
- The model predicts 16 agent slots; extra slots fall back to constant velocity in the planner.
- Accelerations from interpolated logs are noisy: compute them from 2 Hz positions.