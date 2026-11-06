# Code guide (v2 branch)

A map of the repository for review: what lives where, what each v2 file does, the flags, and how to run things.
v1 is the GATv2-GRU model from the report; v2 is the scene transformer on branch `v2-transformer`.

## 1. Repository structure

```
wm/
├── utils/                      library code (imported, never run directly)
│   ├── lane_tokens.py          v2: map lanes -> 20 m lane-segment tokens
│   ├── dataset_v2.py           v2: processed sample -> fixed-size agent / lane tensors
│   ├── models_v2.py            v2: encoders, scene transformer, decoder, losses, V2Predictor
│   ├── models.py               v1 models (GATv2 + MLP / GRU decoders)
│   ├── graph_utils.py          v1: sample -> PyTorch Geometric graph
│   └── training_utils.py       shared: seeds, manifests, multimodal_metrics, v1 train loop
├── scripts/                    command-line entry points
│   ├── build_processed_dataset.py   nuScenes -> data/processed(_v2)/*.pkl
│   ├── check_lane_tokens.py         plot the lane tokens of one sample (visual check)
│   ├── make_anchors.py              endpoint anchors -> assets/anchors*.npy
│   ├── train_model.py               train / evaluate v1 and v2
│   ├── run_turn_angle_analysis.py   errors by turn angle, top-1 errors, example plots
│   ├── run_interaction_analysis.py  v1 neighbour-removal analysis
│   └── sim_check.py                 Phase D prep: MetaDrive / ScenarioNet smoke tests
├── tools/                      v1 plotting (qualitative figures, the README GIF)
├── assets/                     anchors.npy (k-means), anchors_angle.npy (turn-spread), README GIF
├── report/                     v1 report PDF, v2_training_plan.xlsx (settings + results)
├── docs/CODE_GUIDE.md          this file
├── data/                       local only (git-ignored except LICENSE)
│   ├── v1.0-trainval/          nuScenes metadata tables
│   ├── maps/                   map expansion v1.3 (+ prediction splits)
│   ├── nuscenes_mini/          v1.0-mini tables only (maps -> ../maps); used by the simulator test
│   └── processed_v2/           built v2 datasets: train / val / test .pkl
├── checkpoints/                local only: one folder per experiment, *_best.pth + *.json manifest
├── outputs/                    local only: training logs, plots, turn-bin JSONs, sim/ images
├── sim/                        local only: MetaDrive + ScenarioNet clones, converted scenarios
├── .venv/                      main Python env (training)
└── .venv_sim/                  Python 3.10 env for the simulator (uv)
```

Splits: nuScenes prediction challenge `train` = train (25,711 samples), `train_val` = **val** (6,685),
`val` = **test** (7,242). Settings are chosen on val; test is looked at once, at the end.

## 2. The v2 pipeline in one paragraph

Each sample is one target vehicle at its current time, everything in the **target frame** (target at
the origin, heading along +x). `build_processed_dataset.py --lane-tokens` stores 2 s of history and
6 s of future for every agent, the ego, and the lanes within 50 m cut into ≤ 20 m segments.
`TrajectoryDatasetV2` turns that into fixed-size tensors (64 agent slots, 192 lane slots).
`V2Predictor` encodes every agent and lane segment into one token, runs a transformer over all tokens
with a relative-pose attention bias, and decodes 6 trajectories (modes) plus a probability per mode,
for the target (or, with `--predict-agents 16`, for the target and 15 neighbours).
Training uses winner-takes-all: only the best mode gets the regression loss.

## 3. v2 files, in plain words

### `utils/lane_tokens.py`
- `poses_to_local(poses, origin, yaw)`: world poses -> target-frame `[x, y, cos ψ, sin ψ]`.
- `split_lane(points, seg_len=20)`: cuts a lane polyline (1 m spacing) into segments of ≤ 20 m (21 points);
  neighbouring segments share their end point so they touch.
- `build_lane_tokens(nusc_map, origin, yaw, radius=50, seg_len=20)`: all lanes and lane connectors within
  50 m -> segment points, an "is connector" flag per segment, and successor edges between segments.

### `scripts/build_processed_dataset.py`
Builds the datasets from nuScenes. v1 output is unchanged by default; `--lane-tokens` adds the v2
fields and saves to `data/processed_v2/`. Main pieces: `get_map_for_sample` (cached map per location),
`get_agent_states_for_sample` (per agent: past (5 steps) and future (12 steps) of `[x, y, cos, sin]`
with validity masks, size, category; plus the ego's past and future), `validate_dataset`
(shape / finiteness checks; counts target history gaps instead of failing).
`--workers 6` runs it in parallel (~17 min); each split is saved as soon as it is done.

### `utils/dataset_v2.py`
- `TrajectoryDatasetV2(path, max_agents=64, max_lanes=192)`: one item = a dict of tensors.
  Agent slot order: **0 = target, 1 = ego, then neighbours by distance**, so truncation never drops
  the target or the ego. Keeps the 192 lane segments closest to the target and remaps their edges into
  `lane_adj`. Also returns the ego's planned future (`ego_future`) for the ego-plan token.
- `add_velocity`: appends `[vx, vy]` from consecutive positions (0.5 s apart).
- `agent_type`: nuScenes category -> one of 6 classes (car, truck/bus, pedestrian, cyclist, ego, other).
- `EGO_SIZE`: fixed ego size (Renault Zoe, 1.73 × 4.08 m).

### `utils/models_v2.py`
Encoders (one token each, 128-d):
- `AgentEncoder`: an agent's flattened 2 s history + validity flags, plus type and size embeddings.
- `LaneEncoder`: VectorNet-style: per-point MLP -> max-pool -> concat -> MLP -> max-pool, plus a connector flag.
- `EgoPlanEncoder` (block 2): the ego's planned 6 s future, flattened with validity flags, plus a learned
  "plan" embedding.

Scene encoder:
- `token_poses`: (x, y, heading) per token: agents at their last valid history step, lanes at their middle point.
- `rel_features`: for every token pair, where j is seen from i: `[dx, dy]` in i's frame, cos/sin of the
  heading difference, distance (positions / 50 m); with `--topology` also "j is successor / predecessor of i".
- `SceneEncoder` (+ `EncoderLayer`): 3 pre-LN transformer layers, 8 heads. The relative features go through a
  small MLP into **one scalar bias per head** added to the attention scores (computed once, shared by all layers).
  Padded tokens are masked out.

Decoder:
- `AnchorDecoder` (+ `DecoderLayer`): one query per (agent, mode): `MLP(anchor_k) + mode embedding + agent token`.
  2 layers of: self-attention among the 6 modes of the same agent -> cross-attention to all scene tokens -> FFN.
  Heads output a 12-step path (in 50 m units) and a score per mode.
- `to_target_frame`: paths are predicted in each agent's own frame (origin = its current pose, x = its heading)
  and rotated/translated into the target frame.

Losses:
- `v2_loss` (target only): SmoothL1 on the winning mode + cross-entropy on the scores. The winner `k*` is the
  predicted mode with the lowest ADE (`assign="predicted"`) or the anchor closest to the true endpoint (`"anchor"`).
- `multi_agent_loss`: the same per agent, for agents with ≥ 6 valid future steps, averaged over agents.
- `soft_score_ce` (block 3): score target `p_k = softmax(−ADE_k / 1 m)` instead of one-hot on the winner.

Full model:
- `V2Predictor(anchors, ..., predict_agents=1, ego_plan=False, soft_scores=False, topology=False)`:
  encoders -> scene encoder -> decoder. `forward(batch, plan=None)` returns
  `traj (B, 6, 12, 2), logits (B, 6)` for the target, or `(B, 16, 6, 12, 2), (B, 16, 6)` with
  `predict_agents=16` (target at index 0; ego slot skipped). `.loss(batch, traj, logits)` picks the right loss.
  The ego-plan token is dropped per sample with p = 0.5 in training; `plan=True/False` forces it on/off.
  With every option off, the model is the original target-only v2 (bit-identical outputs).

### `scripts/train_model.py` (v2 parts)
- `evaluate_v2(model, loader, device, plan)`: returns a dict: target minADE₆ / minFDE₆ (same code as v1),
  miss rate (minFDE₆ > 2 m, endpoint definition), top-1 ADE / FDE (highest-score mode), mode-win shares;
  for multi-agent models also all-agent and **moving-agent** (> 1 m from the current position in history or
  future) minADE₆ / minFDE₆ / top-1. Use the moving-agent numbers: parked cars make all-agent errors look easy.
- `agent_min_errors`, `moving_agents`: helpers for the multi-agent metrics (ADE over valid steps, FDE at the
  last valid step).
- `run_v2`: AdamW (lr 1e-3, wd 0.01), 1 epoch linear warmup then cosine to 1e-5 over all epochs, grad clip 1.0,
  dropout 0.1, best-val checkpoint, early stopping. Logs per epoch: losses, lr, val metrics (plan on and off
  with `--ego-plan`), mode wins (flags a mode below 3%). After training: val and test once on the best
  checkpoint, written to the manifest JSON next to the checkpoint.

### `scripts/run_turn_angle_analysis.py`
Errors binned by the absolute endpoint angle (0–15°, 15–30°, 30–60°, 60°+, only endpoints ≥ 5 m away; same
definition as the v1 report). Reads anchors and model options from the checkpoint and its manifest.
`--examples out.png` instead plots 8 val examples (2 per bin) with lanes, agents, ground truth and the 6 modes.
Note: the 60°+ bin also contains reversing agents (endpoint behind the target).

### `scripts/make_anchors.py`
`--method kmeans` (default): k-means (K = 6) on train endpoints -> `assets/anchors.npy`.
`--method angle`: 2 straight anchors + one each for right, sharp right, left, sharp left -> `assets/anchors_angle.npy`.

### `scripts/check_lane_tokens.py`
Builds the lane tokens of one sample directly from the map and plots segments, direction arrows and edges.

### `sim/adapter.py` (Phase D: simulator -> model inputs)
`HistoryBuffer` keeps the last 2 s of sim snapshots (10 Hz); `snapshot_from_scenario` (log) and `snapshot_from_env`
(log for the others + the live MetaDrive ego) fill it. `build_sample(buffer, map_features, target_id, ego_plan)` rebuilds
exactly what the dataset builder stores for that target (agents within 35 m, 5 keyframes of history, lanes in the 50 m box
as 20 m segments + edges, the ego plan as `ego_future`); `to_model_inputs` turns it into the tensors of
`TrajectoryDatasetV2`. `add_connector_polygons` restores lane-connector polygons (ScenarioNet drops them) from the
nuScenes map, which the lane selection needs. `sim_check.py match` checks all of this against the processed dataset.

### `scripts/sim_check.py` (Phase D prep)
`metadrive` (reset + 20 random steps, no rendering), `drive` (constant command for the ego while the others replay),
`match` (adapter vs processed dataset on one mini scene, plus one live MetaDrive replay), `convert` (first 5 nuScenes-mini scenes -> ScenarioNet
scenarios in `sim/scenarios/nuscenes_mini`), `replay` (replays one scene with the logged ego and traffic,
saves a top-down PNG and GIF to `outputs/sim/`).

## 4. v2 flags (`train_model.py --model v2`)

| Flag | Default | What it does |
|---|---|---|
| `--assign {predicted,anchor}` | predicted | Which mode is the winner in the loss: lowest-ADE predicted mode (winner-takes-all) or the anchor closest to the true endpoint. Anchor was clearly worse (val minADE₆ 1.84 vs 1.22). |
| `--predict-agents N` | 1 | 1 = target only. 16 = target + 15 nearest neighbours (ego skipped); adds all-agent and moving-agent metrics. |
| `--ego-plan` | off | Adds the ego's planned future as a scene token; dropped with p = 0.5 in training; val logged with plan on and off; checkpoint and early stopping use plan on. |
| `--soft-scores` | off | Score loss against `softmax(−ADE_k / 1 m)` instead of the one-hot winner. |
| `--topology` | off | Adds successor / predecessor lane flags to the attention bias. Ablation only: no gain on val (seed 42). |
| `--scale-inputs` | off | Divides input positions by 50 m. Not used in any reported run. |
| `--anchors PATH` | assets/anchors.npy | Anchor file used to initialise the mode queries. |
| `--patience N` | 10 | Early stopping on val target minADE₆ (0 = off); the cosine schedule still spans `--epochs`. |
| `--epochs`, `--batch-size`, `--lr`, `--seed` | 30, 64, 1e-3, 42 | Reported runs use 60 epochs, batch 64, lr 1e-3, seeds 42 / 123 / 2026. |
| `--checkpoint-dir`, `--evaluate-only`, `--checkpoint` | | Where to save; evaluate an existing checkpoint without training. |

## 5. How to run

Training env: `.venv` (main). Simulator env: `.venv_sim`. Run everything from the repo root.

```bash
# Build the v2 data (once; ~17 min)
.venv/bin/python scripts/build_processed_dataset.py --lane-tokens --workers 6

# Baseline v2 (target only)
.venv/bin/python scripts/train_model.py --model v2 --epochs 60 --seed 42 --checkpoint-dir checkpoints/v2_predicted

# Blocks 1 + 2 + 3 (multi-agent, ego plan, soft scores)
.venv/bin/python scripts/train_model.py --model v2 --predict-agents 16 --ego-plan --soft-scores \
    --epochs 60 --seed 42 --checkpoint-dir checkpoints/v2_b123

# Evaluate a checkpoint (val + test)
.venv/bin/python scripts/train_model.py --model v2 --evaluate-only --checkpoint checkpoints/v2_predicted/v2_seed42_best.pth

# Turn-angle bins on val, and an example plot
.venv/bin/python scripts/run_turn_angle_analysis.py --model v2 --checkpoint checkpoints/v2_predicted/v2_seed42_best.pth \
    --dataset data/processed_v2/nuscenes_val_prediction_samples_map.pkl --output outputs/turn_bins.json
.venv/bin/python scripts/run_turn_angle_analysis.py --model v2 --checkpoint checkpoints/v2_predicted/v2_seed42_best.pth \
    --dataset data/processed_v2/nuscenes_val_prediction_samples_map.pkl --examples outputs/v2_examples_val.png

# Simulator smoke test (CPU)
.venv_sim/bin/python scripts/sim_check.py metadrive
.venv_sim/bin/python scripts/sim_check.py convert
.venv_sim/bin/python scripts/sim_check.py replay --index 0
```

Without `--dataset`, the turn-angle script uses the **test** split; pass the val file until the final evaluation.
The `--evaluate-only` command prints test numbers too.

Simulator env setup (already done): `uv venv --python 3.10 .venv_sim`, then with `VIRTUAL_ENV=.venv_sim`:
`uv pip install -e sim/metadrive -e sim/scenarionet "nuscenes-devkit>=1.1.10"` and
`.venv_sim/bin/python -m metadrive.pull_asset`. Versions: MetaDrive 0.4.3, ScenarioNet 0.0.1
(needs MetaDrive ≥ 0.4.1.2), nuscenes-devkit 1.2.0, Python 3.10.

## 6. Where results and logs are

| What | Where |
|---|---|
| Settings and results table (val rows; test row filled at the end) | `report/v2_training_plan.xlsx` |
| Training logs, one per run | `outputs/train_v2_<run>_seed<N>.log` |
| Best checkpoint + manifest (settings, best / stopped epoch, val and test metrics) | `checkpoints/<run>/v2_seed<N>_best.pth` and `.json` |
| Turn-angle bins (val) | `outputs/turn_bins_v2_val_seed*.json`, `outputs/turn_bins_v2topo_val_seed42.json` |
| Example predictions | `outputs/v2_examples_val.png` |
| Anchor plots | `outputs/anchors.png`, `outputs/anchors_angle.png` |
| Simulator replay | `outputs/sim/replay_scenario0.png`, `.gif` |
| Dataset build log | `outputs/build_v2.log` |

Runs so far (checkpoint folder: what it is):
`v2_anchor` (anchor assignment, stopped early), `v2_predicted` (baseline, seeds 42 / 123 / 2026),
`v2_topology` (topology ablation), `v2_b123` and `v2_b12` (blocks 1+2+3 and 1+2, seed 42),
`v2_check` and `v2_multi_check` (5-epoch checks).
The manifest JSONs also hold test metrics: don't read them before the final evaluation.

## 7. Things to know

- GPU training is not bit-reproducible: the same seed gives slightly different curves. Report the spread over seeds.
- Miss rate here = minFDE₆ > 2 m at the endpoint (Argoverse style), not nuScenes' max-over-time definition.
- `predict_agents=1` with every flag off reproduces the original v2 outputs exactly, so old checkpoints still load.
