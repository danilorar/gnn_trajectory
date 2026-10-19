# GNN trajectory prediction

**When and why does modelling nearby road-user interactions improve future trajectory prediction?**

This nuScenes course project progresses from a constant-velocity reference through a target-history MLP, deterministic GATv2, multimodal GATv2, lane-centerline map context, and a final **map-aware multimodal GATv2 with a GRU displacement decoder**.

Each sample uses **2 s of history** (five positions at 2 Hz) to predict **6 s of future motion** (12 positions). Positions use a target-local frame: the prediction target is at `(0, 0)`, heading along `+x`; it need not be the sensor-equipped ego vehicle. Road users within **35 m** become graph nodes. Each node has **15 features**: ten history coordinates and five category indicators. Selected nodes form a **fully connected directed graph** without explicit self-edges, with edge features `[dx, dy, distance]` (GATv2 adds its own self-loops).

The final model keeps the 64-D agent encoder, two GATv2 layers and pooled lane-centerline map encoder (50 m map radius). Its fused 96-D context and learned 16-D mode embeddings initialize a 64-D, one-layer GRU. It feeds back predicted displacements for 12 steps and cumulatively sums them into **K=6** future trajectories. Training uses best-of-six ADE; evaluation minimizes ADE and FDE independently across modes.

## Final matched decoder result

Seeds 42, 123 and 2026; 60 epochs, standard unweighted sampling, best validation-loss checkpoint. Errors are metres; standard deviations are population values across seeds.

| Map-aware decoder | Test minADE6 | Test minFDE6 |
|---|---:|---:|
| One-shot MLP | 1.532545 ± 0.003414 | 3.121037 ± 0.012009 |
| GRU displacement | **1.474151 ± 0.016075** | **2.930938 ± 0.044113** |

GRU improves overall ADE/FDE by **3.81% / 6.09%**. Gains at 30–60° endpoint angles are **11.62% / 18.08%**, and at 60°+ **6.38% / 7.59%**. Sharp-turn errors remain high. The sharpest bin has 87 samples, and three seeds do not provide scene-level confidence intervals.

See the [canonical comparison](doc/results/final_decoder_multiseed_comparison.md), [machine-readable results and manifests](doc/results/final_decoder_multiseed_comparison.json), [final figure](doc/figures/final_decoder_multiseed_comparison.png), and [diagnostic chain](doc/results/diagnostic_summary.md). The canonical artifacts preserve the pre-cleanup experiment snapshot; use the commands below for the current workflow.

## Setup and data

Use the repository environment files; see [environment setup](instructions/01_environment_setup.md). For example, on Linux:

```bash
conda env create -f conda-environment-files/conda-environment-cpu-unix.yml
conda activate gnn_trajectory
```

The Conda files are starter environments and predate the final experiment runtime (Python 3.12.3, PyTorch 2.14.1, PyG 2.8.0.post1). Original library versions remain recorded in the manifests; cross-version/device reruns need not be bit-identical.

Run commands from the project root. Existing non-map samples are under `data/processed/`. Map-aware training requires these generated files, which are excluded from Git:

```text
data/processed/nuscenes_train_prediction_samples_map.pkl
data/processed/nuscenes_val_prediction_samples_map.pkl
data/processed/nuscenes_test_prediction_samples_map.pkl
```

To prepare them from licensed nuScenes v1.0-trainval metadata and map-expansion assets placed under `data/`:

```bash
python scripts/build_processed_dataset.py
```

This rebuilds the three map datasets at their existing paths; it is unnecessary when those files already exist. Preprocessing remains unchanged: 35 m neighbor radius, 50 m lane-centerline map radius, four past intervals and twelve future intervals. Local `train/val/test` correspond to official prediction splits `train/train_val/val`; local test is not the blind challenge test split.

## Train the final model

```bash
python scripts/train_model.py --model multimodal_map_gru \
  --seed 2026 --epochs 60 --batch-size 64 --lr 0.001 \
  --checkpoint-dir checkpoints_final_gru/seed2026
```

Other supported models are `target_mlp`, `attention`, `multimodal` and `multimodal_map`. The target-only MLP uses the same target history/category features without neighbors. All training uses Adam, uniform shuffling without replacement, and unweighted validation/test loaders. The best validation checkpoint is reloaded for final evaluation, and a manifest is saved beside it. Existing checkpoints are protected against overwrite. Generic defaults remain seed 42 and 30 epochs; the final comparison explicitly uses 60 epochs.

## Evaluate and analyze saved weights

Checkpoint weights are kept locally, not in Git. With the reported seed-2026 GRU checkpoint available:

```bash
python scripts/train_model.py --model multimodal_map_gru --evaluate-only \
  --checkpoint checkpoints_gru_decoder/multimodal_map_gru_seed2026_best.pth

python scripts/run_turn_angle_analysis.py \
  --checkpoint checkpoints_gru_decoder/multimodal_map_gru_seed2026_best.pth

python scripts/run_interaction_analysis.py
```

Evaluation-only mode loads the saved weights and prints validation/test metrics. It uses the checkpoint manifest for the seed when available; no hashes or historical results are required. Turn-angle analysis excludes endpoint displacement below 5 m and uses `[0,15)`, `[15,30)`, `[30,60)`, `[60,180]` degree bins. Endpoint angle measures net future direction, **not curvature**. The matched three-seed comparison is preserved in the result files linked above. For the map-aware MLP baseline, pass its checkpoint to turn-angle analysis; the model is selected from its manifest (or explicitly with `--model multimodal_map`). Both analysis commands print by default; `--output /path/to/new.json` saves a separate record without overwriting existing artifacts.

The interaction analysis compares the saved seed-42 target-only MLP and deterministic GATv2 checkpoints, grouped by neighbor count and nearest-neighbor distance. Positive `MLP ADE − GAT ADE` means GAT performs better; it does not establish a causal effect of neighbors.

For a qualitative view of the final GRU on test sample 6706, including all six modes:

```bash
python tools/plot_qualitative_prediction.py --sample-index 6706 --show
```

`--show` opens a desktop plot without writing files. To save a new plot and prediction JSON, omit `--show` and supply an unused `--output /path/to/new.png`. The [saved seed-2026 example](doc/figures/qualitative_gru_seed2026_sample6706.png) has minADE6 4.376181 m; it illustrates a remaining sharp-turn error, not a performance-selected success case.

## Repository layout

```text
scripts/     build_processed_dataset.py, train_model.py, run_interaction_analysis.py,
             run_turn_angle_analysis.py
scripts/archive/ historical radius sweep (not part of the final workflow)
utils/       models.py, graph_utils.py, training_utils.py, analysis_utils.py
notebooks/   retained exploratory/teaching history; generated outputs cleared
tools/       camera and processed-sample inspection helpers
doc/results/ canonical decoder comparison, diagnostic summary and freeze audit
doc/figures/ final matched decoder figure
```

`scripts/archive/run_radius_sweep.py` remains as the historical reproduction entry point for the [35 m selection](doc/radius_sweep_results.md); no new sweep is needed. Exploratory notebooks retain earlier GINE/radius-graph and historical FDE conventions where labelled. Use scripts for final results. The [freeze audit](doc/results/repository_freeze.md) is historical documentation of the earlier cleanup; normal training and analysis do not depend on it.
