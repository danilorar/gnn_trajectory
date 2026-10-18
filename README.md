# GNN Trajectory Prediction

Interaction-aware vehicle trajectory prediction on the **nuScenes** dataset using Graph Neural Networks.

The main question behind the project is:

> **When and why does modelling nearby road-user interaction improve future trajectory prediction?**

The project is still in development. The current pipeline predicts 6 seconds of future motion from roughly 2 seconds of target and neighbour history, using a target-centred coordinate frame.

## Current approach

The main models currently implemented are:

- **Attention GNN** — deterministic trajectory prediction using two GATv2 layers.
- **Multimodal Attention GNN** — predicts 6 possible future trajectories using best-of-K training.
- **Map context extension** — work in progress; nearby nuScenes lane centre-lines are being added as static road context.

For graph-based models, the target vehicle and nearby road users within **35 m** are represented as nodes. Selected agents are connected with a fully connected directed graph, with relative position and distance used as edge features.

## Data representation

Each prediction sample contains:

```text
target history       5 xy positions  (~2 s)
neighbour histories  5 xy positions per road user
future target        12 xy positions (~6 s)
map lanes            nearby lane centre-line polylines (WIP)
```

All trajectories and map geometry are transformed into the target vehicle's local frame, where the target is at `(0, 0)` and its current heading points along `+x`.

## Project structure

```text
notebooks/   experiments, analysis and visualisation
scripts/     dataset construction and reproducible training entry points
tools/       inspection / visualisation helpers
utils/       graph construction, models, training and analysis utilities
doc/         experiment notes and results
```

## Environment setup

The Python environment is managed with Conda. Full instructions are in [`instructions/01_environment_setup.md`](instructions/01_environment_setup.md).

```bash
conda env create -f conda-environment-files/conda-environment-cpu-<x>.yml
conda activate gnn_trajectory
```

where `<x>` is either `unix` or `win`.

## Training

From the project root:

```bash
python scripts/train_model.py --model attention
python scripts/train_model.py --model multimodal
```

Both models use the processed 35 m datasets under `data/processed/`. Defaults are seed 42, 30 epochs, batch size 64 and Adam with learning rate `0.001`.

A custom run can be started with:

```bash
python scripts/train_model.py \
  --model multimodal \
  --seed 123 \
  --epochs 30 \
  --batch-size 64 \
  --lr 0.001
```

The best validation-loss checkpoint is reloaded for evaluation. A JSON manifest is saved next to each checkpoint with the model configuration, seed, training settings, metrics and library versions.

## Evaluation

The deterministic model reports:

- **ADE** — Average Displacement Error
- **FDE** — Final Displacement Error

The six-mode model reports:

- **minADE6**
- **minFDE6**

where ADE and FDE are minimized independently across the six predicted trajectories.

## Work in progress

Current work is focused on adding **lane/map context** to the multimodal interaction model. The intended comparison is:

```text
Multimodal GATv2
        vs.
Multimodal GATv2 + lane context
```

The next stages include target-only baselines, deeper interaction analysis, qualitative scene comparisons and final model evaluation.
