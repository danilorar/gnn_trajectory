"""Show or save the six GRU predictions for one test sample.

Example:
    python tools/plot_qualitative_prediction.py --sample-index 6706 --show
"""

import argparse
import pickle
import sys
from pathlib import Path

import numpy as np
import torch
from torch_geometric.data import Batch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from utils.graph_utils import sample_to_graph
from utils.models import GraphNetworkAttentionMultimodalWithMapGRU
from utils.training_utils import read_manifest, set_seed

DATASET = PROJECT_ROOT / "data/processed/nuscenes_test_prediction_samples_map.pkl"
CHECKPOINT = PROJECT_ROOT / "checkpoints_gru_decoder/multimodal_map_gru_seed2026_best.pth"


# ========================================================
# Draw the scene and predicted trajectories
# ========================================================


def plot_prediction(ax, sample, predictions, best_mode):
    past = np.asarray(sample["target_past"])
    future = np.asarray(sample["target_future"])

    # Map context and surrounding agents.
    for index, lane in enumerate(sample.get("map_lanes", [])):
        if len(lane):
            ax.plot(
                lane[:, 0], lane[:, 1],
                color="lightgray", linewidth=1, alpha=0.85,
                label="Lane centerlines" if index == 0 else None, zorder=1,
            )
    for index, history in enumerate(sample["neighbors_pasts"]):
        ax.scatter(
            *history[-1], color="black", s=25,
            label="Neighbor" if index == 0 else None, zorder=3,
        )

    # Observed target motion and ground-truth future.
    ax.plot(
        past[:, 0], past[:, 1], "o-", color="blue", linewidth=2.5,
        markersize=4, label="Target past", zorder=5,
    )
    ax.plot(
        future[:, 0], future[:, 1], "o-", color="forestgreen", linewidth=3,
        markersize=4, label="Ground-truth future", zorder=6,
    )
    ax.scatter(0, 0, color="black", marker="X", s=125, label="Target now", zorder=8)

    # Highlight the mode with the lowest ADE; draw other modes as dashed lines.
    colors = ("tab:red", "tab:orange", "tab:purple", "tab:pink", "tab:brown", "tab:cyan")
    for mode, trajectory in enumerate(predictions):
        if mode != best_mode:
            ax.plot(
                trajectory[:, 0], trajectory[:, 1], "--", color=colors[mode],
                linewidth=1.4, alpha=0.65, label=f"Prediction {mode}", zorder=4,
            )
    best_prediction = predictions[best_mode]
    ax.plot(
        best_prediction[:, 0], best_prediction[:, 1], "x-", color="red",
        linewidth=3, markersize=5,
        label=f"ADE-best prediction (mode {best_mode})", zorder=7,
    )

    # Keep equal metre scales and fit the view around the target trajectories.
    points = np.vstack((past, future, predictions.reshape(-1, 2)))
    ax.set_xlim(points[:, 0].min() - 5, points[:, 0].max() + 5)
    ax.set_ylim(points[:, 1].min() - 5, points[:, 1].max() + 5)
    ax.set_aspect("equal", adjustable="box")
    ax.set(xlabel="Local x [m]", ylabel="Local y [m]")
    ax.grid(alpha=0.25)
    ax.legend(loc="best", fontsize=8)


# ========================================================
# Load one sample, predict, and show or save its figure
# ========================================================


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sample-index", type=int, default=6706)
    parser.add_argument("--checkpoint", type=Path, default=CHECKPOINT)
    parser.add_argument("--output", type=Path, help="PNG path; default includes seed and sample index")
    parser.add_argument("--show", action="store_true", help="Open a window instead of saving a PNG")
    args = parser.parse_args()

    manifest = read_manifest(args.checkpoint)
    seed = manifest["seed"]
    if not args.show:
        if args.output is None:
            args.output = PROJECT_ROOT / f"doc/figures/qualitative_gru_seed{seed}_sample{args.sample_index}.png"
        if args.output.suffix.lower() != ".png":
            parser.error("--output must be a PNG path")
        if args.output.exists():
            raise FileExistsError(f"Refusing to overwrite existing output: {args.output}")

    with DATASET.open("rb") as handle:
        samples = pickle.load(handle)
    if not 0 <= args.sample_index < len(samples):
        parser.error(f"--sample-index must be between 0 and {len(samples) - 1}")
    sample = samples[args.sample_index]

    set_seed(seed)
    model = GraphNetworkAttentionMultimodalWithMapGRU(num_modes=6)
    model.load_state_dict(
        torch.load(args.checkpoint, map_location="cpu", weights_only=True),
        strict=True,
    )
    model.eval()
    batch = Batch.from_data_list([sample_to_graph(sample)])
    with torch.no_grad():
        predictions = model(batch)[0].numpy()

    # ADE averages all future steps; FDE uses only the final step.
    future = np.asarray(sample["target_future"])
    if not (np.isfinite(predictions).all() and np.isfinite(future).all()):
        raise ValueError("Non-finite trajectory values")
    errors = np.linalg.norm(predictions - future, axis=-1)
    ade_per_mode = errors.mean(axis=-1)
    fde_per_mode = errors[:, -1]
    best_ade_mode = int(ade_per_mode.argmin())
    best_fde_mode = int(fde_per_mode.argmin())
    min_ade = ade_per_mode[best_ade_mode]
    min_fde = fde_per_mode[best_fde_mode]

    import matplotlib
    if not args.show:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(10, 9))
    plot_prediction(ax, sample, predictions, best_ade_mode)
    ax.set_title(
        f"Sample {args.sample_index} — map-aware GRU, seed {seed}\n"
        f"Best-mode ADE = {min_ade:.2f} m"
    )
    fig.tight_layout()
    print(f"Sample {args.sample_index} | GRU seed {seed}")
    print(f"minADE6={min_ade:.6f} m (mode {best_ade_mode}); "
          f"minFDE6={min_fde:.6f} m (mode {best_fde_mode})")

    if args.show:
        plt.show()
    else:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(args.output, dpi=180)
        print(f"Saved {args.output}")
    plt.close(fig)


if __name__ == "__main__":
    main()
