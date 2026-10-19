"""Plot one test sample using the verified final map-aware GRU checkpoint.

Example:
    python tools/plot_qualitative_prediction.py --sample-index 6706
Use --show to open an interactive window without writing files.
"""

import argparse
import json
import pickle
import sys
from pathlib import Path

import numpy as np
import torch
from torch_geometric.data import Batch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.run_turn_angle_analysis import endpoint_geometry, per_sample_metrics
from utils.experiment_utils import read_manifest, sha256, write_json
from utils.graph_utils import sample_to_graph
from utils.models import GraphNetworkAttentionMultimodalWithMapGRU
from utils.training_utils import set_seed


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sample-index", type=int, default=6706)
    parser.add_argument("--checkpoint", type=Path,
                        default=PROJECT_ROOT / "checkpoints_gru_decoder/multimodal_map_gru_seed2026_best.pth")
    parser.add_argument("--output", type=Path, help="New PNG path; default includes checkpoint seed/sample index")
    parser.add_argument("--show", action="store_true", help="Open a plot window instead of saving PNG/JSON")
    args = parser.parse_args()
    manifest = read_manifest(args.checkpoint)
    if (manifest.get("model") != "multimodal_map_gru" or manifest.get("num_modes") != 6
            or manifest.get("num_epochs") != 60 or manifest.get("smoke_test", False)):
        raise ValueError("Expected a fully trained, six-mode final GRU checkpoint")
    seed = manifest["seed"]
    reference = json.loads((PROJECT_ROOT / "doc/results/final_decoder_multiseed_comparison.json").read_text())
    expected_hash = reference["runs"]["gru"][str(seed)]["checkpoint_sha256"]
    checkpoint_hash = sha256(args.checkpoint)
    if checkpoint_hash != expected_hash:
        raise ValueError("Checkpoint differs from the validated matched experiment")
    dataset = PROJECT_ROOT / "data/processed/nuscenes_test_prediction_samples_map.pkl"
    dataset_hash = sha256(dataset)
    if dataset_hash != reference["dataset_sha256"]["test"]:
        raise ValueError("Test dataset differs from the validated experiment")
    if args.output is None:
        args.output = PROJECT_ROOT / f"doc/figures/qualitative_gru_seed{seed}_sample{args.sample_index}.png"
    if args.output.suffix.lower() != ".png":
        parser.error("--output must be a PNG path")
    for path in (() if args.show else (args.output, args.output.with_suffix(".json"))):
        if path.exists():
            raise FileExistsError(f"Refusing to overwrite existing output: {path}")
    with dataset.open("rb") as handle:
        samples = pickle.load(handle)
    if not 0 <= args.sample_index < len(samples):
        parser.error(f"--sample-index must be between 0 and {len(samples)-1}")
    sample = samples[args.sample_index]
    set_seed(seed)
    model = GraphNetworkAttentionMultimodalWithMapGRU(num_modes=6)
    model.load_state_dict(torch.load(args.checkpoint, map_location="cpu", weights_only=True), strict=True)
    model.eval()
    batch = Batch.from_data_list([sample_to_graph(sample)])
    with torch.no_grad():
        prediction = model(batch)
        minade, minfde = per_sample_metrics(prediction, batch.y.reshape(1, 12, 2))
    predictions = prediction[0].numpy()
    future = np.asarray(sample["target_future"])
    past = np.asarray(sample["target_past"])
    if not (np.isfinite(predictions).all() and np.isfinite(future).all()):
        raise ValueError("Non-finite trajectory values")
    errors = np.linalg.norm(predictions - future, axis=-1)
    ades, fdes = errors.mean(axis=-1), errors[:, -1]
    best = int(ades.argmin())
    if not np.allclose([ades.min(), fdes.min()], [minade.item(), minfde.item()], atol=1e-5):
        raise ValueError("Plot metrics disagree with the existing evaluator")

    import matplotlib
    if not args.show:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(10, 9))
    for i, lane in enumerate(sample.get("map_lanes", [])):
        if len(lane):
            ax.plot(lane[:, 0], lane[:, 1], color="lightgray", linewidth=1,
                    alpha=.85, label="Lane centerlines" if i == 0 else None, zorder=1)
    for i, history in enumerate(sample["neighbors_pasts"]):
        ax.scatter(*history[-1], color="black", s=25,
                   label="Neighbor" if i == 0 else None, zorder=3)
    ax.plot(past[:, 0], past[:, 1], "o-", color="blue", linewidth=2.5,
            markersize=4, label="Target past", zorder=5)
    ax.plot(future[:, 0], future[:, 1], "o-", color="forestgreen", linewidth=3,
            markersize=4, label="Ground-truth future", zorder=6)
    colors = ("tab:red", "tab:orange", "tab:purple", "tab:pink", "tab:brown", "tab:cyan")
    for mode, trajectory in enumerate(predictions):
        if mode != best:
            ax.plot(trajectory[:, 0], trajectory[:, 1], "--", color=colors[mode],
                    linewidth=1.4, alpha=.65, label=f"Prediction {mode}", zorder=4)
    ax.plot(predictions[best, :, 0], predictions[best, :, 1], "x-", color="red",
            linewidth=3, markersize=5, label=f"ADE-best prediction (mode {best})", zorder=7)
    ax.scatter(0, 0, color="black", marker="X", s=125, label="Target now", zorder=8)
    points = np.vstack((past, future, predictions.reshape(-1, 2)))
    ax.set_xlim(points[:, 0].min()-5, points[:, 0].max()+5)
    ax.set_ylim(points[:, 1].min()-5, points[:, 1].max()+5)
    ax.set_aspect("equal", adjustable="box")
    ax.set(xlabel="Local x [m]", ylabel="Local y [m]",
           title=f"Sample {args.sample_index} — map-aware GRU, seed {seed}\nBest-mode ADE = {ades[best]:.2f} m")
    ax.grid(alpha=.25)
    ax.legend(loc="best", fontsize=8)
    fig.tight_layout()
    if not args.show:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(args.output, dpi=180)
    displacement, angle = endpoint_geometry(future)
    metadata = {"model": type(model).__name__, "seed": seed,
                "num_epochs": manifest["num_epochs"], "best_epoch": manifest["best_epoch"],
                "checkpoint": str(args.checkpoint), "checkpoint_sha256": checkpoint_hash,
                "dataset": str(dataset), "dataset_sha256": dataset_hash,
                "sample_index": args.sample_index,
                "sample_token": sample["sample_token"],
                "target_instance_token": sample["target_instance_token"],
                "best_ADE_mode": best, "best_FDE_mode": int(fdes.argmin()),
                "minADE6_m": float(ades.min()), "minFDE6_m": float(fdes.min()),
                "per_mode_ADE_m": ades.tolist(), "per_mode_FDE_m": fdes.tolist(),
                "endpoint_displacement_m": displacement, "endpoint_angle_deg": angle,
                "prediction_xy": predictions.tolist(), "ground_truth_xy": future.tolist(),
                "selection": f"Explicit sample index {args.sample_index}; no performance-based selection"}
    if not args.show:
        write_json(args.output.with_suffix(".json"), metadata)
    print(f"Sample {args.sample_index} | GRU seed {seed}, best epoch {manifest['best_epoch']}/60")
    print(f"minADE6={ades.min():.6f} m (mode {best}); minFDE6={fdes.min():.6f} m (mode {fdes.argmin()})")
    if not args.show:
        print(f"Saved {args.output} and provenance JSON")
    if args.show:
        plt.show()
    plt.close(fig)


if __name__ == "__main__":
    main()
