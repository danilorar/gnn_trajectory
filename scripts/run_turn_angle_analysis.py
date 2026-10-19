"""Evaluate a map-aware model by absolute future endpoint angle.

Bins include endpoint displacement >=5 m. ADE and FDE independently
select their best trajectory mode. This script never trains a model.
"""

import argparse
import json
import pickle
import sys
from pathlib import Path

import numpy as np
import torch
from torch_geometric.loader import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from utils.graph_utils import sample_to_graph
from utils.models import (
    GraphNetworkAttentionMultimodalWithMap,
    GraphNetworkAttentionMultimodalWithMapGRU,
)
from utils.training_utils import read_manifest, set_seed

DATASET = PROJECT_ROOT / "data/processed/nuscenes_test_prediction_samples_map.pkl"
LABELS = ("0-15 deg", "15-30 deg", "30-60 deg", "60+ deg")


def endpoint_geometry(future):
    """Return final displacement in metres and absolute endpoint angle in degrees."""
    x, y = np.asarray(future, dtype=np.float64)[-1]
    return float(np.hypot(x, y)), float(abs(np.degrees(np.arctan2(y, x))))


def per_sample_metrics(predictions, targets):
    """Independently select the ADE-best and FDE-best of six trajectories."""
    if predictions.shape != (targets.shape[0], 6, 12, 2):
        raise ValueError(f"Unexpected prediction shape: {predictions.shape}")
    errors = torch.linalg.vector_norm(predictions - targets[:, None], dim=-1)
    return errors.mean(-1).min(-1).values, errors[:, :, -1].min(-1).values


def summarize(ades, fdes):
    result = {"n": int(len(ades))}
    for name, values in (("minADE6", ades), ("minFDE6", fdes)):
        if len(values):
            result[name] = {
                "mean": float(np.mean(values)),
                "median": float(np.median(values)),
                "std": float(np.std(values)),
            }
        else:
            result[name] = {"mean": None, "median": None, "std": None}
    return result


def print_summary(label, summary):
    print(f"{label}: n={summary['n']}")
    if summary["n"]:
        for name in ("minADE6", "minFDE6"):
            stats = summary[name]
            print(f"  {name}: mean={stats['mean']:.6f}, median={stats['median']:.6f}, "
                  f"std={stats['std']:.6f} m")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, help="Default: saved checkpoint seed, or 2026")
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--model", choices=("multimodal_map", "multimodal_map_gru"),
                        help="Default: saved model name, or multimodal_map_gru")
    parser.add_argument("--dataset", type=Path, default=DATASET)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--output", type=Path, help="Optional new JSON summary")
    parser.add_argument("--plot", action="store_true", help="Save a bar chart beside --output")
    args = parser.parse_args()
    if args.batch_size < 1:
        parser.error("--batch-size must be positive")
    if args.plot and not args.output:
        parser.error("--plot requires --output")
    if args.output and args.output.exists():
        parser.error(f"Output already exists: {args.output}")
    if args.plot and args.output.with_suffix(".png").exists():
        parser.error("Plot output already exists")

    seed = 2026 if args.seed is None else args.seed
    if args.checkpoint is None:
        folder = PROJECT_ROOT / "checkpoints_gru_decoder"
        if seed != 2026:
            folder = folder / f"seed{seed}"
        args.checkpoint = folder / f"multimodal_map_gru_seed{seed}_best.pth"
    manifest = {}
    if args.checkpoint.with_suffix(".json").exists():
        manifest = read_manifest(args.checkpoint)
    if args.seed is None:
        seed = manifest.get("seed", seed)
    if not 0 <= seed < 2**32:
        parser.error("--seed must be between 0 and 2**32 - 1")
    model_name = args.model or manifest.get("model", "multimodal_map_gru")
    set_seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    device = args.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device)

    with args.dataset.open("rb") as handle:
        samples = pickle.load(handle)
    graphs, displacements, angles, invalid = [], [], [], []
    for index, sample in enumerate(samples):
        future = np.asarray(sample["target_future"])
        if future.shape != (12, 2) or not np.isfinite(future).all():
            invalid.append(index)
            continue
        if "map_lanes" not in sample:
            raise ValueError(f"Sample {index} has no map_lanes; use a map-aware dataset")
        graph = sample_to_graph(sample)
        if not all(torch.isfinite(t).all() for t in (graph.x, graph.edge_attr, graph.map_points)):
            invalid.append(index)
            continue
        displacement, angle = endpoint_geometry(future)
        graphs.append(graph)
        displacements.append(displacement)
        angles.append(angle)
    if not graphs:
        raise ValueError("No valid samples")
    loader = DataLoader(graphs, batch_size=args.batch_size, shuffle=False)

    if model_name == "multimodal_map":
        model = GraphNetworkAttentionMultimodalWithMap(num_modes=6)
    elif model_name == "multimodal_map_gru":
        model = GraphNetworkAttentionMultimodalWithMapGRU(num_modes=6)
    else:
        parser.error("Expected a map-aware MLP or GRU checkpoint")
    model = model.to(device)
    model.load_state_dict(torch.load(args.checkpoint, map_location=device, weights_only=True), strict=True)
    model.eval()
    ades, fdes = [], []
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            ade, fde = per_sample_metrics(model(batch), batch.y.reshape(-1, 12, 2))
            ades.extend(ade.cpu().tolist())
            fdes.extend(fde.cpu().tolist())
    ades = np.asarray(ades)
    fdes = np.asarray(fdes)
    if not (np.isfinite(ades).all() and np.isfinite(fdes).all()):
        raise ValueError("Non-finite prediction errors")

    moving = np.asarray(displacements) >= 5.0
    angles = np.asarray(angles)
    masks = (
        moving & (angles < 15),
        moving & (angles >= 15) & (angles < 30),
        moving & (angles >= 30) & (angles < 60),
        moving & (angles >= 60),
    )
    bins = {}
    for label, mask in zip(LABELS, masks):
        bins[label] = summarize(ades[mask], fdes[mask])
    result = {
        "model": model_name, "seed": seed, "checkpoint": str(args.checkpoint),
        "dataset": str(args.dataset), "batch_size": args.batch_size,
        "angle_definition": "abs(degrees(atan2(y_final, x_final))) in target-local frame",
        "minimum_endpoint_displacement_m": 5.0, "std_ddof": 0,
        "input_samples": len(samples), "invalid_sample_indices": invalid,
        "excluded_displacement_below_5m": int((~moving).sum()),
        "full_test": summarize(ades, fdes),
        "non_stationary": summarize(ades[moving], fdes[moving]),
        "turn_bins": bins,
    }
    print(f"Model={model_name} | seed={seed} | skipped invalid samples={len(invalid)}")
    print_summary("Full dataset", result["full_test"])
    print_summary("Displacement >=5 m", result["non_stationary"])
    for label, summary in bins.items():
        print_summary(label, summary)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
        print(f"Saved {args.output}")
    if args.plot:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        means = [bins[label]["minADE6"]["mean"] for label in LABELS]
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.bar(LABELS, [np.nan if value is None else value for value in means])
        ax.set(xlabel="Absolute endpoint angle", ylabel="Mean minADE6 (m)",
               title=f"Seed {seed}; displacement ≥5 m")
        fig.tight_layout()
        fig.savefig(args.output.with_suffix(".png"), dpi=160)
        plt.close(fig)


if __name__ == "__main__":
    main()
