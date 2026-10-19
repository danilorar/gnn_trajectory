"""Evaluate a saved map-aware checkpoint by absolute future endpoint angle.

Only displacement >=5 m enters the bins. Full-test metrics must reproduce
its manifest before interpreting bins. This script never trains a model.
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

from utils.experiment_utils import read_manifest, sha256, write_json
from utils.graph_utils import sample_to_graph
from utils.models import GraphNetworkAttentionMultimodalWithMap, GraphNetworkAttentionMultimodalWithMapGRU
from utils.training_utils import evaluate_multimodal_model, set_seed

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
        result[name] = {"mean": float(np.mean(values)) if len(values) else None,
                        "median": float(np.median(values)) if len(values) else None,
                        "std": float(np.std(values)) if len(values) else None}
    return result


def print_summary(label, summary):
    if not summary["n"]:
        print(f"{label}: n=0 (no samples)")
        return
    metrics = " | ".join(
        f"{name} mean={summary[name]['mean']:.6f}, median={summary[name]['median']:.6f}, "
        f"std={summary[name]['std']:.6f} m" for name in ("minADE6", "minFDE6")
    )
    print(f"{label}: n={summary['n']} | {metrics}")


def load_turn_dataset(path=DATASET, batch_size=64):
    with Path(path).open("rb") as handle:
        samples = pickle.load(handle)
    graphs, displacements, angles, invalid = [], [], [], []
    for index, sample in enumerate(samples):
        future = np.asarray(sample["target_future"])
        if future.shape != (12, 2) or not np.isfinite(future).all():
            invalid.append(index)
            continue
        if "map_lanes" not in sample:
            raise ValueError(f"Sample {index} has no map_lanes; use the map-aware dataset")
        graph = sample_to_graph(sample)
        if not all(torch.isfinite(t).all() for t in (graph.x, graph.edge_attr, graph.map_points)):
            invalid.append(index)
            continue
        distance, angle = endpoint_geometry(future)
        graphs.append(graph)
        displacements.append(distance)
        angles.append(angle)
    if not graphs:
        raise ValueError("No valid test samples")
    return {"loader": DataLoader(graphs, batch_size=batch_size, shuffle=False),
            "displacements": np.asarray(displacements), "angles": np.asarray(angles),
            "input_samples": len(samples), "invalid_sample_indices": invalid,
            "dataset": str(path), "dataset_sha256": sha256(path)}


def analyze_checkpoint(checkpoint, seed=None, device="cpu", data=None, batch_size=64):
    checkpoint = Path(checkpoint)
    manifest = read_manifest(checkpoint)
    if seed is not None and manifest["seed"] != seed:
        raise ValueError("Requested seed differs from the manifest")
    seed = manifest["seed"]
    if not 0 <= seed < 2**32 or manifest.get("num_modes") != 6:
        raise ValueError("Expected a valid seed and six-mode checkpoint")
    classes = {"multimodal_map": GraphNetworkAttentionMultimodalWithMap,
               "multimodal_map_gru": GraphNetworkAttentionMultimodalWithMapGRU}
    if manifest.get("model") not in classes:
        raise ValueError("Expected a map-aware multimodal checkpoint")
    data = load_turn_dataset(batch_size=batch_size) if data is None else data
    if manifest.get("dataset_sha256", {}).get("test", data["dataset_sha256"]) != data["dataset_sha256"]:
        raise ValueError("Test dataset differs from the manifest")
    set_seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    device = torch.device(device)
    loader = data["loader"]
    model = classes[manifest["model"]](num_modes=6).to(device)
    model.load_state_dict(torch.load(checkpoint, map_location=device, weights_only=True), strict=True)
    model.eval()
    ades, fdes = [], []
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            ade, fde = per_sample_metrics(model(batch), batch.y.reshape(-1, 12, 2))
            ades.extend(ade.cpu().tolist())
            fdes.extend(fde.cpu().tolist())
    ades, fdes = np.asarray(ades), np.asarray(fdes)
    shared = evaluate_multimodal_model(model, loader, device)
    if not (np.isfinite(ades).all() and np.isfinite(fdes).all()
            and np.allclose([ades.mean(), fdes.mean()], shared, atol=1e-5, rtol=1e-5)):
        raise ValueError("Non-finite metrics or disagreement with the shared evaluator")
    expected = manifest["test_metrics"]
    names = ("minADE6", "minFDE6")
    differences = {name: float(actual - expected[name])
                   for name, actual in zip(names, (ades.mean(), fdes.mean()))}
    if not all(np.isclose(actual, expected[name], atol=.001, rtol=.0001)
               for name, actual in zip(names, (ades.mean(), fdes.mean()))):
        raise ValueError(f"Full-test metrics differ from manifest; do not trust turn bins: {differences}")
    moving = data["displacements"] >= 5.
    angles = data["angles"]
    masks = (moving & (angles < 15), moving & (angles >= 15) & (angles < 30),
             moving & (angles >= 30) & (angles < 60), moving & (angles >= 60))
    return {"model": type(model).__name__, "seed": seed, "checkpoint": str(checkpoint),
            "checkpoint_sha256": sha256(checkpoint), "dataset": data["dataset"],
            "dataset_sha256": data["dataset_sha256"], "device": str(device),
            "batch_size": loader.batch_size, "torch_version": torch.__version__,
            "angle_definition": "abs(degrees(atan2(y_final, x_final))) in target-local frame",
            "minimum_endpoint_displacement_m": 5., "std_ddof": 0,
            "input_samples": data["input_samples"], "invalid_sample_indices": data["invalid_sample_indices"],
            "excluded_displacement_below_5m": int((~moving).sum()),
            "full_test": summarize(ades, fdes), "non_stationary": summarize(ades[moving], fdes[moving]),
            "turn_bins": {label: summarize(ades[mask], fdes[mask]) for label, mask in zip(LABELS, masks)},
            "shared_full_test_metrics": dict(zip(names, map(float, shared))),
            "sanity_check": {"shared_evaluator_matches": True, "manifest_matches": True,
                             "manifest_num_epochs": manifest["num_epochs"],
                             "manifest_test_metrics": expected, "full_test_minus_manifest": differences}}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, help="Checkpoint seed; default 2026 if no explicit checkpoint")
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--dataset", type=Path, default=DATASET)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--output", type=Path, help="Optional new JSON path; existing files are protected")
    parser.add_argument("--plot", action="store_true", help="Save a chart beside --output")
    args = parser.parse_args()
    if args.batch_size < 1 or (args.seed is not None and not 0 <= args.seed < 2**32):
        parser.error("Expected a positive batch size and seed in [0, 2**32)")
    if args.plot and not args.output:
        parser.error("--plot requires --output")
    for path in ([args.output, args.output.with_suffix('.png')] if args.plot else [args.output]):
        if path and path.exists():
            parser.error(f"Output already exists: {path}")
    if args.checkpoint is None:
        seed = args.seed if args.seed is not None else 2026
        folder = PROJECT_ROOT / "checkpoints_gru_decoder"
        if seed != 2026:
            folder /= f"seed{seed}"
        args.checkpoint = folder / f"multimodal_map_gru_seed{seed}_best.pth"
    device = ("cuda" if torch.cuda.is_available() else "cpu") if args.device == "auto" else args.device
    data = load_turn_dataset(args.dataset, args.batch_size)
    result = analyze_checkpoint(args.checkpoint, args.seed, device, data)
    print_summary("Full test", result["full_test"])
    print_summary("Displacement >=5 m", result["non_stationary"])
    print("Manifest and shared-evaluator checks passed.")
    for label, summary in result["turn_bins"].items():
        print_summary(label, summary)
    if args.output:
        write_json(args.output, result)
        print(f"Saved {args.output}")
    if args.plot:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.bar(LABELS, [result["turn_bins"][label]["minADE6"]["mean"] for label in LABELS])
        ax.set(xlabel="Absolute endpoint angle", ylabel="Mean minADE6 (m)", title=f"Seed {result['seed']}; displacement ≥5 m")
        fig.tight_layout()
        fig.savefig(args.output.with_suffix(".png"), dpi=160)
        plt.close(fig)


if __name__ == "__main__":
    main()
