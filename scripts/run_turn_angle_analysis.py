"""Evaluate the existing map model by ground-truth endpoint angle; no training.

Angles use half-open bins [0,15), [15,30), [30,60), [60,180] degrees.
The manifest covers the full test set, so it is checked against that same
population before interpreting results on the displacement >= 5 m subset.
"""

import argparse
import hashlib
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
from utils.models import GraphNetworkAttentionMultimodalWithMap
from utils.gru_future_decoder import GraphNetworkAttentionMultimodalWithMapGRU
from utils.training_utils import evaluate_multimodal_model, set_seed


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def endpoint_geometry(future):
    """Return displacement in metres and absolute endpoint angle in degrees."""
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
        result[name] = {
            "mean": float(np.mean(values)) if len(values) else None,
            "median": float(np.median(values)) if len(values) else None,
            "std": float(np.std(values)) if len(values) else None,
        }
    return result


def print_summary(label, summary):
    if not summary["n"]:
        print(f"{label}: n=0 (no samples)")
        return
    metrics = " | ".join(
        f"{name} mean={summary[name]['mean']:.6f}, "
        f"median={summary[name]['median']:.6f}, std={summary[name]['std']:.6f} m"
        for name in ("minADE6", "minFDE6")
    )
    print(f"{label}: n={summary['n']} | {metrics}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=123, help="Training seed of the checkpoint (default: 123)")
    parser.add_argument("--checkpoint", type=Path, help="Override the seed-derived checkpoint path")
    parser.add_argument("--dataset", type=Path,
                        default=PROJECT_ROOT / "data/processed/nuscenes_test_prediction_samples_map.pkl")
    parser.add_argument("--output", type=Path, help="Override the seed-derived JSON output path")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--plot", action="store_true", help="Save a mean-minADE6 bar chart")
    args = parser.parse_args()
    if args.batch_size < 1:
        parser.error("--batch-size must be positive")
    if not 0 <= args.seed < 2**32:
        parser.error("--seed must be between 0 and 2**32 - 1")
    if args.checkpoint is None:
        args.checkpoint = PROJECT_ROOT / f"checkpoints/multimodal_map_seed{args.seed}_best.pth"
    if args.output is None:
        args.output = PROJECT_ROOT / f"doc/results/turn_angle_analysis_seed{args.seed}.json"
    manifest_path = args.checkpoint.with_suffix(".json")
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("seed") != args.seed:
        raise ValueError(f"Manifest seed {manifest.get('seed')} differs from --seed {args.seed}")
    expected = manifest.get("test_metrics", {})
    if not all(name in expected for name in ("minADE6", "minFDE6")):
        raise ValueError("Manifest must contain full-test minADE6 and minFDE6")
    set_seed(args.seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    device = torch.device(
        ("cuda" if torch.cuda.is_available() else "cpu") if args.device == "auto" else args.device
    )

    with args.dataset.open("rb") as handle:
        samples = pickle.load(handle)
    graphs, displacements, angles, invalid_indices = [], [], [], []
    for index, sample in enumerate(samples):
        future = np.asarray(sample["target_future"])
        if future.shape != (12, 2) or not np.isfinite(future).all():
            invalid_indices.append(index)
            continue
        # Missing map context is an input error, not a valid map-model experiment.
        if "map_lanes" not in sample:
            raise ValueError(f"Sample {index} has no map_lanes; use the map-aware dataset")
        graph = sample_to_graph(sample)
        if not all(torch.isfinite(value).all() for value in
                   (graph.x, graph.edge_attr, graph.map_points)):
            invalid_indices.append(index)
            continue
        displacement, angle = endpoint_geometry(future)
        graphs.append(graph)
        displacements.append(displacement)
        angles.append(angle)
    if not graphs:
        raise ValueError("No valid test samples")

    loader = DataLoader(graphs, batch_size=args.batch_size, shuffle=False)
    model_classes = {
        "multimodal_map": GraphNetworkAttentionMultimodalWithMap,
        "multimodal_map_gru": GraphNetworkAttentionMultimodalWithMapGRU,
    }
    if manifest.get("model") not in model_classes:
        raise ValueError(f"Unsupported manifest model: {manifest.get('model')}")
    model = model_classes[manifest["model"]](num_modes=6).to(device)
    model.load_state_dict(torch.load(args.checkpoint, map_location=device, weights_only=True))
    model.eval()
    ades, fdes = [], []
    print(f"Evaluating {len(graphs)} valid samples on {device}", flush=True)
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            ade, fde = per_sample_metrics(model(batch), batch.y.reshape(-1, 12, 2))
            ades.extend(ade.cpu().tolist())
            fdes.extend(fde.cpu().tolist())
    ades, fdes = np.asarray(ades), np.asarray(fdes)
    if not (np.isfinite(ades).all() and np.isfinite(fdes).all()):
        raise ValueError("Non-finite model metrics; do not trust turn bins")

    # Independent implementation check against the project's existing evaluator.
    reference = evaluate_multimodal_model(model, loader, device)
    evaluator_matches = bool(np.allclose([ades.mean(), fdes.mean()], reference,
                                         atol=1e-5, rtol=1e-5))
    if not evaluator_matches:
        raise RuntimeError("Per-sample metrics disagree with the shared evaluator")
    differences = {
        name: float(actual - expected[name])
        for name, actual in zip(("minADE6", "minFDE6"), (ades.mean(), fdes.mean()))
        if name in expected
    }
    manifest_matches = bool(len(differences) == 2 and all(
        np.isclose(actual, expected[name], atol=0.001, rtol=0.0001)
        for name, actual in zip(("minADE6", "minFDE6"), (ades.mean(), fdes.mean()))
    ))
    # Stop before producing or interpreting turn-bin results on a mismatch.
    if not manifest_matches:
        print_summary("Full test (manifest population)", summarize(ades, fdes))
        raise RuntimeError(
            f"Seed {args.seed}: full-test metrics do not match manifest. "
            f"Expected={expected}; actual-minus-manifest={differences}. "
            "Shared evaluator agrees with per-sample implementation; inspect "
            "checkpoint/dataset provenance before interpreting bins. No outputs saved."
        )
    moving = np.asarray(displacements) >= 5.0
    angles = np.asarray(angles)
    masks = {
        "0-15 deg": moving & (angles < 15),
        "15-30 deg": moving & (angles >= 15) & (angles < 30),
        "30-60 deg": moving & (angles >= 30) & (angles < 60),
        "60+ deg": moving & (angles >= 60),
    }
    bins = {label: summarize(ades[mask], fdes[mask]) for label, mask in masks.items()}
    result = {
        "model": type(model).__name__, "seed": args.seed,
        "checkpoint": str(args.checkpoint), "checkpoint_sha256": sha256(args.checkpoint),
        "dataset": str(args.dataset), "dataset_sha256": sha256(args.dataset),
        "device": str(device), "batch_size": args.batch_size,
        "torch_version": torch.__version__,
        "angle_definition": "abs(degrees(atan2(y_final, x_final))) in target-local frame",
        "minimum_endpoint_displacement_m": 5.0, "std_ddof": 0,
        "input_samples": len(samples), "invalid_sample_indices": invalid_indices,
        "excluded_displacement_below_5m": int((~moving).sum()),
        "full_test": summarize(ades, fdes),
        "non_stationary": summarize(ades[moving], fdes[moving]), "turn_bins": bins,
        "sanity_check": {
            "shared_evaluator_matches": evaluator_matches,
            "manifest": str(manifest_path), "manifest_num_epochs": manifest.get("num_epochs"),
            "manifest_test_metrics": expected, "full_test_minus_manifest": differences,
            "manifest_matches": manifest_matches,
            "tolerance": {"atol_m": 0.001, "rtol": 0.0001},
            "note": "Manifest comparison uses full test, not displacement-filtered subset. "
                    "Execution stops before turn bins if manifest metrics do not match.",
        },
    }
    print_summary("Full test (manifest population)", result["full_test"])
    print_summary("Displacement >= 5 m", result["non_stationary"])
    print(f"Shared evaluator matches: {evaluator_matches}; manifest matches: {manifest_matches}")
    for label, summary in bins.items():
        print_summary(label, summary)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(f"Saved {args.output}")
    if args.plot:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.bar(list(bins), [s["minADE6"]["mean"] if s["n"] else np.nan for s in bins.values()])
        ax.set(xlabel="Absolute endpoint angle", ylabel="Mean minADE6 (m)",
               title=f"Current map model, seed {args.seed}; displacement ≥ 5 m")
        fig.tight_layout()
        # Keep custom-checkpoint plots separate just like their JSON outputs.
        plot_path = PROJECT_ROOT / "doc/figures" / f"{args.output.stem}.png"
        plot_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(plot_path, dpi=160)
        plt.close(fig)
        print(f"Saved {plot_path}")


if __name__ == "__main__":
    main()
