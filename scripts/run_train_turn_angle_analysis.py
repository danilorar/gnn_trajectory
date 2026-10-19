"""Evaluate fitting versus generalization using an existing map-model checkpoint.

Reuses the test analysis's endpoint-angle, per-sample metric and summary
definitions. Train metrics are not compared against the test-set manifest.
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

from scripts.run_turn_angle_analysis import per_sample_metrics, print_summary, sha256, summarize
from scripts.run_turn_angle_map_comparison import angle_masks
from utils.graph_utils import sample_to_graph
from utils.models import GraphNetworkAttentionMultimodalWithMap
from utils.training_utils import evaluate_multimodal_model, set_seed


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path,
                        default=PROJECT_ROOT / "checkpoints_60ep/multimodal_map_seed2026_best.pth")
    parser.add_argument("--test-reference", type=Path,
                        default=PROJECT_ROOT / "doc/results/turn_angle_analysis_seed2026_60ep.json")
    parser.add_argument("--output", type=Path,
                        default=PROJECT_ROOT / "doc/results/train_turn_angle_analysis_seed2026_60ep.json")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    args = parser.parse_args()
    if args.batch_size < 1:
        parser.error("--batch-size must be positive")

    manifest = json.loads(args.checkpoint.with_suffix(".json").read_text())
    reference = json.loads(args.test_reference.read_text())
    checkpoint_hash = sha256(args.checkpoint)
    if checkpoint_hash != reference["checkpoint_sha256"]:
        raise ValueError("Training and saved test evaluations must use the identical checkpoint")
    if manifest.get("model") != "multimodal_map" or manifest.get("num_modes") != 6:
        raise ValueError("Expected a six-mode map-aware checkpoint")
    if not reference["sanity_check"]["manifest_matches"]:
        raise ValueError("Saved test analysis did not pass manifest verification")
    for metric in ("minADE6", "minFDE6"):
        if not np.isclose(reference["full_test"][metric]["mean"],
                          manifest["test_metrics"][metric], atol=.001, rtol=.0001):
            raise ValueError("Saved test metrics disagree with the checkpoint manifest")
    if reference["dataset_sha256"] != sha256(PROJECT_ROOT / "data/processed/nuscenes_test_prediction_samples_map.pkl"):
        raise ValueError("Test dataset has changed since the saved evaluation")
    if reference["minimum_endpoint_displacement_m"] != 5. or reference["std_ddof"] != 0:
        raise ValueError("Saved test evaluation uses a different protocol")

    set_seed(manifest["seed"])
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    device = torch.device(("cuda" if torch.cuda.is_available() else "cpu")
                          if args.device == "auto" else args.device)
    dataset = PROJECT_ROOT / "data/processed/nuscenes_train_prediction_samples_map.pkl"
    with dataset.open("rb") as handle:
        samples = pickle.load(handle)
    valid, moving, masks = angle_masks(samples)
    graphs = []
    for index in np.flatnonzero(valid):
        sample = samples[index]
        if "map_lanes" not in sample:
            raise ValueError(f"Sample {index} has no map context")
        graph = sample_to_graph(sample)
        if not all(torch.isfinite(t).all() for t in (graph.x, graph.edge_attr, graph.map_points)):
            raise ValueError(f"Sample {index} contains non-finite model inputs")
        graphs.append(graph)
    if not graphs:
        raise ValueError("No valid training samples")
    loader = DataLoader(graphs, batch_size=args.batch_size, shuffle=False)
    model = GraphNetworkAttentionMultimodalWithMap(num_modes=6).to(device)
    model.load_state_dict(torch.load(args.checkpoint, map_location=device, weights_only=True))
    model.eval()
    ades, fdes = [], []
    print(f"Evaluating {len(graphs)} training samples on {device}; no optimization", flush=True)
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            ade, fde = per_sample_metrics(model(batch), batch.y.reshape(-1, 12, 2))
            ades.extend(ade.cpu().tolist())
            fdes.extend(fde.cpu().tolist())
    ades, fdes = np.asarray(ades), np.asarray(fdes)
    shared = evaluate_multimodal_model(model, loader, device)
    if not (np.isfinite(ades).all() and np.isfinite(fdes).all() and
            np.allclose([ades.mean(), fdes.mean()], shared, atol=1e-5, rtol=1e-5)):
        raise ValueError("Training metrics disagree with shared evaluator")
    retained = moving[valid]
    bins = {label: summarize(ades[mask[valid]], fdes[mask[valid]]) for label, mask in masks.items()}
    comparisons = {}
    for label, train in bins.items():
        test = reference["turn_bins"][label]
        comparisons[label] = {"n_train": train["n"], "n_test": test["n"]}
        for metric in ("minADE6", "minFDE6"):
            a, b = train[metric]["mean"], test[metric]["mean"]
            comparisons[label][metric] = {
                "train_mean": a, "test_mean": b,
                "test_minus_train_m": b - a if a is not None and b is not None else None,
            }
    result = {
        "seed": manifest["seed"], "num_epochs": manifest["num_epochs"],
        "best_epoch": manifest["best_epoch"], "checkpoint": str(args.checkpoint),
        "checkpoint_sha256": checkpoint_hash, "dataset": str(dataset),
        "dataset_sha256": sha256(dataset), "device": str(device), "batch_size": args.batch_size,
        "torch_version": torch.__version__, "input_samples": len(samples),
        "invalid_future_indices": np.flatnonzero(~valid).tolist(),
        "excluded_displacement_below_5m": int((valid & ~moving).sum()),
        "minimum_endpoint_displacement_m": 5., "std_ddof": 0,
        "angle_definition": reference["angle_definition"],
        "overall_train_all_valid": summarize(ades, fdes),
        "overall_train_retained": summarize(ades[retained], fdes[retained]),
        "overall_test_retained": reference["non_stationary"], "turn_bins": bins,
        "train_vs_test": comparisons, "gap_definition": "test mean minus train mean",
        "sanity_check": {"same_checkpoint_as_test": True, "test_reference": str(args.test_reference),
                         "test_matches_manifest": True, "shared_evaluator_matches": True},
    }
    print_summary("Train retained", result["overall_train_retained"])
    for label, summary in bins.items():
        print_summary(label, summary)
    print("Train/test gaps:", json.dumps(comparisons, indent=2))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(f"Saved {args.output}")


if __name__ == "__main__":
    main()
