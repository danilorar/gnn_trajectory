"""
Historical experiment: sweep the neighbor-selection radius (issue #8).
The final radius is fixed at 35 m; this is not part of the main workflow.

Sweep procedure:
regenerate the processed datasets at each candidate radius, train
the attention GNN on each with the same protocol, and evaluate.

Requires raw nuScenes data under --data-dir and torch_geometric;
neither is available in every environment, so this script is meant
to be run wherever both are set up (see issue #8 / PR linked to it).

Usage:
    python scripts/archive/run_radius_sweep.py --radii 20 35 50
"""

import argparse
import json
import pickle
import sys
from pathlib import Path

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from torch_geometric.loader import DataLoader

from utils.graph_utils import sample_to_graph
from utils.models import GraphNetworkAttention
from utils.training_utils import (
    ade_loss,
    train_and_validate,
    evaluate_model,
    set_seed,
    save_run_manifest,
)
from utils.analysis_utils import evaluate_by_density
from utils.graph_utils import remove_neighbors
from torch_geometric.data import Batch

# Historical ablations below retain the original negative-means-help sign.
def neighbor_effect_by_distance(
    model,
    dataset,
    graphs,
    device=None
):
    """
    Legacy radius-sweep convention: normal ADE minus ablated ADE.
    Compare every sample that has a neighbor, paired with the distance
    to that sample's nearest neighbor.

    Returns:
        nearest_distances
        delta_ades

    delta_ADE = normal ADE - ablated ADE

    Negative -> neighbors help
    Positive -> neighbors hurt
    """

    if device is None:
        device = next(model.parameters()).device

    nearest_distances = []
    delta_ades = []

    model.eval()

    with torch.no_grad():

        for sample, graph in zip(dataset, graphs):

            if len(sample["neighbors_distances"]) == 0:
                continue

            nearest_distance = min(sample["neighbors_distances"])

            # Normal graph
            graph_device = graph.to(device)

            batch = Batch.from_data_list(
                [graph_device]
            )

            pred = model(batch)

            target = graph_device.y.view(
                1,
                12,
                2
            )

            normal_errors = torch.norm(
                pred - target,
                dim=2
            )

            normal_ade = (
                normal_errors.mean().item()
            )

            # Ablated graph
            ablated_graph = remove_neighbors(
                graph
            ).to(device)

            ablated_batch = Batch.from_data_list(
                [ablated_graph]
            )

            ablated_pred = model(
                ablated_batch
            )

            ablated_errors = torch.norm(
                ablated_pred - target,
                dim=2
            )

            ablated_ade = (
                ablated_errors.mean().item()
            )

            delta_ade = (
                normal_ade - ablated_ade
            )

            nearest_distances.append(
                nearest_distance
            )

            delta_ades.append(
                delta_ade
            )

    return nearest_distances, delta_ades

# ========================================================

def bin_neighbor_effect_by_distance(
    nearest_distances,
    delta_ades,
    bin_edges=(10, 20, 30, 40, 50)
):
    """
    Group delta ADE values by nearest-neighbor-distance ranges.

    delta_ADE = normal ADE - ablated ADE

    Negative -> neighbors help
    Positive -> neighbors hurt
    """

    nearest_distances = np.asarray(nearest_distances)
    delta_ades = np.asarray(delta_ades)

    edges = [0] + list(bin_edges) + [np.inf]

    summary = {}

    for lo, hi in zip(edges[:-1], edges[1:]):

        label = f"{lo:g}-{hi:g}" if np.isfinite(hi) else f"{lo:g}+"

        mask = (nearest_distances >= lo) & (nearest_distances < hi)
        values = delta_ades[mask]

        summary[label] = {
            "mean_delta_ade": float(values.mean()) if len(values) else None,
            "n": int(mask.sum()),
        }

    return summary


# Official nuScenes split name -> local split name, matching the
# mapping already used in build_processed_dataset.py's __main__:
# "val" here is the development split used for checkpoint selection
# (official "train_val"); "test" is the official held-out "val".
SPLITS = {
    "train": "train",
    "val": "train_val",
    "test": "val",
}


def neighbor_count_stats(dataset):
    """Dataset size and neighbor-count distribution, for the sweep report."""

    counts = np.array([len(s["neighbors_pasts"]) for s in dataset])

    if len(counts) == 0:
        return {"num_samples": 0}

    return {
        "num_samples": len(dataset),
        "mean_neighbors": float(counts.mean()),
        "median_neighbors": float(np.median(counts)),
        "p90_neighbors": float(np.percentile(counts, 90)),
        "p99_neighbors": float(np.percentile(counts, 99)),
        "max_neighbors": int(counts.max()),
    }


def build_or_load_split(
    get_nusc,
    local_name,
    official_split,
    radius,
    past_steps,
    future_steps,
    out_path,
    force,
):
    if out_path.exists() and not force:
        print(f"  {local_name}: found {out_path.name}, loading cached samples")
        with open(out_path, "rb") as f:
            return pickle.load(f)

    # Imported here, not at module level: nuscenes-devkit (and the
    # rest of build_processed_dataset's imports) are only needed when
    # a candidate radius actually has no cached samples yet, so a
    # cache-only run doesn't require raw nuScenes data or a working
    # nuscenes-devkit install at all.
    from nuscenes.eval.prediction.splits import get_prediction_challenge_split
    from scripts.build_processed_dataset import build_prediction_dataset

    nusc = get_nusc()
    targets = get_prediction_challenge_split(official_split, dataroot=nusc.dataroot)

    dataset, skip_reasons = build_prediction_dataset(
        nusc,
        targets,
        radius=radius,
        past_steps=past_steps,
        future_steps=future_steps,
    )

    print(f"  {local_name}: {len(dataset)} samples, skipped {dict(skip_reasons)}")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "wb") as f:
        pickle.dump(dataset, f)

    return dataset


def run_one_radius(get_nusc, radius, args):
    print(f"\n=== radius = {radius:g} m ===")

    out_dir = Path(args.output_dir)

    datasets = {}
    for local_name, official_split in SPLITS.items():
        out_path = out_dir / f"r{radius:g}_{local_name}_prediction_samples.pkl"
        datasets[local_name] = build_or_load_split(
            get_nusc,
            local_name,
            official_split,
            radius,
            args.past_steps,
            args.future_steps,
            out_path,
            args.force,
        )

    dataset_stats = {
        name: neighbor_count_stats(ds) for name, ds in datasets.items()
    }

    train_graphs = [sample_to_graph(s) for s in datasets["train"]]
    val_graphs = [sample_to_graph(s) for s in datasets["val"]]
    test_graphs = [sample_to_graph(s) for s in datasets["test"]]

    train_loader = DataLoader(train_graphs, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_graphs, batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(test_graphs, batch_size=args.batch_size, shuffle=False)

    device = torch.device(
        args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu")
    )

    set_seed(args.seed)
    model = GraphNetworkAttention().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    checkpoint_path = out_dir / f"r{radius:g}_best_attention_gnn_model.pth"

    train_losses, val_losses = train_and_validate(
        model,
        train_loader,
        val_loader,
        optimizer,
        criterion=ade_loss,
        num_epochs=args.epochs,
        checkpoint_path=checkpoint_path,
        device=device,
    )

    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.eval()

    val_ade, val_fde = evaluate_model(model, val_loader, device)
    test_ade, test_fde = evaluate_model(model, test_loader, device)

    save_run_manifest(
        checkpoint_path,
        model="attention",
        seed=args.seed,
        radius=radius,
        num_epochs=args.epochs,
        best_epoch=int(np.argmin(val_losses)) + 1,
        best_val_loss=min(val_losses),
        batch_size=args.batch_size,
        lr=args.lr,
        val_ade=val_ade,
        val_fde=val_fde,
        test_ade=test_ade,
        test_fde=test_fde,
    )

    density_breakdown = evaluate_by_density(model, datasets["val"], val_graphs, device)

    nearest_distances, delta_ades = neighbor_effect_by_distance(
        model, datasets["val"], val_graphs, device
    )
    distance_breakdown = bin_neighbor_effect_by_distance(nearest_distances, delta_ades)

    return {
        "radius": radius,
        "val_ade": val_ade,
        "val_fde": val_fde,
        "test_ade": test_ade,
        "test_fde": test_fde,
        "dataset_stats": dataset_stats,
        "density_breakdown": density_breakdown,
        "distance_breakdown": distance_breakdown,
        "train_losses": train_losses,
        "val_losses": val_losses,
        "checkpoint_path": str(checkpoint_path),
    }


def _json_default(obj):
    """Let json.dump handle the numpy scalars evaluate_model/evaluate_by_density return."""
    if isinstance(obj, np.generic):
        return obj.item()
    raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")


def write_summary(results, out_dir):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    with open(out_dir / "radius_sweep_results.json", "w") as f:
        json.dump(results, f, indent=2, default=_json_default)

    lines = [
        "| Radius | Val ADE | Val FDE | Test ADE | Test FDE | Mean neighbors (val) | Notes |",
        "|---|---:|---:|---:|---:|---:|---|",
    ]

    for r in results:
        mean_neighbors = r["dataset_stats"]["val"].get("mean_neighbors", float("nan"))
        num_val = r["dataset_stats"]["val"].get("num_samples", 0)
        lines.append(
            f"| {r['radius']:g} m | {r['val_ade']:.3f} | {r['val_fde']:.3f} | "
            f"{r['test_ade']:.3f} | {r['test_fde']:.3f} | {mean_neighbors:.2f} | "
            f"{num_val} val samples |"
        )

    table = "\n".join(lines)

    with open(out_dir / "radius_sweep_results.md", "w") as f:
        f.write(table + "\n")

    print("\n" + table)


def main():
    parser = argparse.ArgumentParser(
        description="Sweep the neighbor-selection radius end to end (issue #8)."
    )
    parser.add_argument("--radii", type=float, nargs="+", default=[20.0, 35.0, 50.0])
    parser.add_argument("--data-dir", type=str, default=str(PROJECT_ROOT / "data"))
    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(PROJECT_ROOT / "data" / "processed" / "radius_sweep"),
    )
    parser.add_argument("--version", type=str, default="v1.0-trainval")
    parser.add_argument("--past-steps", type=int, choices=[4], default=4)
    parser.add_argument("--future-steps", type=int, choices=[12], default=12)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument(
        "--force",
        action="store_true",
        help="Regenerate datasets even if cached files already exist",
    )
    args = parser.parse_args()

    _nusc_cache = {}

    def get_nusc():
        if "nusc" not in _nusc_cache:
            from nuscenes.nuscenes import NuScenes

            _nusc_cache["nusc"] = NuScenes(
                version=args.version, dataroot=args.data_dir, verbose=True
            )
        return _nusc_cache["nusc"]

    results = [run_one_radius(get_nusc, radius, args) for radius in args.radii]
    write_summary(results, args.output_dir)


if __name__ == "__main__":
    main()
