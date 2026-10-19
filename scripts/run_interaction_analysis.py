"""Compare saved target-only MLP and deterministic GATv2 predictions.

Delta ADE = MLP ADE - GAT ADE; positive means GAT is better. This compares
separately trained models, not a causal intervention removing neighbors.
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

from utils.experiment_utils import read_manifest, sha256, write_json
from utils.graph_utils import sample_to_graph
from utils.models import GraphNetworkAttention, TargetHistoryMLP
from utils.training_utils import set_seed


def ade_for_sample(model, graph, device):
    batch = Batch.from_data_list([graph.to(device)])
    target = batch.y.view(1, 12, 2)
    return torch.norm(model(batch) - target, dim=2).mean().item()


def summarize(values):
    values = np.asarray(values)
    return {"n": int(len(values)),
            "mean_delta_ade_m": float(values.mean()) if len(values) else None,
            "median_delta_ade_m": float(np.median(values)) if len(values) else None,
            "std_delta_ade_m": float(values.std()) if len(values) else None,
            "gat_win_percentage": float(100 * np.mean(values > 0)) if len(values) else None}


def print_bins(title, bins):
    print(title)
    for name, values in bins.items():
        if not values["n"]:
            print(f"{name}: no samples")
            continue
        print(f"{name} | mean ΔADE={values['mean_delta_ade_m']:.6f} m | "
              f"median={values['median_delta_ade_m']:.6f} m | "
              f"std={values['std_delta_ade_m']:.6f} m | "
              f"GAT wins={values['gat_win_percentage']:.1f}% | n={values['n']}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--checkpoint-dir", type=Path, default=PROJECT_ROOT / "checkpoints")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if not 0 <= args.seed < 2**32:
        parser.error("Expected seed in [0, 2**32)")
    if args.output and args.output.exists():
        parser.error(f"Output already exists: {args.output}")
    set_seed(args.seed)
    device = torch.device(("cuda" if torch.cuda.is_available() else "cpu") if args.device == "auto" else args.device)
    dataset = PROJECT_ROOT / "data/processed/nuscenes_test_prediction_samples.pkl"
    with dataset.open("rb") as handle:
        samples = pickle.load(handle)
    graphs = [sample_to_graph(sample) for sample in samples]
    models, manifests, provenance = {}, {}, {}
    for name, cls in (("target_mlp", TargetHistoryMLP), ("attention", GraphNetworkAttention)):
        checkpoint = args.checkpoint_dir / f"{name}_seed{args.seed}_best.pth"
        manifest = read_manifest(checkpoint)
        if manifest["model"] != name or manifest["seed"] != args.seed or manifest["num_modes"] != 1:
            raise ValueError("Expected matching deterministic checkpoint models/seeds")
        model = cls().to(device)
        model.load_state_dict(torch.load(checkpoint, map_location=device, weights_only=True), strict=True)
        models[name] = model.eval()
        manifests[name] = manifest
        provenance[name] = {"checkpoint": str(checkpoint), "checkpoint_sha256": sha256(checkpoint),
                            "manifest": manifest}

    bins = {"0": [], "1-4": [], "5+": []}
    distance_bins = {"0-10 m": [], "10-20 m": [], "20-35 m": []}
    mlp_ades, gat_ades = [], []
    with torch.no_grad():
        for sample, graph in zip(samples, graphs):
            mlp_ade = ade_for_sample(models["target_mlp"], graph, device)
            gat_ade = ade_for_sample(models["attention"], graph, device)
            mlp_ades.append(mlp_ade)
            gat_ades.append(gat_ade)
            delta = mlp_ade - gat_ade
            # ---- Neighbor-count analysis
            count = len(sample["neighbors_pasts"])
            bins["0" if count == 0 else "1-4" if count <= 4 else "5+"].append(delta)
            # ---- Nearest-neighbor distance analysis (zero-neighbor samples excluded)
            if len(sample["neighbors_distances"]):
                distance = min(sample["neighbors_distances"])
                label = "0-10 m" if distance < 10 else "10-20 m" if distance < 20 else "20-35 m"
                distance_bins[label].append(delta)
    overall = {"target_mlp_ADE": float(np.mean(mlp_ades)), "attention_ADE": float(np.mean(gat_ades))}
    for name in models:
        if not np.isclose(overall[f"{name}_ADE"], manifests[name]["test_metrics"]["ADE"], atol=.001, rtol=.0001):
            raise ValueError(f"Test ADE does not reproduce manifest: {name}")
    result = {"seed": args.seed, "dataset_sha256": sha256(dataset), "checkpoints": provenance,
              "delta_definition": "MLP ADE minus GAT ADE; positive means GAT wins; ties are not wins",
              "std_ddof": 0, "manifest_ADE_matches": True, "overall": overall,
              "neighbor_count": {name: summarize(values) for name, values in bins.items()},
              "nearest_neighbor_distance": {name: summarize(values) for name, values in distance_bins.items()}}
    print(result["delta_definition"])
    print_bins("GAT benefit by neighbor count:", result["neighbor_count"])
    print_bins("GAT benefit by nearest-neighbor distance [0,10), [10,20), [20,35] m:", result["nearest_neighbor_distance"])
    if args.output:
        write_json(args.output, result)


if __name__ == "__main__":
    main()
