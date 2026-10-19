"""Compare saved target-only MLP and deterministic GATv2 predictions.

Delta ADE = MLP ADE - GAT ADE; positive means GAT is better. This compares
separately trained models, not a causal intervention removing neighbors.
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

from utils.graph_utils import sample_to_graph
from utils.models import GraphNetworkAttention, TargetHistoryMLP
from utils.training_utils import set_seed


def ade_for_sample(model, batch):
    target = batch.y.view(1, 12, 2)
    return torch.norm(model(batch) - target, dim=2).mean().item()


def summarize(values):
    if not len(values):
        return {
            "n": 0, "mean_delta_ade_m": None, "median_delta_ade_m": None,
            "std_delta_ade_m": None, "gat_win_percentage": None,
        }
    values = np.asarray(values)
    return {
        "n": len(values),
        "mean_delta_ade_m": float(values.mean()),
        "median_delta_ade_m": float(np.median(values)),
        "std_delta_ade_m": float(values.std()),
        "gat_win_percentage": float(100 * np.mean(values > 0)),
    }


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
    parser.add_argument("--output", type=Path, help="Optional new JSON summary")
    args = parser.parse_args()
    if not 0 <= args.seed < 2**32:
        parser.error("Expected seed in [0, 2**32)")
    if args.output and args.output.exists():
        parser.error(f"Output already exists: {args.output}")
    set_seed(args.seed)
    device = args.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device)

    mlp_checkpoint = args.checkpoint_dir / f"target_mlp_seed{args.seed}_best.pth"
    gat_checkpoint = args.checkpoint_dir / f"attention_seed{args.seed}_best.pth"
    mlp = TargetHistoryMLP().to(device)
    gat = GraphNetworkAttention().to(device)
    mlp.load_state_dict(torch.load(mlp_checkpoint, map_location=device, weights_only=True), strict=True)
    gat.load_state_dict(torch.load(gat_checkpoint, map_location=device, weights_only=True), strict=True)
    mlp.eval()
    gat.eval()

    dataset = PROJECT_ROOT / "data/processed/nuscenes_test_prediction_samples.pkl"
    with dataset.open("rb") as handle:
        samples = pickle.load(handle)
    count_bins = {"0": [], "1-4": [], "5+": []}
    distance_bins = {"0-10 m": [], "10-20 m": [], "20-35 m": []}
    mlp_ades, gat_ades = [], []
    with torch.no_grad():
        for sample in samples:
            graph = sample_to_graph(sample)
            batch = Batch.from_data_list([graph]).to(device)
            mlp_ade = ade_for_sample(mlp, batch)
            gat_ade = ade_for_sample(gat, batch)
            mlp_ades.append(mlp_ade)
            gat_ades.append(gat_ade)
            delta_ade = mlp_ade - gat_ade  # Positive means GAT is better.

            # ---- Neighbor-count analysis
            count = len(sample["neighbors_pasts"])
            if count == 0:
                label = "0"
            elif count <= 4:
                label = "1-4"
            else:
                label = "5+"
            count_bins[label].append(delta_ade)

            # ---- Nearest-neighbor distance analysis (zero-neighbor samples excluded)
            if len(sample["neighbors_distances"]):
                distance = min(sample["neighbors_distances"])
                if distance < 10:
                    label = "0-10 m"
                elif distance < 20:
                    label = "10-20 m"
                else:
                    label = "20-35 m"
                distance_bins[label].append(delta_ade)

    count_summary = {}
    for label, values in count_bins.items():
        count_summary[label] = summarize(values)
    distance_summary = {}
    for label, values in distance_bins.items():
        distance_summary[label] = summarize(values)
    result = {
        "seed": args.seed, "dataset": str(dataset),
        "mlp_checkpoint": str(mlp_checkpoint), "gat_checkpoint": str(gat_checkpoint),
        "delta_definition": "MLP ADE minus GAT ADE; positive means GAT wins; ties are not wins",
        "std_ddof": 0,
        "overall": {"target_mlp_ADE": float(np.mean(mlp_ades)), "attention_ADE": float(np.mean(gat_ades))},
        "neighbor_count": count_summary,
        "nearest_neighbor_distance": distance_summary,
    }
    print(result["delta_definition"])
    print(f"Overall ADE: MLP={result['overall']['target_mlp_ADE']:.6f} m | "
          f"GAT={result['overall']['attention_ADE']:.6f} m")
    print_bins("GAT benefit by neighbor count:", count_summary)
    print_bins("GAT benefit by nearest-neighbor distance [0,10), [10,20), [20,35] m:", distance_summary)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
