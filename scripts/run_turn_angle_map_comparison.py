"""Compare trained map/no-map checkpoints and measure training angle frequency.

No training or preprocessing changes. If canonical no-map checkpoints are
absent, use the archived multi-seed checkpoints and verify their saved ADE
and FDE-at-best-ADE results without relabeling that FDE as minFDE6.
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

from scripts.run_turn_angle_analysis import endpoint_geometry, per_sample_metrics, sha256, summarize
from utils.graph_utils import sample_to_graph
from utils.models import GraphNetworkAttentionMultimodal, GraphNetworkAttentionMultimodalWithMap
from utils.training_utils import evaluate_multimodal_model, set_seed

LABELS = ("0-15 deg", "15-30 deg", "30-60 deg", "60+ deg")


def load_samples(path):
    with path.open("rb") as handle:
        return pickle.load(handle)


def angle_masks(samples):
    future = [np.asarray(s["target_future"]) for s in samples]
    valid = np.array([f.shape == (12, 2) and np.isfinite(f).all() for f in future])
    geometry = [endpoint_geometry(f) if ok else (np.nan, np.nan)
                for f, ok in zip(future, valid)]
    displacement, angle = np.asarray(geometry).T
    moving = valid & (displacement >= 5.)
    return valid, moving, {
        LABELS[0]: moving & (angle < 15),
        LABELS[1]: moving & (angle >= 15) & (angle < 30),
        LABELS[2]: moving & (angle >= 30) & (angle < 60),
        LABELS[3]: moving & (angle >= 60),
    }


def write_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, allow_nan=False) + "\n")


def train_distribution(plot):
    path = PROJECT_ROOT / "data/processed/nuscenes_train_prediction_samples.pkl"
    samples = load_samples(path)
    valid, moving, masks = angle_masks(samples)
    retained = int(moving.sum())
    if not retained:
        raise ValueError("No valid non-stationary training samples")
    bins = {label: {"n": int(mask.sum()),
                    "percentage_of_non_stationary": float(mask.sum() * 100 / retained)}
            for label, mask in masks.items()}
    result = {"dataset": str(path), "dataset_sha256": sha256(path),
              "total_train_samples": len(samples), "invalid_samples": int((~valid).sum()),
              "excluded_displacement_below_5m": int((valid & ~moving).sum()),
              "retained": retained, "minimum_endpoint_displacement_m": 5., "bins": bins}
    write_json(PROJECT_ROOT / "doc/results/train_turn_angle_distribution.json", result)
    print("Training distribution:", json.dumps(result, indent=2), flush=True)
    if plot:
        plt = plotting()
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.bar(LABELS, [bins[k]["percentage_of_non_stationary"] for k in LABELS])
        ax.set(xlabel="Absolute endpoint angle", ylabel="Non-stationary training samples (%)",
               title=f"Training angle distribution (n={retained:,})")
        save_figure(fig, "train_turn_angle_distribution.png", plt)


def identity(sample):
    return sample["sample_token"], sample["target_instance_token"]


def align_samples(no_map, with_map):
    ids = [identity(s) for s in no_map]
    map_ids = [identity(s) for s in with_map]
    if len(set(ids)) != len(ids) or len(set(map_ids)) != len(map_ids):
        raise ValueError("Duplicate sample/target identities")
    if set(ids) != set(map_ids):
        raise ValueError("Test datasets have different sample identities; comparison stopped")
    lookup = dict(zip(map_ids, with_map))
    aligned = [lookup[key] for key in ids]
    for a, b in zip(no_map, aligned):
        for field in ("target_past", "target_future"):
            if not np.array_equal(a[field], b[field]):
                raise ValueError(f"Different {field} for {identity(a)}")
        if a["target_category"] != b["target_category"]:
            raise ValueError("Target category differs")
        if a["neighbor_instance_tokens"] != b["neighbor_instance_tokens"]:
            raise ValueError("Neighbor identity/order differs")
        if len(a["neighbors_pasts"]) != len(b["neighbors_pasts"]) or any(
            not np.array_equal(x, y) for x, y in zip(a["neighbors_pasts"], b["neighbors_pasts"])
        ):
            raise ValueError("Neighbor histories differ")
        if "map_lanes" not in b:
            raise ValueError("Map sample has no map_lanes")
    return aligned, {"n": len(ids), "original_order_matches": ids == map_ids,
                     "identity_fields": ["sample_token", "target_instance_token"],
                     "identical_target_pasts_futures_and_agent_inputs": True}


def checkpoint_info(seed, with_map):
    name = "multimodal_map" if with_map else "multimodal"
    checkpoint = PROJECT_ROOT / f"checkpoints/{name}_seed{seed}_best.pth"
    if checkpoint.exists():
        source = checkpoint.with_suffix(".json")
        manifest = json.loads(source.read_text())
        if (manifest.get("num_epochs", 0) < 30 or manifest.get("seed") != seed
                or manifest.get("model") != name or manifest.get("num_modes") != 6):
            raise ValueError(f"Not a verified fully-trained {name} checkpoint: {source}")
        return checkpoint, source, manifest["test_metrics"], "manifest"
    if with_map:
        raise FileNotFoundError(checkpoint)
    source = PROJECT_ROOT / "data/processed/multimodal_seed_check/results.json"
    row = next(r for r in json.loads(source.read_text()) if r["seed"] == seed)
    if len(row["train_losses"]) != 30 or len(row["val_losses"]) != 30:
        raise ValueError("Archived no-map checkpoint is not a 30-epoch run")
    checkpoint = source.parent / f"seed{seed}_best_model.pth"
    if not checkpoint.exists():
        raise FileNotFoundError(checkpoint)
    expected = {"minADE6": row["test_minade6"],
                "FDE_at_best_ADE": row["test_fde_at_best_ade"]}
    return checkpoint, source, expected, "archived_30_epoch_run_report"


def evaluate(seed, with_map, graphs, batch_size, device):
    checkpoint, source, expected, source_type = checkpoint_info(seed, with_map)
    cls = GraphNetworkAttentionMultimodalWithMap if with_map else GraphNetworkAttentionMultimodal
    set_seed(seed)
    model = cls(num_modes=6).to(device)
    model.load_state_dict(torch.load(checkpoint, map_location=device, weights_only=True))
    model.eval()
    loader = DataLoader(graphs, batch_size=batch_size, shuffle=False)
    ades, fdes, legacy_fdes = [], [], []
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            pred = model(batch)
            targets = batch.y.reshape(-1, 12, 2)
            ade, fde = per_sample_metrics(pred, targets)
            errors = torch.linalg.vector_norm(pred - targets[:, None], dim=-1)
            best_ade_mode = errors.mean(-1).argmin(-1)
            legacy = errors[torch.arange(batch.num_graphs, device=device), best_ade_mode, -1]
            ades.extend(ade.cpu().tolist())
            fdes.extend(fde.cpu().tolist())
            legacy_fdes.extend(legacy.cpu().tolist())
    ades, fdes = np.asarray(ades), np.asarray(fdes)
    reference = evaluate_multimodal_model(model, loader, device)
    actual = {"minADE6": float(ades.mean()), "minFDE6": float(fdes.mean()),
              "FDE_at_best_ADE": float(np.mean(legacy_fdes))}
    if not (np.isfinite(ades).all() and np.isfinite(fdes).all() and
            np.allclose([ades.mean(), fdes.mean()], reference, atol=1e-5, rtol=1e-5)):
        raise ValueError("Metrics disagree with shared evaluator")
    matches = all(np.isclose(actual[key], value, atol=.001, rtol=.0001)
                  for key, value in expected.items())
    print(f"Seed {seed}, {'map' if with_map else 'no-map'}: {actual}; saved-report match={matches}",
          flush=True)
    if not matches:
        raise ValueError(f"Checkpoint/report mismatch; stop before turn bins. {expected=}, {actual=}")
    return ades, fdes, {"checkpoint": str(checkpoint), "checkpoint_sha256": sha256(checkpoint),
                        "report": str(source), "report_type": source_type,
                        "expected": expected, "actual_full_test": actual,
                        "saved_report_matches": matches, "shared_evaluator_matches": True,
                        "caveat": "Archived no-map report verifies FDE at best ADE, not independent minFDE6."
                                  if source_type != "manifest" else None}


def benefit(no_map_ade, no_map_fde, map_ade, map_fde):
    result = {}
    for metric, a, b in (("minADE6", no_map_ade, map_ade), ("minFDE6", no_map_fde, map_fde)):
        result[metric] = {
            "mean_paired_improvement_m": float(np.mean(a - b)) if len(a) else None,
            # Percentage reduction of the bin mean, not mean per-sample percentage.
            "percentage_reduction_of_no_map_mean": float(100 * (a.mean() - b.mean()) / a.mean())
                if len(a) and a.mean() != 0 else None,
        }
    return result


def plotting():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    return plt


def save_figure(fig, name, plt):
    path = PROJECT_ROOT / "doc/figures" / name
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", type=int, nargs="+", default=[123, 2026])
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--plot", action="store_true")
    args = parser.parse_args()
    if args.batch_size < 1:
        parser.error("--batch-size must be positive")
    # Validate all checkpoint provenance before doing model comparisons.
    for seed in args.seeds:
        for with_map in (False, True):
            checkpoint_info(seed, with_map)
    train_distribution(args.plot)
    no_map_path = PROJECT_ROOT / "data/processed/nuscenes_test_prediction_samples.pkl"
    map_path = PROJECT_ROOT / "data/processed/nuscenes_test_prediction_samples_map.pkl"
    no_map_samples = load_samples(no_map_path)
    map_samples, alignment = align_samples(no_map_samples, load_samples(map_path))
    valid, moving, masks = angle_masks(no_map_samples)
    if not valid.all():
        raise ValueError("Invalid test futures; full-test provenance comparison would be incomplete")
    graphs = [[sample_to_graph(s) for s in samples] for samples in (no_map_samples, map_samples)]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    dataset_hashes = {"no_map": sha256(no_map_path), "map": sha256(map_path)}
    for seed in args.seeds:
        a, f, no_map_check = evaluate(seed, False, graphs[0], args.batch_size, device)
        b, g, map_check = evaluate(seed, True, graphs[1], args.batch_size, device)
        bins = {label: {"n": int(mask.sum()), "no_map": summarize(a[mask], f[mask]),
                        "map": summarize(b[mask], g[mask]),
                        "map_benefit": benefit(a[mask], f[mask], b[mask], g[mask])}
                for label, mask in masks.items()}
        result = {"seed": seed, "device": str(device), "batch_size": args.batch_size,
                  "torch_version": torch.__version__, "std_ddof": 0,
                  "angle_definition": "abs(degrees(atan2(y_final, x_final)))",
                  "minimum_endpoint_displacement_m": 5., "dataset_sha256": dataset_hashes,
                  "alignment": alignment, "provenance": {"no_map": no_map_check, "map": map_check},
                  "full_test": {"no_map": summarize(a, f), "map": summarize(b, g)},
                  "non_stationary": {"no_map": summarize(a[moving], f[moving]),
                                     "map": summarize(b[moving], g[moving])},
                  "excluded_displacement_below_5m": int((~moving).sum()), "turn_bins": bins}
        write_json(PROJECT_ROOT / f"doc/results/turn_angle_map_comparison_seed{seed}.json", result)
        print(f"Seed {seed} bins:", json.dumps(bins, indent=2), flush=True)
        if args.plot:
            plt = plotting()
            fig, ax = plt.subplots(figsize=(8, 4))
            x = np.arange(len(LABELS))
            for offset, key, label in ((-.2, "no_map", "No map"), (.2, "map", "Map")):
                ax.bar(x + offset, [bins[k][key]["minADE6"]["mean"] for k in LABELS],
                       width=.4, label=label)
            ax.set_xticks(x, LABELS)
            ax.set(xlabel="Absolute endpoint angle", ylabel="Mean minADE6 (m)",
                   title=f"Map/no-map comparison, seed {seed}; displacement ≥ 5 m")
            ax.legend()
            save_figure(fig, f"turn_angle_map_comparison_seed{seed}.png", plt)


if __name__ == "__main__":
    main()
