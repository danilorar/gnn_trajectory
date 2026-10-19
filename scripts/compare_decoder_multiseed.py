"""Reproduce the frozen matched three-seed decoder comparison without training.

The canonical JSON embeds all six manifests and per-seed statistics. It is
an immutable reference, not a generated-output dependency chain. Default
execution verifies and prints results; --output writes a separate record.
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

from scripts.run_turn_angle_analysis import DATASET, LABELS, analyze_checkpoint, load_turn_dataset
from utils.experiment_utils import read_manifest, sha256, verify_frozen_implementation, write_json
from utils.graph_utils import sample_to_graph
from utils.models import GraphNetworkAttentionMultimodalWithMap, GraphNetworkAttentionMultimodalWithMapGRU
from utils.training_utils import evaluate_multimodal_model

REFERENCE = PROJECT_ROOT / "doc/results/final_decoder_multiseed_comparison.json"
METRICS = ("minADE6", "minFDE6")
CLASSES = {"mlp": GraphNetworkAttentionMultimodalWithMap, "gru": GraphNetworkAttentionMultimodalWithMapGRU}


def checkpoint_path(decoder, seed):
    if decoder == "mlp":
        return PROJECT_ROOT / f"checkpoints_60ep/multimodal_map_seed{seed}_best.pth"
    folder = PROJECT_ROOT / "checkpoints_gru_decoder"
    if seed != 2026:
        folder /= f"seed{seed}"
    return folder / f"multimodal_map_gru_seed{seed}_best.pth"


def compare(values):
    result = {name: {"mean": float(np.mean(numbers)), "std": float(np.std(numbers, ddof=0)),
                     "per_seed": dict(zip((42, 123, 2026), map(float, numbers)))}
              for name, numbers in values.items()}
    mlp, gru = result["mlp"]["mean"], result["gru"]["mean"]
    result.update(absolute_improvement_m=mlp-gru, percentage_improvement=100*(mlp-gru)/mlp)
    return result


def verify_run(decoder, seed, reference, hashes, data, val_loader, device):
    checkpoint = checkpoint_path(decoder, seed)
    manifest = read_manifest(checkpoint)
    saved = reference["runs"][decoder][str(seed)]
    expected_name = "multimodal_map" if decoder == "mlp" else "multimodal_map_gru"
    for name, expected in {"model": expected_name, "seed": seed, "num_epochs": 60,
                           "num_modes": 6, "batch_size": 64, "lr": .001, "radius": 35}.items():
        if manifest.get(name) != expected:
            raise ValueError(f"Unexpected manifest setting: {decoder}/{seed}/{name}")
    if sha256(checkpoint) != saved["checkpoint_sha256"]:
        raise ValueError(f"Checkpoint changed: {checkpoint}")
    # Preserve historical metadata limitations rather than inventing old hashes.
    if manifest.get("dataset_sha256") and manifest["dataset_sha256"] != hashes:
        raise ValueError("Dataset hashes differ from training provenance")
    if manifest.get("sampling") not in (None, "Standard uniform shuffle without replacement; unmodified validation/test"):
        raise ValueError("Weighted sampling does not belong in this comparison")
    model = CLASSES[decoder](num_modes=6).to(device)
    model.load_state_dict(torch.load(checkpoint, map_location=device, weights_only=True), strict=True)
    validation = dict(zip(METRICS, map(float, evaluate_multimodal_model(model, val_loader, device))))
    for name in METRICS:
        if not np.isclose(validation[name], manifest["validation_metrics"][name], atol=.001, rtol=.0001):
            raise ValueError(f"Validation does not reproduce manifest: {decoder}/{seed}/{name}")
    analysis = analyze_checkpoint(checkpoint, seed, device, data)
    old = saved["turn_analysis"]
    for group in ("full_test", "non_stationary", "turn_bins"):
        actual_groups = analysis[group] if group == "turn_bins" else {group: analysis[group]}
        expected_groups = old[group] if group == "turn_bins" else {group: old[group]}
        for label, stats in actual_groups.items():
            expected = expected_groups[label]
            if stats["n"] != expected["n"]:
                raise ValueError("Different evaluation population")
            for metric in METRICS:
                for statistic in ("mean", "median", "std"):
                    if not np.isclose(stats[metric][statistic], expected[metric][statistic], atol=1e-5, rtol=1e-5):
                        raise ValueError(f"Canonical statistics differ: {decoder}/{seed}/{label}/{metric}/{statistic}")
    print(f"Verified {decoder} seed {seed}: test={analysis['shared_full_test_metrics']}", flush=True)
    return {"best_epoch": manifest["best_epoch"], "validation_metrics": validation,
            "test_metrics": analysis["shared_full_test_metrics"], "analysis": analysis}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--output", type=Path, help="Optional new JSON path; canonical files are never overwritten")
    parser.add_argument("--plot", action="store_true", help="Save one chart beside --output")
    args = parser.parse_args()
    if args.plot and not args.output:
        parser.error("--plot requires --output")
    for path in ([args.output, args.output.with_suffix('.png')] if args.plot else [args.output]):
        if path and path.exists():
            parser.error(f"Output already exists: {path}")
    behavior = verify_frozen_implementation()
    reference = json.loads(REFERENCE.read_text())
    hashes = {split: sha256(PROJECT_ROOT / f"data/processed/nuscenes_{split}_prediction_samples_map.pkl")
              for split in ("train", "val", "test")}
    if hashes != reference["dataset_sha256"]:
        raise ValueError("Processed datasets differ from the final experiment")
    data = load_turn_dataset(DATASET)
    with (PROJECT_ROOT / "data/processed/nuscenes_val_prediction_samples_map.pkl").open("rb") as handle:
        val_samples = pickle.load(handle)
    val_loader = DataLoader([sample_to_graph(s) for s in val_samples], batch_size=64, shuffle=False)
    device = torch.device(args.device)
    runs = {d: {seed: verify_run(d, seed, reference, hashes, data, val_loader, device)
                for seed in (42, 123, 2026)} for d in CLASSES}
    overall = {name: compare({d: [runs[d][s]["test_metrics"][name] for s in (42, 123, 2026)] for d in CLASSES})
               for name in METRICS}
    bins = {label: {name: compare({d: [runs[d][s]["analysis"]["turn_bins"][label][name]["mean"]
                                       for s in (42, 123, 2026)] for d in CLASSES}) for name in METRICS}
            for label in LABELS}
    for actual, saved in [(overall, reference["overall"])] + [(bins[label], reference["turn_bins"][label]) for label in LABELS]:
        for name in METRICS:
            for decoder in CLASSES:
                if not np.allclose([actual[name][decoder][k] for k in ("mean", "std")],
                                   [saved[name][decoder][k] for k in ("mean", "std")], atol=1e-5, rtol=1e-5):
                    raise ValueError("Matched comparison differs from canonical results")
    result = {"reference": str(REFERENCE), "reference_sha256": sha256(REFERENCE),
              "behavior_check": behavior, "canonical_results_match": True,
              "dataset_sha256": hashes, "std_ddof": 0, "runs": runs, "overall": overall, "turn_bins": bins}
    print(json.dumps({"overall": overall, "turn_bins": bins}, indent=2))
    print("All six validation/test evaluations and turn-angle statistics reproduce the canonical comparison.")
    print("Historical MLP seed-2026 sampler/training-data provenance remains limited as documented in the canonical report.")
    if args.output:
        write_json(args.output, result)
    if args.plot:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(1, 2, figsize=(11, 4))
        x = np.arange(len(LABELS))
        for ax, name in zip(axes, METRICS):
            for offset, decoder in ((-.18, "mlp"), (.18, "gru")):
                ax.bar(x+offset, [bins[label][name][decoder]["mean"] for label in LABELS],
                       yerr=[bins[label][name][decoder]["std"] for label in LABELS], width=.36, capsize=4, label=decoder.upper())
            ax.set_xticks(x, LABELS)
            ax.set(xlabel="Absolute endpoint angle (displacement ≥5 m)", ylabel=f"Mean {name} (m)")
            ax.legend()
        fig.tight_layout()
        fig.savefig(args.output.with_suffix(".png"), dpi=160)
        plt.close(fig)


if __name__ == "__main__":
    main()
