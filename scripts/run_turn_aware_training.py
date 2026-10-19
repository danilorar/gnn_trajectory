"""Controlled train-only oversampling experiment for the existing map model.

Fresh initialization, seed 2026, 60 epochs, unchanged Adam/loss/validation.
Weights for non-stationary endpoint-angle bins: 1, 1, 2, 4. Samples with
displacement <5 m keep weight 1. Each epoch still draws N_train examples.
"""

import argparse
import hashlib
import json
import pickle
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import WeightedRandomSampler
from torch_geometric.loader import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.run_turn_angle_analysis import endpoint_geometry, sha256
from utils.graph_utils import sample_to_graph
from utils.models import GraphNetworkAttentionMultimodalWithMap
from utils.training_utils import (
    best_of_k_ade_loss, evaluate_multimodal_model, save_run_manifest,
    set_seed, train_and_validate,
)

GROUPS = ("stationary (<5 m)", "0-15 deg", "15-30 deg", "30-60 deg", "60+ deg")
GROUP_WEIGHTS = np.array([1., 1., 1., 2., 4.])


def sampling_plan(samples, seed):
    groups = []
    for index, sample in enumerate(samples):
        future = np.asarray(sample["target_future"])
        if future.shape != (12, 2) or not np.isfinite(future).all():
            raise ValueError(f"Invalid training future at index {index}")
        displacement, angle = endpoint_geometry(future)
        group = 0 if displacement < 5 else 1 + int(np.searchsorted([15, 30, 60], angle, side="right"))
        groups.append(group)
    groups = np.asarray(groups)
    weights = torch.tensor(GROUP_WEIGHTS[groups], dtype=torch.double)
    # A separate generator previews exactly the first epoch without advancing
    # either the real sampler generator or the global model-initialization RNG.
    preview = torch.multinomial(weights, len(samples), replacement=True,
                                generator=torch.Generator().manual_seed(seed)).numpy()
    rows = {}
    for index, label in enumerate(GROUPS):
        original = int((groups == index).sum())
        drawn = int((groups[preview] == index).sum())
        rows[label] = {
            "weight": float(GROUP_WEIGHTS[index]), "original_n": original,
            "original_percentage": 100 * original / len(samples),
            "expected_percentage": float(100 * weights[groups == index].sum() / weights.sum()),
            "first_epoch_draws": drawn, "first_epoch_percentage": 100 * drawn / len(samples),
        }
    report = {"method": "WeightedRandomSampler", "replacement": True,
              "samples_per_epoch": len(samples), "groups": rows,
              "preview_unique_samples": int(len(np.unique(preview))),
              "preview_draw_indices_sha256": hashlib.sha256(preview.tobytes()).hexdigest(),
              "validation_and_test_sampling": "Sequential, unweighted; all examples once"}
    return weights, report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint-dir", type=Path, default=PROJECT_ROOT / "checkpoints_turn_aware")
    args = parser.parse_args()
    baseline_path = PROJECT_ROOT / "checkpoints_60ep/multimodal_map_seed2026_best.pth"
    baseline = json.loads(baseline_path.with_suffix(".json").read_text())
    seed, epochs, batch_size, lr = (baseline[k] for k in ("seed", "num_epochs", "batch_size", "lr"))
    if (seed, epochs, batch_size, lr) != (2026, 60, 64, .001):
        raise ValueError("Baseline hyperparameters differ from the planned controlled experiment")
    checkpoint = args.checkpoint_dir / "multimodal_map_seed2026_best.pth"
    if checkpoint.exists() or checkpoint.with_suffix(".json").exists():
        raise FileExistsError(f"Refusing to overwrite an existing experiment: {checkpoint}")
    if args.checkpoint_dir.resolve() == baseline_path.parent.resolve():
        raise ValueError("Experiment directory must differ from the baseline directory")
    args.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    set_seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if str(device) != baseline["device"]:
        raise RuntimeError(f"Use the baseline device ({baseline['device']}) for this comparison")

    dataset_paths = {split: PROJECT_ROOT / f"data/processed/nuscenes_{split}_prediction_samples_map.pkl"
                     for split in ("train", "val", "test")}
    graphs = {}
    report = None
    for split, path in dataset_paths.items():
        with path.open("rb") as handle:
            samples = pickle.load(handle)
        if not samples:
            raise ValueError(f"Empty {split} split")
        if split == "train":
            weights, report = sampling_plan(samples, seed)
            report["batches_per_epoch"] = int(np.ceil(len(samples) / batch_size))
            print("Sampling plan (expected and exact preview of epoch 1):", flush=True)
            print(json.dumps(report, indent=2), flush=True)
            (args.checkpoint_dir / "sampling_distribution.json").write_text(json.dumps(report, indent=2) + "\n")
        print(f"Building {split} graphs: {len(samples)} samples", flush=True)
        graphs[split] = [sample_to_graph(s) for s in samples]
    del samples

    # Separate explicit generators isolate sampling from model initialization
    # and DataLoader's base seed. The preview therefore matches the first draw.
    sampler = WeightedRandomSampler(weights, num_samples=len(graphs["train"]), replacement=True,
                                    generator=torch.Generator().manual_seed(seed))
    train_loader = DataLoader(graphs["train"], batch_size=batch_size, sampler=sampler,
                              generator=torch.Generator().manual_seed(seed))
    val_loader = DataLoader(graphs["val"], batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(graphs["test"], batch_size=batch_size, shuffle=False)
    model = GraphNetworkAttentionMultimodalWithMap(num_modes=6).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    protected_hashes = {str(p): sha256(p) for p in (baseline_path, baseline_path.with_suffix(".json"))}
    dataset_hashes = {split: sha256(path) for split, path in dataset_paths.items()}
    code_hashes = {name: sha256(PROJECT_ROOT / name) for name in
                   ("utils/models.py", "utils/graph_utils.py", "utils/training_utils.py")}
    print(f"Training fresh map model: seed={seed}, epochs={epochs}, device={device}, "
          f"draws/epoch={len(sampler)}, batches/epoch={len(train_loader)}", flush=True)
    train_losses, val_losses = train_and_validate(
        model, train_loader, val_loader, optimizer, best_of_k_ade_loss,
        num_epochs=epochs, checkpoint_path=checkpoint, device=device,
    )
    model.load_state_dict(torch.load(checkpoint, map_location=device, weights_only=True))
    val_metrics = evaluate_multimodal_model(model, val_loader, device)
    test_metrics = evaluate_multimodal_model(model, test_loader, device)
    for name, digest in protected_hashes.items():
        if sha256(name) != digest:
            raise RuntimeError(f"Baseline artifact changed during training: {name}")
    for split, path in dataset_paths.items():
        if sha256(path) != dataset_hashes[split]:
            raise RuntimeError(f"Dataset changed during training: {path}")
    names = ("minADE6", "minFDE6")
    save_run_manifest(
        checkpoint, model="multimodal_map", seed=seed, radius=35, num_modes=6,
        num_epochs=epochs, best_epoch=int(np.argmin(val_losses)) + 1,
        best_val_loss=min(val_losses), batch_size=batch_size, lr=lr, device=str(device),
        validation_metrics=dict(zip(names, val_metrics)), test_metrics=dict(zip(names, test_metrics)),
        experiment="turn_aware_training", sampling=report,
        baseline_checkpoint=str(baseline_path), baseline_sha256=protected_hashes[str(baseline_path)],
        dataset_sha256=dataset_hashes, model_graph_training_code_sha256=code_hashes,
        initialization="Fresh seed-2026 initialization; not resumed from a trained checkpoint",
    )
    history = {"train_losses": train_losses, "val_losses": val_losses}
    (args.checkpoint_dir / "training_history.json").write_text(json.dumps(history, indent=2) + "\n")
    print(f"Best epoch: {int(np.argmin(val_losses)) + 1}", flush=True)
    print(f"Validation: {dict(zip(names, val_metrics))}", flush=True)
    print(f"Test: {dict(zip(names, test_metrics))}", flush=True)
    print(f"Saved {checkpoint} and manifest; baseline and datasets verified unchanged.", flush=True)


if __name__ == "__main__":
    main()
