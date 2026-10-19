"""Train or evaluate the project models through one entry point."""

import argparse
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
from utils.models import (
    TargetHistoryMLP, GraphNetworkAttention, GraphNetworkAttentionMultimodal,
    GraphNetworkAttentionMultimodalWithMap, GraphNetworkAttentionMultimodalWithMapGRU,
)
from utils.training_utils import (
    ade_loss, best_of_k_ade_loss, evaluate_model, evaluate_multimodal_model,
    save_run_manifest, set_seed, train_and_validate,
)

MODEL_CLASSES = {
    "target_mlp": TargetHistoryMLP,
    "attention": GraphNetworkAttention,
    "multimodal": GraphNetworkAttentionMultimodal,
    "multimodal_map": GraphNetworkAttentionMultimodalWithMap,
    "multimodal_map_gru": GraphNetworkAttentionMultimodalWithMapGRU,
}
MAP_MODELS = {"multimodal_map", "multimodal_map_gru"}
MULTIMODAL_MODELS = {"multimodal", *MAP_MODELS}


def dataset_path(split, use_map=False):
    suffix = "_map" if use_map else ""
    return PROJECT_ROOT / f"data/processed/nuscenes_{split}_prediction_samples{suffix}.pkl"


def load_graphs(split, use_map=False):
    path = dataset_path(split, use_map)
    with path.open("rb") as handle:
        samples = pickle.load(handle)
    if not samples:
        raise ValueError(f"Empty processed dataset: {path}")
    print(f"{split}: {len(samples)} samples", flush=True)
    return [sample_to_graph(sample) for sample in samples]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=MODEL_CLASSES, required=True)
    parser.add_argument("--seed", type=int, help="Training seed; default 42, or checkpoint seed during evaluation")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=.001)
    parser.add_argument("--checkpoint-dir", type=Path, default=PROJECT_ROOT / "checkpoints")
    parser.add_argument("--evaluate-only", action="store_true", help="Load existing weights; never train or save weights")
    parser.add_argument("--checkpoint", type=Path, help="Explicit checkpoint for --evaluate-only")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--output", type=Path, help="Optional separate JSON evaluation record; refuses overwrite")
    args = parser.parse_args()
    if args.epochs < 1 or args.batch_size < 1:
        parser.error("--epochs and --batch-size must be positive")
    if not np.isfinite(args.lr) or args.lr <= 0:
        parser.error("--lr must be finite and positive")
    if args.checkpoint and not args.evaluate_only:
        parser.error("--checkpoint is only used with --evaluate-only")
    if args.output and args.output.exists():
        parser.error(f"Output already exists: {args.output}")
    if args.seed is not None and not 0 <= args.seed < 2**32:
        parser.error("--seed must be between 0 and 2**32 - 1")
    checkpoint = args.checkpoint or args.checkpoint_dir / f"{args.model}_seed{args.seed if args.seed is not None else 42}_best.pth"
    manifest = read_manifest(checkpoint) if args.evaluate_only else None
    seed = args.seed if args.seed is not None else (manifest["seed"] if manifest else 42)
    if manifest and (manifest["model"] != args.model or manifest["seed"] != seed):
        parser.error("Checkpoint manifest model/seed differs from the requested model/seed")
    if not args.evaluate_only and (checkpoint.exists() or checkpoint.with_suffix(".json").exists()):
        raise FileExistsError(f"Refusing to overwrite existing checkpoint: {checkpoint}")
    set_seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    device = torch.device(("cuda" if torch.cuda.is_available() else "cpu") if args.device == "auto" else args.device)
    use_map = args.model in MAP_MODELS
    multimodal = args.model in MULTIMODAL_MODELS
    criterion = best_of_k_ade_loss if multimodal else ade_loss
    evaluate = evaluate_multimodal_model if multimodal else evaluate_model
    metric_names = ("minADE6", "minFDE6") if multimodal else ("ADE", "FDE")

    # Build loaders before initializing the model, preserving the tested RNG order.
    train_loader = None
    if not args.evaluate_only:
        train_loader = DataLoader(load_graphs("train", use_map), batch_size=args.batch_size,
                                  shuffle=True, generator=torch.Generator().manual_seed(seed))
    val_loader = DataLoader(load_graphs("val", use_map), batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(load_graphs("test", use_map), batch_size=args.batch_size, shuffle=False)
    hashes = {split: sha256(dataset_path(split, use_map))
              for split in (("val", "test") if args.evaluate_only else ("train", "val", "test"))}
    if manifest and manifest.get("dataset_sha256"):
        for split, digest in hashes.items():
            if manifest["dataset_sha256"].get(split) != digest:
                raise ValueError(f"Dataset differs from the checkpoint manifest: {split}")
    model = MODEL_CLASSES[args.model]().to(device)
    if not args.evaluate_only:
        checkpoint.parent.mkdir(parents=True, exist_ok=True)
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
        print(f"Training {args.model} | seed={seed} | device={device}", flush=True)
        train_losses, val_losses = train_and_validate(
            model, train_loader, val_loader, optimizer, criterion,
            args.epochs, checkpoint, device,
        )
    model.load_state_dict(torch.load(checkpoint, map_location=device, weights_only=True), strict=True)
    validation = dict(zip(metric_names, map(float, evaluate(model, val_loader, device))))
    testing = dict(zip(metric_names, map(float, evaluate(model, test_loader, device))))
    if manifest:
        for field, actual in (("validation_metrics", validation), ("test_metrics", testing)):
            for name in metric_names:
                if not np.isclose(actual[name], manifest[field][name], atol=.001, rtol=.0001):
                    raise ValueError(f"Evaluation does not reproduce manifest: {field}/{name}")
    else:
        save_run_manifest(
            checkpoint, model=args.model, model_class=type(model).__name__, seed=seed,
            radius=35, num_modes=6 if multimodal else 1, num_epochs=args.epochs,
            best_epoch=int(np.argmin(val_losses)) + 1, best_val_loss=min(val_losses),
            batch_size=args.batch_size, lr=args.lr, device=str(device),
            validation_metrics=validation, test_metrics=testing, dataset_sha256=hashes,
            sampling="Standard uniform shuffle without replacement; unmodified validation/test",
        )
    result = {"model": args.model, "seed": seed, "checkpoint": str(checkpoint),
              "checkpoint_sha256": sha256(checkpoint), "dataset_sha256": hashes,
              "validation_metrics": validation, "test_metrics": testing,
              "manifest_matches": True, "evaluate_only": args.evaluate_only}
    print(f"Final results | model={args.model} | seed={seed}")
    for label, metrics in (("Validation", validation), ("Test", testing)):
        print(label + ": " + " | ".join(f"{name}: {value:.6f} m" for name, value in metrics.items()))
    print(f"Checkpoint: {checkpoint}")
    if args.output:
        write_json(args.output, result)


if __name__ == "__main__":
    main()
