"""Train a trajectory model, or evaluate an existing checkpoint."""

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

from utils.graph_utils import sample_to_graph
from utils.models import (
    TargetHistoryMLP,
    GraphNetworkAttention,
    GraphNetworkAttentionMultimodal,
    GraphNetworkAttentionMultimodalWithMap,
    GraphNetworkAttentionMultimodalWithMapGRU,
)
from utils.training_utils import (
    ade_loss,
    best_of_k_ade_loss,
    evaluate_model,
    evaluate_multimodal_model,
    read_manifest,
    save_run_manifest,
    set_seed,
    train_and_validate,
)


def load_graphs(split, use_map):
    suffix = "_map" if use_map else ""
    path = PROJECT_ROOT / f"data/processed/nuscenes_{split}_prediction_samples{suffix}.pkl"
    with path.open("rb") as handle:
        samples = pickle.load(handle)
    if not samples:
        raise ValueError(f"Empty processed dataset: {path}")
    print(f"{split}: {len(samples)} samples", flush=True)
    return [sample_to_graph(sample) for sample in samples]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, choices=(
        "target_mlp", "attention", "multimodal", "multimodal_map", "multimodal_map_gru",
    ))
    parser.add_argument("--seed", type=int, help="Default: 42, or the saved seed when evaluating")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--checkpoint-dir", type=Path, default=PROJECT_ROOT / "checkpoints")
    parser.add_argument("--evaluate-only", action="store_true", help="Load weights without training or saving them")
    parser.add_argument("--checkpoint", type=Path, help="Existing weights for --evaluate-only")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--output", type=Path, help="Optional new JSON evaluation summary")
    args = parser.parse_args()

    if args.epochs < 1 or args.batch_size < 1:
        parser.error("--epochs and --batch-size must be positive")
    if not np.isfinite(args.lr) or args.lr <= 0:
        parser.error("--lr must be finite and positive")
    if args.checkpoint and not args.evaluate_only:
        parser.error("--checkpoint requires --evaluate-only")
    if args.output and args.output.exists():
        parser.error(f"Output already exists: {args.output}")

    seed = 42 if args.seed is None else args.seed
    checkpoint = args.checkpoint or args.checkpoint_dir / f"{args.model}_seed{seed}_best.pth"
    if args.evaluate_only and args.seed is None and checkpoint.with_suffix(".json").exists():
        seed = read_manifest(checkpoint)["seed"]
    if not 0 <= seed < 2**32:
        parser.error("--seed must be between 0 and 2**32 - 1")
    if not args.evaluate_only and (checkpoint.exists() or checkpoint.with_suffix(".json").exists()):
        raise FileExistsError(f"Refusing to overwrite existing checkpoint: {checkpoint}")

    set_seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    device = args.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device)

    use_map = args.model in ("multimodal_map", "multimodal_map_gru")
    if not args.evaluate_only:
        train_graphs = load_graphs("train", use_map)
    val_graphs = load_graphs("val", use_map)
    test_graphs = load_graphs("test", use_map)

    # All five models use the same graph features and fixed 35 m datasets.
    if args.model == "target_mlp":
        model = TargetHistoryMLP()
    elif args.model == "attention":
        model = GraphNetworkAttention()
    elif args.model == "multimodal":
        model = GraphNetworkAttentionMultimodal(num_modes=6)
    elif args.model == "multimodal_map":
        model = GraphNetworkAttentionMultimodalWithMap(num_modes=6)
    else:
        model = GraphNetworkAttentionMultimodalWithMapGRU(num_modes=6)
    model = model.to(device)

    multimodal = args.model in ("multimodal", "multimodal_map", "multimodal_map_gru")
    if multimodal:
        criterion = best_of_k_ade_loss
        evaluator = evaluate_multimodal_model
        metric_names = ("minADE6", "minFDE6")
    else:
        criterion = ade_loss
        evaluator = evaluate_model
        metric_names = ("ADE", "FDE")

    if not args.evaluate_only:
        train_loader = DataLoader(
            train_graphs, batch_size=args.batch_size, shuffle=True,
            generator=torch.Generator().manual_seed(seed),
        )
    val_loader = DataLoader(val_graphs, batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(test_graphs, batch_size=args.batch_size, shuffle=False)

    if not args.evaluate_only:
        checkpoint.parent.mkdir(parents=True, exist_ok=True)
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
        print(f"Training {args.model} | seed={seed} | device={device}", flush=True)
        train_losses, val_losses = train_and_validate(
            model, train_loader, val_loader, optimizer, criterion,
            args.epochs, checkpoint, device,
        )

    # Evaluate the best validation checkpoint, rather than the final epoch.
    model.load_state_dict(torch.load(checkpoint, map_location=device, weights_only=True), strict=True)
    val_ade, val_fde = evaluator(model, val_loader, device)
    test_ade, test_fde = evaluator(model, test_loader, device)
    validation = {metric_names[0]: float(val_ade), metric_names[1]: float(val_fde)}
    testing = {metric_names[0]: float(test_ade), metric_names[1]: float(test_fde)}

    if not args.evaluate_only:
        save_run_manifest(
            checkpoint, model=args.model, model_class=type(model).__name__, seed=seed,
            radius=35, num_modes=6 if multimodal else 1, num_epochs=args.epochs,
            best_epoch=int(np.argmin(val_losses)) + 1, best_val_loss=min(val_losses),
            batch_size=args.batch_size, lr=args.lr, device=str(device),
            validation_metrics=validation, test_metrics=testing,
            sampling="Standard uniform shuffle without replacement; unmodified validation/test",
        )

    print(f"Final results | model={args.model} | seed={seed}")
    for label, metrics in (("Validation", validation), ("Test", testing)):
        print(f"{label}: {metric_names[0]}: {metrics[metric_names[0]]:.6f} m | "
              f"{metric_names[1]}: {metrics[metric_names[1]]:.6f} m")
    print(f"Checkpoint: {checkpoint}")
    if args.output:
        result = {
            "model": args.model, "seed": seed, "checkpoint": str(checkpoint),
            "validation_metrics": validation, "test_metrics": testing,
        }
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
