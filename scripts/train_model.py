"""Train and evaluate the trajectory prediction models.

Example usage:
    python scripts/train_model.py --model attention
    python scripts/train_model.py --model multimodal --seed 123
"""

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


from utils.graph_utils import sample_to_graph
from utils.models import (
    GraphNetworkAttention,
    GraphNetworkAttentionMultimodal,
)
from utils.training_utils import (
    save_run_manifest,
    set_seed,
    ade_loss,
    best_of_k_ade_loss,
    evaluate_model,
    evaluate_multimodal_model,
    train_and_validate,
)

# ========================================================

def load_graphs(split):
    path = (
        PROJECT_ROOT / "data" / "processed" / f"nuscenes_{split}_prediction_samples.pkl")

    with open(path, "rb") as f:
        samples = pickle.load(f)

    if not samples:
        raise ValueError(f"Empty processed dataset: {path}")

    print(f"{split}: {len(samples)} samples")

    return [sample_to_graph(sample) for sample in samples]


def main():

    # -------------------
    # ARGUMENTS
    # -------------------

    parser = argparse.ArgumentParser()

    # Model selection 
    parser.add_argument(
        "--model",
        choices=["attention", "multimodal"],
        required=True,
    )

    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=0.001)

    parser.add_argument(
        "--checkpoint-dir",
        type=Path,
        default=PROJECT_ROOT / "checkpoints",
    )

    args = parser.parse_args()


    # Guard clauses for argument validation
    if args.epochs < 1 or args.batch_size < 1:
        parser.error("--epochs and --batch-size must be positive")
    if not np.isfinite(args.lr) or args.lr <= 0:
        parser.error("--lr must be finite and positive")
    if not 0 <= args.seed < 2**32:
        parser.error("--seed must be between 0 and 2**32 - 1")


    # -------------------
    # REPRODUCIBILITY
    # -------------------

    set_seed(args.seed)
    # Seeded CUDA scatter operations may still vary across runs/devices.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    generator = torch.Generator()
    generator.manual_seed(args.seed)


    # -------------------
    # DATA
    # -------------------

    train_graphs = load_graphs("train")
    val_graphs = load_graphs("val")
    test_graphs = load_graphs("test")

    train_loader = DataLoader(train_graphs, batch_size=args.batch_size, shuffle=True, generator=generator)
    val_loader = DataLoader(val_graphs, batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(test_graphs, batch_size=args.batch_size, shuffle=False)


    # -------------------
    # MODEL
    # -------------------

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if args.model == "attention":

        model = GraphNetworkAttention().to(device)
        criterion = ade_loss
        evaluate = evaluate_model

        metric_names = ["ADE", "FDE"]

    else:

        model = GraphNetworkAttentionMultimodal(num_modes=6).to(device)
        criterion = best_of_k_ade_loss
        evaluate = evaluate_multimodal_model

        metric_names = ["minADE6", "minFDE6"]


    # -------------------
    # TRAINING
    # -------------------

    args.checkpoint_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    checkpoint_path = (args.checkpoint_dir
        / f"{args.model}_seed{args.seed}_best.pth"
    )

    optimizer = torch.optim.Adam( model.parameters(),lr=args.lr)

    print(
        f"\nTraining {args.model}"
        f" | seed={args.seed}"
        f" | device={device}\n"
    )

    train_losses, val_losses = train_and_validate(
        model,
        train_loader,
        val_loader,
        optimizer,
        criterion,
        num_epochs=args.epochs,
        checkpoint_path=checkpoint_path,
        device=device,
    )


    # -------------------
    # EVALUATION
    # -------------------

    model.load_state_dict(
        torch.load(
            checkpoint_path,
            map_location=device,
            weights_only=True,
        )
    )

    # Evaluate on validation and test sets
    val_metrics = evaluate(model,val_loader,device)

    # Evaluate on test set
    test_metrics = evaluate(model,test_loader, device)

    # To ensure reproducibility, save a JSON of the run with info
    save_run_manifest(
        checkpoint_path,
        model=args.model,
        seed=args.seed,
        radius=35,
        num_modes=6 if args.model == "multimodal" else 1,
        num_epochs=args.epochs,
        best_epoch=int(np.argmin(val_losses)) + 1,
        best_val_loss=min(val_losses),
        batch_size=args.batch_size,
        lr=args.lr,
        device=str(device),
        validation_metrics=dict(zip(metric_names, val_metrics)),
        test_metrics=dict(zip(metric_names, test_metrics)),
    )


    # -------------------
    # RESULTS
    # -------------------

    print(f"\nFinal results | model={args.model} | seed={args.seed}")

    print(
        "Validation:",
        " | ".join(
            f"{name}: {value:.3f} m"
            for name, value in zip(metric_names, val_metrics)
        ),
    )

    print(
        "Test:",
        " | ".join(
            f"{name}: {value:.3f} m"
            for name, value in zip(metric_names, test_metrics)
        ),
    )

    print(f"\nCheckpoint: {checkpoint_path}")


if __name__ == "__main__":
    main()
