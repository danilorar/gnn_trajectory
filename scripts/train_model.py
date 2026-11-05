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
    multimodal_metrics,
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


def evaluate_v2(model, loader, device, miss_threshold=2.0):
    """minADE6 / minFDE6 (same metric code as v1) and miss rate = share of samples with minFDE6 > 2 m."""
    model.eval()
    ade_sum = fde_sum = miss_sum = n = 0.0
    wins = None
    with torch.no_grad():
        for batch in loader:
            batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}
            traj, _ = model(batch)
            ade, fde = multimodal_metrics(traj, batch["target_future"])
            ade_sum, fde_sum, n = ade_sum + ade.sum().item(), fde_sum + fde.sum().item(), n + len(ade)
            miss_sum += (fde > miss_threshold).sum().item()
            # Which mode wins (lowest ADE over the path) per sample, to spot dead modes
            winner = (traj - batch["target_future"][:, None]).norm(dim=-1).mean(-1).argmin(-1)
            counts = torch.bincount(winner, minlength=traj.shape[1]).cpu()
            wins = counts if wins is None else wins + counts
    evaluate_v2.last_mode_wins = (wins / n).tolist()
    return ade_sum / n, fde_sum / n, miss_sum / n


def run_v2(args, seed, device, checkpoint):
    """Train / evaluate the v2 scene transformer (dict batches from data/processed_v2)."""
    import time
    from torch.utils.data import DataLoader as TorchLoader
    from utils.dataset_v2 import TrajectoryDatasetV2
    from utils.models_v2 import V2Predictor

    def loader(split, shuffle):
        ds = TrajectoryDatasetV2(PROJECT_ROOT / f"data/processed_v2/nuscenes_{split}_prediction_samples_map.pkl")
        print(f"{split}: {len(ds)} samples", flush=True)
        return TorchLoader(ds, batch_size=args.batch_size, shuffle=shuffle, num_workers=4, pin_memory=True,
                           persistent_workers=True,
                           generator=torch.Generator().manual_seed(seed) if shuffle else None)

    model = V2Predictor(np.load(PROJECT_ROOT / "assets/anchors.npy"), assign=args.assign,
                        topology=args.topology, scale_inputs=args.scale_inputs).to(device)
    val_loader, test_loader = loader("val", False), loader("test", False)
    val_metrics = []
    stopped_epoch = None

    if not args.evaluate_only:
        train_loader = loader("train", True)
        checkpoint.parent.mkdir(parents=True, exist_ok=True)
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
        # 1 epoch linear warmup, then cosine down to 1e-5 (stepped per batch)
        warmup = len(train_loader)
        scheduler = torch.optim.lr_scheduler.SequentialLR(optimizer, [
            torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=0.01, total_iters=warmup),
            torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs * len(train_loader) - warmup,
                                                       eta_min=1e-5),
        ], milestones=[warmup])
        print(f"Training v2 (assign={args.assign}, topology={args.topology}, scale_inputs={args.scale_inputs}) | {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M params | "
              f"seed={seed} | device={device}", flush=True)

        for epoch in range(1, args.epochs + 1):
            model.train()
            start, loss_sum, reg_sum, cls_sum = time.time(), 0.0, 0.0, 0.0
            for batch in train_loader:
                batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}
                traj, logits = model(batch)
                loss, parts = model.loss(batch, traj, logits)
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                loss_sum, reg_sum, cls_sum = loss_sum + loss.item(), reg_sum + parts["reg"], cls_sum + parts["cls"]

            steps = len(train_loader)
            val_ade, val_fde, val_miss = evaluate_v2(model, val_loader, device)
            val_metrics.append(val_ade)
            if val_ade == min(val_metrics):
                torch.save(model.state_dict(), checkpoint)
            best_epoch = int(np.argmin(val_metrics)) + 1
            print(f"epoch {epoch:3d} | loss {loss_sum / steps:.3f} (reg {reg_sum / steps:.3f}, "
                  f"cls {cls_sum / steps:.3f}) | val minADE6 {val_ade:.3f} m, minFDE6 {val_fde:.3f} m, "
                  f"miss {val_miss:.1%} | lr {scheduler.get_last_lr()[0]:.1e} | {time.time() - start:.0f}s", flush=True)
            wins = evaluate_v2.last_mode_wins
            print("          val mode wins: " + " ".join(f"{w:.1%}" for w in wins)
                  + ("  <- DEAD MODE (<3%)" if min(wins) < 0.03 else ""), flush=True)

            # Early stopping on val minADE6; the cosine schedule stays defined over all args.epochs
            if args.patience and epoch - best_epoch >= args.patience and epoch < args.epochs:
                stopped_epoch = epoch
                print(f"Early stopping at epoch {epoch}: no val minADE6 improvement for {args.patience} epochs "
                      f"(best epoch {best_epoch}, {min(val_metrics):.3f} m)", flush=True)
                break

    model.load_state_dict(torch.load(checkpoint, map_location=device, weights_only=True), strict=True)
    names = ("minADE6", "minFDE6", "miss_rate_2m")
    validation = dict(zip(names, map(float, evaluate_v2(model, val_loader, device))))
    # Test is evaluated once, on the best-val checkpoint, only after training (settings are chosen on val).
    testing = dict(zip(names, map(float, evaluate_v2(model, test_loader, device))))

    if not args.evaluate_only:
        save_run_manifest(
            checkpoint, model="v2", assign=args.assign, topology=args.topology, scale_inputs=args.scale_inputs,
            model_class=type(model).__name__, seed=seed, num_modes=6,
            num_epochs=args.epochs, best_epoch=int(np.argmin(val_metrics)) + 1, batch_size=args.batch_size,
            patience=args.patience, stopped_epoch=stopped_epoch, epochs_trained=len(val_metrics),
            lr=args.lr, device=str(device), validation_metrics=validation, test_metrics=testing,
            optimizer="AdamW (wd 0.01), 1-epoch linear warmup + cosine to 1e-5, grad clip 1.0, dropout 0.1",
            anchors="assets/anchors.npy",
        )
    print("Final results | model=v2 | seed=%d" % seed)
    for label, m in (("Validation", validation), ("Test", testing)):
        print(f"{label}: minADE6: {m['minADE6']:.6f} m | minFDE6: {m['minFDE6']:.6f} m | "
              f"miss rate (>2 m): {m['miss_rate_2m']:.4f}")
    print(f"Checkpoint: {checkpoint}")
    return validation, testing


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model", required=True,
        choices=("target_mlp", "attention", "multimodal", "multimodal_map", "multimodal_map_gru", "v2"),
    )
    parser.add_argument("--seed", type=int, help="Default: 42, or the saved seed when evaluating")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--checkpoint-dir", type=Path, default=PROJECT_ROOT / "checkpoints")
    parser.add_argument("--evaluate-only", action="store_true", help="Load weights without training or saving them")
    parser.add_argument("--checkpoint", type=Path, help="Existing weights for --evaluate-only")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--output", type=Path, help="Optional new JSON evaluation summary")
    parser.add_argument(
        "--assign", choices=("predicted", "anchor"), default="predicted",
        help="v2 only: mode used for the loss (predicted = lowest-ADE mode, anchor = closest anchor)",
    )
    parser.add_argument("--patience", type=int, default=10,
                        help="v2 only: early stopping patience on val minADE6 in epochs (0 = off)")
    parser.add_argument("--topology", action="store_true",
                        help="v2 only: add successor/predecessor lane flags to the attention bias (7 inputs)")
    parser.add_argument("--scale-inputs", action="store_true",
                        help="v2 only: divide agent/lane positions by 50 m before the token MLPs")
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

    if args.model == "v2":
        run_v2(args, seed, device, checkpoint)
        return

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
            train_graphs,
            batch_size=args.batch_size,
            shuffle=True,
            generator=torch.Generator().manual_seed(seed),
        )
    val_loader = DataLoader(val_graphs, batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(test_graphs, batch_size=args.batch_size, shuffle=False)

    if not args.evaluate_only:
        checkpoint.parent.mkdir(parents=True, exist_ok=True)
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
        print(f"Training {args.model} | seed={seed} | device={device}", flush=True)
        train_losses, val_losses = train_and_validate(
            model,
            train_loader,
            val_loader,
            optimizer,
            criterion,
            args.epochs,
            checkpoint,
            device,
        )

    # Evaluate the best validation checkpoint, rather than the final epoch.
    model.load_state_dict(
        torch.load(checkpoint, map_location=device, weights_only=True),
        strict=True,
    )
    val_ade, val_fde = evaluator(model, val_loader, device)
    test_ade, test_fde = evaluator(model, test_loader, device)
    validation = {metric_names[0]: float(val_ade), metric_names[1]: float(val_fde)}
    testing = {metric_names[0]: float(test_ade), metric_names[1]: float(test_fde)}

    if not args.evaluate_only:
        save_run_manifest(
            checkpoint,
            model=args.model,
            model_class=type(model).__name__,
            seed=seed,
            radius=35,
            num_modes=6 if multimodal else 1,
            num_epochs=args.epochs,
            best_epoch=int(np.argmin(val_losses)) + 1,
            best_val_loss=min(val_losses),
            batch_size=args.batch_size,
            lr=args.lr,
            device=str(device),
            validation_metrics=validation,
            test_metrics=testing,
            sampling="Standard uniform shuffle without replacement; unmodified validation/test",
        )

    print(f"Final results | model={args.model} | seed={seed}")
    for label, metrics in (("Validation", validation), ("Test", testing)):
        print(
            f"{label}: {metric_names[0]}: {metrics[metric_names[0]]:.6f} m | "
            f"{metric_names[1]}: {metrics[metric_names[1]]:.6f} m"
        )
    print(f"Checkpoint: {checkpoint}")
    if args.output:
        result = {
            "model": args.model,
            "seed": seed,
            "checkpoint": str(checkpoint),
            "validation_metrics": validation,
            "test_metrics": testing,
        }
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
