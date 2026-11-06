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


def agent_min_errors(traj, future, valid):
    """Multi-agent minADE / minFDE per agent: ADE over the valid future steps, FDE at the last valid step,
    each with its own best mode. traj (B, M, K, F, 2), future (B, M, F, 2), valid (B, M, F) -> (B, M), (B, M)."""
    err = (traj - future[:, :, None]).norm(dim=-1)                                      # (B, M, K, F)
    v = valid.float()[:, :, None]
    ade = ((err * v).sum(-1) / v.sum(-1).clamp(min=1)).min(-1).values
    last = (valid * torch.arange(1, valid.shape[-1] + 1, device=valid.device)).argmax(-1)  # last valid step
    fde = err.gather(3, last[:, :, None, None].expand(-1, -1, err.shape[2], 1))[..., 0].min(-1).values
    return ade, fde


def moving_agents(batch, slots, threshold=1.0):
    """(B, M) bool: the agent is > threshold m from its current position at some valid history or future step.
    Parked or stopped agents are trivial to predict, so all-agent numbers are also reported on moving ones."""
    from utils.models_v2 import token_poses
    now = token_poses(batch)[0][:, slots, :2]                                            # (B, M, 2)
    hist, hv = batch["agent_hist"][:, slots, :, :2], batch["agent_hist_valid"][:, slots]
    fut, fv = batch["agent_future"][:, slots], batch["agent_future_valid"][:, slots]
    d_hist = ((hist - now[:, :, None]).norm(dim=-1) * hv).amax(-1)
    d_fut = ((fut - now[:, :, None]).norm(dim=-1) * fv).amax(-1)
    return torch.maximum(d_hist, d_fut) > threshold


def near_ego_agents(batch, slots, radius=10.0):
    """(B, M) bool: at some future step t where both are valid, the agent's true position is within radius m
    of the ego's true position at the same t. These are the agents an ego plan should matter for."""
    fut, fv = batch["agent_future"][:, slots], batch["agent_future_valid"][:, slots]       # (B, M, F, 2)
    ego, ev = batch["ego_future"][:, None, :, :2], batch["ego_future_valid"][:, None]      # (B, 1, F, 2)
    d = (fut - ego).norm(dim=-1).masked_fill(~(fv & ev), float("inf"))                      # (B, M, F)
    return d.amin(-1) <= radius


AGENT_GROUPS = ("all_agents", "moving_agents", "near_ego_agents")


def evaluate_v2(model, loader, device, plan=None, miss_threshold=2.0):
    """Val/test metrics of a v2 model as a dict.

    Target: minADE6 / minFDE6 (same metric code as v1), miss_rate_2m = share with minFDE6 > 2 m,
    top1_ADE / top1_FDE of the highest-score mode, mode_wins (share each mode is the lowest-ADE one).
    Multi-agent models add the same over all predicted agents with >= 6 valid future steps (all_agents_*)
    and over the moving ones only (moving_agents_*, > 1 m from the current position in history or future):
    ADE over valid steps, FDE at the last valid step. With an ego-plan model also near_ego_agents_* (predicted
    agents within 10 m of the ego at the same future step). plan: forwarded to the model (on/off)."""
    model.eval()
    sums, wins = {}, None

    def add(key, values):
        sums[key] = sums.get(key, 0.0) + values.sum().item()

    with torch.no_grad():
        for batch in loader:
            batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}
            traj, logits = model(batch, plan=plan) if model.ego_plan else model(batch)
            if traj.ndim == 5:                       # multi-agent: (B, M, K, F, 2), target at index 0
                future, valid, include = model.agent_targets(batch)
                moving = include & moving_agents(batch, model.slots)
                a_ade, a_fde = agent_min_errors(traj, future, valid)
                top = logits.argmax(-1)[..., None, None, None].expand(-1, -1, 1, *traj.shape[3:])
                t_ade, t_fde = agent_min_errors(traj.gather(2, top), future, valid)   # 1 mode -> top-1 errors
                groups = [("all_agents", include), ("moving_agents", moving)]
                if model.ego_plan:
                    groups.append(("near_ego_agents", include & near_ego_agents(batch, model.slots)))
                for name, mask in groups:
                    add(f"{name}_n", mask)
                    for metric, values in (("minADE6", a_ade), ("minFDE6", a_fde),
                                           ("top1_ADE", t_ade), ("top1_FDE", t_fde)):
                        add(f"{name}_{metric}", values[mask])
                traj, logits = traj[:, 0], logits[:, 0]
            target = batch["target_future"]
            ade, fde = multimodal_metrics(traj, target)
            err = (traj - target[:, None]).norm(dim=-1)                                     # (B, K, F)
            top1 = err[torch.arange(len(err), device=err.device), logits.argmax(-1)]        # (B, F)
            add("n", torch.ones_like(ade))
            add("minADE6", ade)
            add("minFDE6", fde)
            add("miss_rate_2m", (fde > miss_threshold).float())
            add("top1_ADE", top1.mean(-1))
            add("top1_FDE", top1[:, -1])
            counts = torch.bincount(err.mean(-1).argmin(-1), minlength=traj.shape[1]).cpu()
            wins = counts if wins is None else wins + counts
    n = sums.pop("n")
    out = {"n": int(n), "mode_wins": (wins / n).tolist()}
    for key, value in sums.items():
        if key.endswith("_n"):
            continue
        group = key.split("_minADE6")[0].split("_minFDE6")[0].split("_top1")[0]
        out[key] = value / (sums[f"{group}_n"] if group in AGENT_GROUPS else n)
    for group in AGENT_GROUPS:
        if f"{group}_n" in sums:
            out[f"{group}_n"] = int(sums[f"{group}_n"])
    return out


def format_v2(m):
    """One or two log lines for an evaluate_v2 dict."""
    line = (f"minADE6 {m['minADE6']:.3f} m, minFDE6 {m['minFDE6']:.3f} m, miss {m['miss_rate_2m']:.1%}, "
            f"top-1 ADE {m['top1_ADE']:.3f} / FDE {m['top1_FDE']:.3f} m")
    if "moving_agents_n" in m:
        line += (f" | moving agents ({m['moving_agents_n']}): minADE6 {m['moving_agents_minADE6']:.3f}, "
                 f"minFDE6 {m['moving_agents_minFDE6']:.3f}, top-1 ADE {m['moving_agents_top1_ADE']:.3f} / "
                 f"FDE {m['moving_agents_top1_FDE']:.3f} m | all agents ({m['all_agents_n']}): "
                 f"minADE6 {m['all_agents_minADE6']:.3f}")
    if "near_ego_agents_n" in m:
        line += (f" | near ego ({m['near_ego_agents_n']}, {m['near_ego_agents_n'] / m['n']:.2f}/sample): "
                 f"minADE6 {m['near_ego_agents_minADE6']:.3f}, minFDE6 {m['near_ego_agents_minFDE6']:.3f}")
    return line


def run_v2(args, seed, device, checkpoint):
    """Train / evaluate the v2 scene transformer (dict batches from data/processed_v2).

    With --ego-plan, val is logged with the plan on and off; checkpointing and early stopping use
    target minADE6 with the plan on (the planner always provides a plan)."""
    import time
    from torch.utils.data import DataLoader as TorchLoader
    from utils.dataset_v2 import TrajectoryDatasetV2
    from utils.models_v2 import V2Predictor

    def loader(split, shuffle):
        ds = TrajectoryDatasetV2(PROJECT_ROOT / f"data/processed_v2/nuscenes_{split}_prediction_samples_map.pkl")
        if split == "train" and args.train_fraction < 1.0:
            # Fixed permutation (seed 0, independent of --seed): the 25% subset is part of the 50% subset
            order = np.random.default_rng(0).permutation(len(ds))
            ds = torch.utils.data.Subset(ds, np.sort(order[: int(round(args.train_fraction * len(ds)))]).tolist())
        print(f"{split}: {len(ds)} samples", flush=True)
        return TorchLoader(ds, batch_size=args.batch_size, shuffle=shuffle, num_workers=4, pin_memory=True,
                           persistent_workers=True,
                           generator=torch.Generator().manual_seed(seed) if shuffle else None)

    model = V2Predictor(np.load(PROJECT_ROOT / args.anchors), assign=args.assign,
                        topology=args.topology, scale_inputs=args.scale_inputs,
                        predict_agents=args.predict_agents, ego_plan=args.ego_plan,
                        soft_scores=args.soft_scores, target_split=args.target_split).to(device)
    plans = (("plan on", True), ("plan off", False)) if args.ego_plan else (("", None),)

    def evaluate_all(loader_):
        return {label or "default": evaluate_v2(model, loader_, device, plan) for label, plan in plans}

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
            torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, args.epochs * len(train_loader) - warmup),
                                                       eta_min=1e-5),
        ], milestones=[warmup])
        print(f"Training v2 (assign={args.assign}, topology={args.topology}, scale_inputs={args.scale_inputs}, "
              f"predict_agents={args.predict_agents}, ego_plan={args.ego_plan}, soft_scores={args.soft_scores}, "
              f"target_split={args.target_split}, "
              f"anchors={args.anchors}) | {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M params | "
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
            train_time = time.time() - start
            val = evaluate_all(val_loader)
            main = next(iter(val.values()))                    # plan on (or the only one)
            val_metrics.append(main["minADE6"])
            if main["minADE6"] == min(val_metrics):
                torch.save(model.state_dict(), checkpoint)
            best_epoch = int(np.argmin(val_metrics)) + 1
            print(f"epoch {epoch:3d} | loss {loss_sum / steps:.3f} (reg {reg_sum / steps:.3f}, "
                  f"cls {cls_sum / steps:.3f}) | lr {scheduler.get_last_lr()[0]:.1e} | {train_time:.0f}s", flush=True)
            for label, m in val.items():
                print(f"          val{' ' + label if label != 'default' else ''}: {format_v2(m)}", flush=True)
            wins = main["mode_wins"]
            print("          val mode wins: " + " ".join(f"{w:.1%}" for w in wins)
                  + ("  <- DEAD MODE (<3%)" if min(wins) < 0.03 else ""), flush=True)

            # Early stopping on val minADE6; the cosine schedule stays defined over all args.epochs
            if (args.patience and epoch - best_epoch >= args.patience and epoch < args.epochs
                    and epoch >= args.patience_start):
                stopped_epoch = epoch
                print(f"Early stopping at epoch {epoch}: no val minADE6 improvement for {args.patience} epochs "
                      f"(best epoch {best_epoch}, {min(val_metrics):.3f} m)", flush=True)
                break

    model.load_state_dict(torch.load(checkpoint, map_location=device, weights_only=True), strict=True)
    validation = evaluate_all(val_loader)
    # Test is evaluated once, on the best-val checkpoint, only after training (settings are chosen on val).
    testing = evaluate_all(test_loader)
    if not args.ego_plan:                                     # keep the old flat manifest layout
        validation, testing = validation["default"], testing["default"]

    if not args.evaluate_only:
        save_run_manifest(
            checkpoint, model="v2", assign=args.assign, topology=args.topology, scale_inputs=args.scale_inputs,
            predict_agents=args.predict_agents, ego_plan=args.ego_plan, plan_drop=0.5 if args.ego_plan else None,
            soft_scores=args.soft_scores, target_split=args.target_split, patience_start=args.patience_start,
            train_fraction=args.train_fraction,
            model_class=type(model).__name__, seed=seed, num_modes=6,
            num_epochs=args.epochs, best_epoch=int(np.argmin(val_metrics)) + 1, batch_size=args.batch_size,
            patience=args.patience, stopped_epoch=stopped_epoch, epochs_trained=len(val_metrics),
            lr=args.lr, device=str(device), validation_metrics=validation, test_metrics=testing,
            optimizer="AdamW (wd 0.01), 1-epoch linear warmup + cosine to 1e-5, grad clip 1.0, dropout 0.1",
            anchors=str(args.anchors),
        )
    print("Final results | model=v2 | seed=%d" % seed)
    for label, m in (("Validation", validation), ("Test", testing)):
        for sub, mm in (m.items() if args.ego_plan else (("", m),)):
            print(f"{label}{' ' + sub if sub else ''}: {format_v2(mm)}")
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
    parser.add_argument("--train-fraction", type=float, default=1.0,
                        help="v2 only: train on this fraction of the train split (nested random subsets, seed 0)")
    parser.add_argument("--patience-start", type=int, default=0,
                        help="v2 only: no early stopping before this epoch")
    parser.add_argument("--target-split", action="store_true",
                        help="v2 multi-agent only: loss = target loss + mean of the other agents' losses")
    parser.add_argument("--anchors", default="assets/anchors.npy",
                        help="v2 only: anchor file (K, 2), relative to the project root")
    parser.add_argument("--predict-agents", type=int, default=1,
                        help="v2 only: 1 = target only; 16 = target + 15 nearest neighbours (ego slot skipped)")
    parser.add_argument("--ego-plan", action="store_true",
                        help="v2 only: ego planned future as an extra scene token (dropped with p=0.5 in training)")
    parser.add_argument("--soft-scores", action="store_true",
                        help="v2 only: score loss against softmax(-ADE_k / 1 m) instead of the one-hot winner")
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
