"""Evaluate a map-aware model by absolute future endpoint angle, plus v2 diagnostics.

Bins include endpoint displacement >=5 m. ADE and FDE independently
select their best trajectory mode. This script never trains a model.

v2 only (pass --dataset .../nuscenes_val_...pkl):
  --examples PNG     grid of example predictions, 2 per turn bin
  --follower-test D  does the ego plan reach a car following the ego? (real / braking / accelerating plan)
  --cv-baseline      constant-velocity baseline vs the model (target, moving agents, near-ego agents)
  --timing           forward-pass time for one scene with --plans different ego plans in one batch
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

from utils.graph_utils import sample_to_graph
from utils.models import (
    GraphNetworkAttentionMultimodalWithMap,
    GraphNetworkAttentionMultimodalWithMapGRU,
)
from utils.training_utils import multimodal_metrics, read_manifest, set_seed

DATASET = PROJECT_ROOT / "data/processed/nuscenes_test_prediction_samples_map.pkl"
DATASET_V2 = PROJECT_ROOT / "data/processed_v2/nuscenes_test_prediction_samples_map.pkl"
LABELS = ("0-15 deg", "15-30 deg", "30-60 deg", "60+ deg")


def endpoint_geometry(future):
    """Return final displacement in metres and absolute endpoint angle in degrees."""
    x, y = np.asarray(future, dtype=np.float64)[-1]
    return float(np.hypot(x, y)), float(abs(np.degrees(np.arctan2(y, x))))


def summarize(ades, fdes, top1=None):
    result = {"n": int(len(ades))}
    pairs = [("minADE6", ades), ("minFDE6", fdes)]
    if top1 is not None:
        pairs += [("top1_ADE", top1[0]), ("top1_FDE", top1[1])]
    for name, values in pairs:
        if len(values):
            result[name] = {
                "mean": float(np.mean(values)),
                "median": float(np.median(values)),
                "std": float(np.std(values)),
            }
        else:
            result[name] = {"mean": None, "median": None, "std": None}
    return result


def print_summary(label, summary):
    print(f"{label}: n={summary['n']}")
    if summary["n"]:
        for name in ("minADE6", "minFDE6", "top1_ADE", "top1_FDE"):
            if name not in summary:
                continue
            stats = summary[name]
            print(
                f"  {name}: mean={stats['mean']:.6f}, median={stats['median']:.6f}, "
                f"std={stats['std']:.6f} m"
            )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, help="Default: saved checkpoint seed, or 2026")
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument(
        "--model", choices=("multimodal_map", "multimodal_map_gru", "v2"),
        help="Default: saved model name, or multimodal_map_gru",
    )
    parser.add_argument("--dataset", type=Path, help="Default: the v1 or v2 test split")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--output", type=Path, help="Optional new JSON summary")
    parser.add_argument("--plot", action="store_true", help="Save a bar chart beside --output")
    parser.add_argument("--follower-test", type=Path, metavar="DIR",
                        help="v2 ego-plan model: follower test, JSON + plot of 2 examples into DIR, then exit")
    parser.add_argument("--followers", type=int, default=20, help="number of scenes for --follower-test")
    parser.add_argument("--cv-baseline", type=Path, metavar="JSON",
                        help="v2 multi-agent model: constant-velocity baseline vs the model, saved to JSON, then exit")
    parser.add_argument("--timing", action="store_true",
                        help="v2 ego-plan model: time one forward pass with one scene x --plans ego plans, then exit")
    parser.add_argument("--plans", type=int, default=50)
    parser.add_argument("--examples", type=Path,
                        help="v2 only: save a grid of example predictions (2 per turn bin) to this PNG and exit")
    args = parser.parse_args()
    if args.batch_size < 1:
        parser.error("--batch-size must be positive")
    if args.plot and not args.output:
        parser.error("--plot requires --output")
    if args.output and args.output.exists():
        parser.error(f"Output already exists: {args.output}")
    if args.plot and args.output.with_suffix(".png").exists():
        parser.error("Plot output already exists")

    seed = 2026 if args.seed is None else args.seed
    if args.checkpoint is None:
        folder = PROJECT_ROOT / "checkpoints_gru_decoder"
        if seed != 2026:
            folder = folder / f"seed{seed}"
        args.checkpoint = folder / f"multimodal_map_gru_seed{seed}_best.pth"
    manifest = {}
    if args.checkpoint.with_suffix(".json").exists():
        manifest = read_manifest(args.checkpoint)
    if args.seed is None:
        seed = manifest.get("seed", seed)
    if not 0 <= seed < 2**32:
        parser.error("--seed must be between 0 and 2**32 - 1")
    model_name = args.model or manifest.get("model", "multimodal_map_gru")
    set_seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    device = args.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device)

    if args.dataset is None:
        args.dataset = DATASET_V2 if model_name == "v2" else DATASET
    top1 = None
    if args.timing:
        timing(args, device, load_v2(args.checkpoint, device))
        return
    if args.follower_test or args.cv_baseline:
        if model_name != "v2":
            parser.error("--follower-test / --cv-baseline support --model v2 only")
        if args.follower_test:
            follower_test(args, device, load_v2(args.checkpoint, device))
        if args.cv_baseline:
            cv_baseline(args, device, load_v2(args.checkpoint, device))
        return
    if args.examples:
        if model_name != "v2":
            parser.error("--examples supports --model v2 only")
        plot_examples(args, device, seed)
        return
    if model_name == "v2":
        ades, fdes, top1, displacements, angles, invalid, n_samples = evaluate_v2(args, device)
    else:
        ades, fdes, displacements, angles, invalid, n_samples = evaluate_v1(args, model_name, device, parser)
    bin_and_report(args, model_name, seed, ades, fdes, displacements, angles, invalid, n_samples, top1)


def load_v2(checkpoint, device):
    """A v2 checkpoint with the options from its manifest; anchors come from the checkpoint (a buffer)."""
    from utils.models_v2 import V2Predictor

    manifest = read_manifest(checkpoint) if checkpoint.with_suffix(".json").exists() else {}
    state = torch.load(checkpoint, map_location=device, weights_only=True)
    model = V2Predictor(state["decoder.anchors"].cpu().numpy(),
                        topology=manifest.get("topology", False),
                        scale_inputs=manifest.get("scale_inputs", False),
                        predict_agents=manifest.get("predict_agents", 1),
                        ego_plan=manifest.get("ego_plan", False),
                        target_split=manifest.get("target_split", False),
                        plan_queries=manifest.get("plan_queries", False)).to(device)
    model.load_state_dict(state, strict=True)
    return model.eval()


def evaluate_v2(args, device):
    """Per-sample minADE6 / minFDE6 and top-1 (highest-score mode) ADE / FDE of a v2 checkpoint,
    plus endpoint geometry."""
    from torch.utils.data import DataLoader as TorchLoader
    from utils.dataset_v2 import TrajectoryDatasetV2
    from utils.models_v2 import V2Predictor

    dataset = TrajectoryDatasetV2(args.dataset)
    geometry = [endpoint_geometry(sample["target_future"]) for sample in dataset.samples]
    manifest = read_manifest(args.checkpoint) if args.checkpoint.with_suffix(".json").exists() else {}
    model = load_v2(args.checkpoint, device)
    model.eval()
    ades, fdes, top_ades, top_fdes = [], [], [], []
    with torch.no_grad():
        for batch in TorchLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=4):
            batch = {k: v.to(device) for k, v in batch.items()}
            traj, logits = model(batch)
            if traj.ndim == 5:                       # multi-agent model: keep the target (index 0)
                traj, logits = traj[:, 0], logits[:, 0]
            ade, fde = multimodal_metrics(traj, batch["target_future"])
            ades.extend(ade.cpu().tolist())
            fdes.extend(fde.cpu().tolist())
            best = traj[torch.arange(len(traj)), logits.argmax(-1)]           # (B, 12, 2)
            err = torch.linalg.norm(best - batch["target_future"], dim=-1)    # (B, 12)
            top_ades.extend(err.mean(-1).cpu().tolist())
            top_fdes.extend(err[:, -1].cpu().tolist())
    displacements, angles = zip(*geometry)
    top1 = (np.asarray(top_ades), np.asarray(top_fdes))
    return np.asarray(ades), np.asarray(fdes), top1, list(displacements), list(angles), [], len(dataset)


def plot_examples(args, device, seed, per_bin=2):
    """Lanes, agent histories, ground truth and the 6 predicted modes for a few samples per turn bin."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from utils.dataset_v2 import TrajectoryDatasetV2
    from utils.models_v2 import V2Predictor

    dataset = TrajectoryDatasetV2(args.dataset)
    manifest = read_manifest(args.checkpoint) if args.checkpoint.with_suffix(".json").exists() else {}
    model = load_v2(args.checkpoint, device)
    model.eval()

    geometry = np.array([endpoint_geometry(s["target_future"]) for s in dataset.samples])
    moving, angles = geometry[:, 0] >= 5.0, geometry[:, 1]
    edges = ((0, 15), (15, 30), (30, 60), (60, 361))
    rng = np.random.default_rng(seed)
    picks = [(label, i) for label, (lo, hi) in zip(LABELS, edges)
             for i in rng.choice(np.flatnonzero(moving & (angles >= lo) & (angles < hi)), per_bin, replace=False)]

    fig, axes = plt.subplots(per_bin, len(LABELS), figsize=(5 * len(LABELS), 5 * per_bin))
    for ax, (label, idx) in zip(axes.T.flat, picks):
        item = dataset[int(idx)]
        batch = {k: v[None].to(device) for k, v in item.items()}
        with torch.no_grad():
            traj, logits = model(batch)
        if traj.ndim == 5:
            traj, logits = traj[:, 0], logits[:, 0]
        traj, prob = traj[0].cpu().numpy(), logits[0].softmax(-1).cpu().numpy()
        gt = item["target_future"].numpy()

        for pts, valid in zip(item["lane_points"].numpy()[item["lane_mask"].numpy()],
                              item["lane_point_valid"].numpy()[item["lane_mask"].numpy()]):
            ax.plot(pts[valid, 0], pts[valid, 1], color="0.8", lw=1, zorder=1)
        hist, hvalid = item["agent_hist"].numpy(), item["agent_hist_valid"].numpy()
        for a in np.flatnonzero(item["agent_mask"].numpy())[1:]:
            h = hist[a][hvalid[a]]
            if len(h):
                ax.plot(h[:, 0], h[:, 1], color="tab:purple" if a == 1 else "0.45", lw=1, zorder=2)
                ax.plot(h[-1, 0], h[-1, 1], "s", ms=3, color="tab:purple" if a == 1 else "0.45", zorder=2)
        h = hist[0][hvalid[0]]
        ax.plot(h[:, 0], h[:, 1], color="tab:blue", lw=2, zorder=3)
        ax.plot(0, 0, "o", color="tab:blue", ms=6, zorder=5)

        best_score = prob.argmax()
        best_ade = np.linalg.norm(traj - gt, axis=-1).mean(-1).argmin()
        for k in np.argsort(prob):
            ax.plot(traj[k, :, 0], traj[k, :, 1], color="tab:orange", lw=1 + 3 * prob[k],
                    alpha=0.35 + 0.65 * prob[k] / prob.max(), zorder=4,
                    ls="-" if k == best_score else "--")
            ax.annotate(f"{prob[k]:.2f}", traj[k, -1], fontsize=7, color="tab:orange")
        ax.plot(gt[:, 0], gt[:, 1], color="tab:green", lw=2.5, marker=".", ms=4, zorder=6)

        ade = np.linalg.norm(traj[best_ade] - gt, axis=-1).mean()
        top1 = np.linalg.norm(traj[best_score] - gt, axis=-1).mean()
        centre = np.concatenate([gt, traj.reshape(-1, 2), [[0, 0]]])
        mid, span = (centre.max(0) + centre.min(0)) / 2, max(np.ptp(centre, 0).max() / 2 + 8, 20)
        ax.set_xlim(mid[0] - span, mid[0] + span)
        ax.set_ylim(mid[1] - span, mid[1] + span)
        ax.set_aspect("equal")
        ax.grid(alpha=0.2)
        ax.set_title(f"{label} | #{idx}\nminADE6 {ade:.2f} m, top-1 ADE {top1:.2f} m", fontsize=10)

    handles = [plt.Line2D([], [], color=c, lw=w, ls=ls) for c, w, ls in
               (("tab:green", 2.5, "-"), ("tab:orange", 3, "-"), ("tab:orange", 1.5, "--"),
                ("tab:blue", 2, "-"), ("0.45", 1, "-"), ("tab:purple", 1, "-"), ("0.8", 1, "-"))]
    fig.legend(handles, ["ground truth (6 s)", "top-score mode", "other modes (width/number = prob.)",
                         "target history (2 s)", "other agents", "ego", "lane segments"],
               loc="lower center", ncol=7, fontsize=9)
    fig.suptitle(f"v2 predictions on {args.dataset.name} | {args.checkpoint.name} | target frame, heading +x")
    fig.tight_layout(rect=(0, 0.04, 1, 0.97))
    args.examples.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.examples, dpi=120)
    print(f"Saved {args.examples}")


def evaluate_v1(args, model_name, device, parser):
    """Per-sample minADE6 / minFDE6 of a v1 map-aware checkpoint, plus endpoint geometry."""
    with args.dataset.open("rb") as handle:
        samples = pickle.load(handle)
    graphs, displacements, angles, invalid = [], [], [], []
    for index, sample in enumerate(samples):
        future = np.asarray(sample["target_future"])
        if future.shape != (12, 2) or not np.isfinite(future).all():
            invalid.append(index)
            continue
        if "map_lanes" not in sample:
            raise ValueError(f"Sample {index} has no map_lanes; use a map-aware dataset")
        graph = sample_to_graph(sample)
        if not all(torch.isfinite(t).all() for t in (graph.x, graph.edge_attr, graph.map_points)):
            invalid.append(index)
            continue
        displacement, angle = endpoint_geometry(future)
        graphs.append(graph)
        displacements.append(displacement)
        angles.append(angle)
    if not graphs:
        raise ValueError("No valid samples")
    loader = DataLoader(graphs, batch_size=args.batch_size, shuffle=False)

    if model_name == "multimodal_map":
        model = GraphNetworkAttentionMultimodalWithMap(num_modes=6)
    elif model_name == "multimodal_map_gru":
        model = GraphNetworkAttentionMultimodalWithMapGRU(num_modes=6)
    else:
        parser.error("Expected a map-aware MLP or GRU checkpoint")
    model = model.to(device)
    model.load_state_dict(
        torch.load(args.checkpoint, map_location=device, weights_only=True),
        strict=True,
    )
    model.eval()
    ades, fdes = [], []
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            ade, fde = multimodal_metrics(model(batch), batch.y.reshape(-1, 12, 2))
            ades.extend(ade.cpu().tolist())
            fdes.extend(fde.cpu().tolist())
    return np.asarray(ades), np.asarray(fdes), displacements, angles, invalid, len(samples)


def bin_and_report(args, model_name, seed, ades, fdes, displacements, angles, invalid, n_samples, top1=None):
    def sub(mask):
        return None if top1 is None else (top1[0][mask], top1[1][mask])

    if not (np.isfinite(ades).all() and np.isfinite(fdes).all()):
        raise ValueError("Non-finite prediction errors")

    moving = np.asarray(displacements) >= 5.0
    angles = np.asarray(angles)
    masks = (
        moving & (angles < 15),
        moving & (angles >= 15) & (angles < 30),
        moving & (angles >= 30) & (angles < 60),
        moving & (angles >= 60),
    )
    bins = {}
    for label, mask in zip(LABELS, masks):
        bins[label] = summarize(ades[mask], fdes[mask], sub(mask))
    result = {
        "model": model_name,
        "seed": seed,
        "checkpoint": str(args.checkpoint),
        "dataset": str(args.dataset),
        "batch_size": args.batch_size,
        "angle_definition": "abs(degrees(atan2(y_final, x_final))) in target-local frame",
        "minimum_endpoint_displacement_m": 5.0,
        "std_ddof": 0,
        "input_samples": n_samples,
        "invalid_sample_indices": invalid,
        "excluded_displacement_below_5m": int((~moving).sum()),
        "full_test": summarize(ades, fdes, top1),
        "non_stationary": summarize(ades[moving], fdes[moving], sub(moving)),
        "turn_bins": bins,
    }
    print(f"Model={model_name} | seed={seed} | skipped invalid samples={len(invalid)}")
    print_summary("Full dataset", result["full_test"])
    print_summary("Displacement >=5 m", result["non_stationary"])
    for label, summary in bins.items():
        print_summary(label, summary)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
        print(f"Saved {args.output}")
    if args.plot:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        means = [bins[label]["minADE6"]["mean"] for label in LABELS]
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.bar(LABELS, [np.nan if value is None else value for value in means])
        ax.set(
            xlabel="Absolute endpoint angle",
            ylabel="Mean minADE6 (m)",
            title=f"Seed {seed}; displacement ≥5 m",
        )
        fig.tight_layout()
        fig.savefig(args.output.with_suffix(".png"), dpi=160)
        plt.close(fig)


# ============================================================
# v2 diagnostics: follower test and constant-velocity baseline
# ============================================================

def v2_val_items(args):
    from utils.dataset_v2 import TrajectoryDatasetV2
    return TrajectoryDatasetV2(args.dataset)


def retime_plan(ego_now, heading, future, v0, accel, steps=12, dt=0.5):
    """Ego plan (F, 4) [x, y, cos, sin] along the real planned path, re-timed to constant acceleration from v0.
    accel < 0 brakes to a stop (and stays there). Past the end of the real path it continues straight."""
    path = np.vstack([ego_now, future[:, :2]])
    seg = np.diff(path, axis=0)
    seg_len = np.linalg.norm(seg, axis=1)
    cum = np.concatenate([[0.0], np.cumsum(seg_len)])
    last_dir = seg[seg_len > 1e-3][-1] / seg_len[seg_len > 1e-3][-1] if (seg_len > 1e-3).any() \
        else np.array([np.cos(heading), np.sin(heading)])
    out = np.zeros((steps, 4), np.float32)
    for k in range(steps):
        t = dt * (k + 1)
        if accel < 0:
            t_stop = v0 / -accel
            tt = min(t, t_stop)
            dist = v0 * tt + 0.5 * accel * tt ** 2
        else:
            dist = v0 * t + 0.5 * accel * t ** 2
        if dist <= cum[-1]:
            i = min(np.searchsorted(cum, dist, side="right") - 1, len(seg) - 1)
            frac = (dist - cum[i]) / max(seg_len[i], 1e-6)
            pos = path[i] + frac * seg[i]
            d = seg[i] / seg_len[i] if seg_len[i] > 1e-3 else last_dir
        else:
            pos, d = path[-1] + (dist - cum[-1]) * last_dir, last_dir
        out[k] = [pos[0], pos[1], d[0], d[1]]
    return out


def follower_test(args, device, model, brake=-4.0, accel=2.0):
    """Does the ego plan reach a car following the ego?

    Follower: a vehicle in a predicted slot, behind the ego in the same lane: in the ego frame 2-15 m behind,
    |lateral| <= 1.5 m, heading within 20 deg; ego moving >= 3 m/s. One follower per val scene (first match).
    Three ego plans along the real planned path: real timing, braking to a stop (-4 m/s^2), accelerating
    (+2 m/s^2); plus plan off. Reported per plan: the follower's probability-weighted predicted travel at 6 s,
    and the probability-weighted mean distance between its predicted modes and those under the real plan."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from utils.models_v2 import token_poses

    if not (model.ego_plan and model.predict_agents > 1):
        raise SystemExit("--follower-test needs a multi-agent ego-plan checkpoint")
    ds = v2_val_items(args)
    found, scenes = [], set()
    for idx, sample in enumerate(ds.samples):
        if len(found) == args.followers:
            break
        if sample["scene_token"] in scenes:
            continue
        item = ds[idx]
        batch = {k: v[None] for k, v in item.items()}
        pose = token_poses(batch)[0][0].numpy()                              # (A, 3)
        hist, hv = item["agent_hist"].numpy(), item["agent_hist_valid"].numpy()
        ego = pose[1]
        v0 = float(np.hypot(*hist[1, -1, 4:6]))
        if not hv[1, -1] or v0 < 3.0:
            continue
        c, s_ = np.cos(ego[2]), np.sin(ego[2])
        for j, slot in enumerate(model.slots):
            if not item["agent_mask"][slot] or not hv[slot, -1] or int(item["agent_type"][slot]) not in (0, 1):
                continue
            d = pose[slot, :2] - ego[:2]
            x_rel, y_rel = c * d[0] + s_ * d[1], -s_ * d[0] + c * d[1]
            dth = np.degrees(np.angle(np.exp(1j * (pose[slot, 2] - ego[2]))))
            if -15 <= x_rel <= -2 and abs(y_rel) <= 1.5 and abs(dth) <= 20:
                found.append((idx, j, slot, v0, float(x_rel)))
                scenes.add(sample["scene_token"])
                break
    print(f"follower test: {len(found)} val scenes with a follower (wanted {args.followers})")

    plans = ("real", "brake", "accel", "off")
    rows, keep = [], []
    for idx, j, slot, v0, gap in found:
        item = ds[idx]
        batch = {k: v[None].to(device) for k, v in item.items()}
        ego_now = item["agent_hist"][1, -1, :2].numpy()
        heading = float(np.arctan2(item["agent_hist"][1, -1, 3], item["agent_hist"][1, -1, 2]))
        real = item["ego_future"].numpy()
        variants = {"real": real, "brake": retime_plan(ego_now, heading, real, v0, brake),
                    "accel": retime_plan(ego_now, heading, real, v0, accel)}
        preds = {}
        with torch.no_grad():
            for name in plans:
                b = dict(batch)
                if name != "off":
                    b["ego_future"] = torch.as_tensor(variants[name], device=device)[None]
                traj, logits = model(b, plan=name != "off")
                preds[name] = (traj[0, j].cpu().numpy(), logits[0, j].softmax(-1).cpu().numpy())
        now = item["agent_hist"][slot, -1, :2].numpy()
        row = {"sample": idx, "slot": slot, "ego_speed": v0, "gap_m": gap}
        traj_real, p_real = preds["real"]
        for name in plans:
            traj, p = preds[name]
            row[f"travel_{name}"] = float((np.linalg.norm(traj[:, -1] - now, axis=-1) * p).sum())
            row[f"shift_{name}"] = float((np.linalg.norm(traj - traj_real, axis=-1).mean(-1) * p_real).sum())
        rows.append(row)
        keep.append((item, slot, j, variants, preds))

    summary = {name: {"travel_6s_m": float(np.mean([r[f"travel_{name}"] for r in rows])),
                      "shift_vs_real_m": float(np.mean([r[f"shift_{name}"] for r in rows]))} for name in plans}
    print(f"ego speed {np.mean([r['ego_speed'] for r in rows]):.1f} m/s, gap {np.mean([r['gap_m'] for r in rows]):.1f} m (means)")
    print(" plan  | follower predicted travel at 6 s | mean shift vs real-plan prediction")
    for name in plans:
        print(f" {name:5s} | {summary[name]['travel_6s_m']:31.2f} m | {summary[name]['shift_vs_real_m']:.3f} m")
    ego_travel = {name: float(np.mean([np.linalg.norm(k[3][name][-1, :2] - k[0]['agent_hist'][1, -1, :2].numpy())
                                       for k in keep])) for name in ("real", "brake", "accel")}
    print(" ego plan travel at 6 s: " + ", ".join(f"{k} {v:.1f} m" for k, v in ego_travel.items()))

    out = args.follower_test
    out.mkdir(parents=True, exist_ok=True)
    (out / "follower_test.json").write_text(json.dumps(
        {"checkpoint": str(args.checkpoint), "dataset": str(args.dataset), "brake_mps2": brake, "accel_mps2": accel,
         "summary": summary, "ego_plan_travel_6s_m": ego_travel, "followers": rows}, indent=2) + "\n")

    colors = {"real": "tab:green", "brake": "tab:red", "accel": "tab:blue", "off": "0.5"}
    fig, axes = plt.subplots(1, 2, figsize=(14, 7))
    for ax, r, (item, slot, j, variants, preds) in zip(axes, rows, keep):
        for pts, valid in zip(item["lane_points"].numpy(), item["lane_point_valid"].numpy()):
            if valid.any():
                ax.plot(pts[valid, 0], pts[valid, 1], color="0.88", lw=1, zorder=1)
        ego_now = item["agent_hist"][1, -1, :2].numpy()
        ax.plot(*ego_now, "s", color="tab:purple", ms=9, zorder=6, label="ego")
        for name, plan in variants.items():
            ax.plot(plan[:, 0], plan[:, 1], color=colors[name], lw=2.5, ls=":", zorder=3)
        h = item["agent_hist"][slot].numpy()[item["agent_hist_valid"][slot].numpy()]
        ax.plot(h[:, 0], h[:, 1], color="k", lw=2, zorder=4)
        ax.plot(*h[-1, :2], "o", color="k", ms=7, zorder=6, label="follower")
        fut = item["agent_future"][slot].numpy()[item["agent_future_valid"][slot].numpy()]
        ax.plot(fut[:, 0], fut[:, 1], color="k", lw=1, ls="--", zorder=4)
        for name in ("real", "brake", "accel"):
            traj, p = preds[name]
            for k in np.argsort(p)[-3:]:
                ax.plot(traj[k, :, 0], traj[k, :, 1], color=colors[name], lw=0.8 + 3 * p[k], alpha=0.8, zorder=5)
        pts = np.concatenate([fut, h[:, :2], *[v[:, :2] for v in variants.values()],
                              *[preds[n][0].reshape(-1, 2) for n in ("real", "brake", "accel")]])
        mid, span = (pts.max(0) + pts.min(0)) / 2, np.ptp(pts, 0).max() / 2 + 8
        ax.set_xlim(mid[0] - span, mid[0] + span)
        ax.set_ylim(mid[1] - span, mid[1] + span)
        ax.set_aspect("equal")
        ax.grid(alpha=0.2)
        ax.set_title(f"val #{r['sample']}: ego {r['ego_speed']:.1f} m/s, follower {-r['gap_m']:.1f} m behind")
    handles = [plt.Line2D([], [], color=colors[n], lw=2.5, ls=":") for n in ("real", "brake", "accel")] + \
              [plt.Line2D([], [], color=colors[n], lw=2.5) for n in ("real", "brake", "accel")] + \
              [plt.Line2D([], [], color="k", lw=1, ls="--")]
    fig.legend(handles, ["ego plan: real", "ego plan: brake to stop", "ego plan: accelerate",
                         "follower pred. (real plan)", "follower pred. (brake)", "follower pred. (accelerate)",
                         "follower ground truth"], loc="lower center", ncol=4, fontsize=9)
    fig.suptitle("Follower test: top-3 predicted modes of the car behind the ego under three ego plans")
    fig.tight_layout(rect=(0, 0.07, 1, 0.96))
    fig.savefig(out / "follower_test.png", dpi=120)
    print(f"saved {out / 'follower_test.json'} and {out / 'follower_test.png'}")


def cv_baseline(args, device, model):
    """Constant velocity from the last valid history step vs the model (plan on), on val.
    CV is one guess, so it is compared with the model's top-1 (and minADE6 for reference)."""
    from torch.utils.data import DataLoader as TorchLoader
    sys.path.insert(0, str(PROJECT_ROOT / "scripts"))
    from train_model import agent_min_errors, evaluate_v2 as evaluate_v2_dict, moving_agents, near_ego_agents
    from utils.models_v2 import token_poses

    if model.predict_agents == 1:
        raise SystemExit("--cv-baseline needs a multi-agent checkpoint")
    loader = TorchLoader(v2_val_items(args), batch_size=args.batch_size, shuffle=False, num_workers=4)
    sums = {}
    with torch.no_grad():
        for batch in loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            hist, hv = batch["agent_hist"][:, model.slots], batch["agent_hist_valid"][:, model.slots]
            T = hist.shape[2]
            last = (hv * torch.arange(1, T + 1, device=device)).argmax(-1)                     # (B, M)
            state = hist.gather(2, last[..., None, None].expand(-1, -1, 1, hist.shape[-1]))[:, :, 0]
            t = 0.5 * ((T - 1 - last)[..., None] + torch.arange(1, 13, device=device))          # (B, M, F)
            cv = state[..., None, :2] + t[..., None] * state[..., None, 4:6]                    # (B, M, F, 2)
            future, valid, include = model.agent_targets(batch)
            ade, fde = agent_min_errors(cv[:, :, None], future, valid)
            groups = {"target": include[:, :1],
                      "moving_agents": include & moving_agents(batch, model.slots),
                      "near_ego_agents": include & near_ego_agents(batch, model.slots)}
            for g, mask in groups.items():
                a, f = (ade[:, :1], fde[:, :1]) if g == "target" else (ade, fde)
                sums[g] = [x + y for x, y in zip(sums.get(g, [0, 0, 0]),
                                                 (a[mask].sum().item(), f[mask].sum().item(), mask.sum().item()))]
    m = evaluate_v2_dict(model, loader, device, plan=True if model.ego_plan else None)
    keys = {"target": ("minADE6", "minFDE6", "top1_ADE", "top1_FDE"),
            "moving_agents": tuple(f"moving_agents_{k}" for k in ("minADE6", "minFDE6", "top1_ADE", "top1_FDE")),
            "near_ego_agents": tuple(f"near_ego_agents_{k}" for k in ("minADE6", "minFDE6", "top1_ADE", "top1_FDE"))}
    result = {}
    print(" group           |    n   | CV ADE / FDE   | model top-1 ADE / FDE | model minADE6 / minFDE6")
    for g, (a, f, n) in sums.items():
        mk = keys[g]
        result[g] = {"n": int(n), "cv_ADE": a / n, "cv_FDE": f / n, "model_top1_ADE": m[mk[2]],
                     "model_top1_FDE": m[mk[3]], "model_minADE6": m[mk[0]], "model_minFDE6": m[mk[1]]}
        r = result[g]
        print(f" {g:15s} | {r['n']:6d} | {r['cv_ADE']:5.2f} / {r['cv_FDE']:5.2f} | {r['model_top1_ADE']:10.2f} / "
              f"{r['model_top1_FDE']:5.2f}    | {r['model_minADE6']:6.2f} / {r['model_minFDE6']:5.2f}")
    args.cv_baseline.parent.mkdir(parents=True, exist_ok=True)
    args.cv_baseline.write_text(json.dumps({"checkpoint": str(args.checkpoint), "dataset": str(args.dataset),
                                            "results": result}, indent=2) + "\n")
    print(f"saved {args.cv_baseline}")


def timing(args, device, model, index=0, warmup=5, runs=20):
    """One val scene repeated --plans times, each copy with a different ego plan (the real path re-timed with
    accelerations from -4 to +2 m/s^2), in one batch. Median wall time of a no-grad forward over `runs` runs."""
    import time
    item = v2_val_items(args)[index]
    ego_now = item["agent_hist"][1, -1, :2].numpy()
    heading = float(np.arctan2(item["agent_hist"][1, -1, 3], item["agent_hist"][1, -1, 2]))
    v0 = float(np.hypot(*item["agent_hist"][1, -1, 4:6]))
    plans = np.stack([retime_plan(ego_now, heading, item["ego_future"].numpy(), v0, a)
                      for a in np.linspace(-4.0, 2.0, args.plans)])
    for n in (args.plans, 1):
        batch = {k: v[None].expand(n, *v.shape).contiguous().to(device) for k, v in item.items()}
        batch["ego_future"] = torch.as_tensor(plans[:n], device=device)
        times = []
        with torch.no_grad():
            for r in range(warmup + runs):
                if device.type == "cuda":
                    torch.cuda.synchronize()
                t0 = time.perf_counter()
                traj, logits = model(batch, plan=True)
                if device.type == "cuda":
                    torch.cuda.synchronize()
                if r >= warmup:
                    times.append(1000 * (time.perf_counter() - t0))
        spread = float(traj[:, 0].std(0).mean()) if n > 1 else 0.0
        print(f"batch {n:3d} (1 scene x {n} ego plans) on {device}: median {np.median(times):.2f} ms "
              f"(min {np.min(times):.2f}, max {np.max(times):.2f}) over {runs} runs | output {tuple(traj.shape)}"
              + (f" | target prediction spread across plans {spread:.2f} m" if n > 1 else ""))


if __name__ == "__main__":
    main()
