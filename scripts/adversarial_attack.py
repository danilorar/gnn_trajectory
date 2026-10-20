"""White-box PGD attack on the target vehicle's observed trajectory.

Follows Zhang et al., "On Adversarial Robustness of Trajectory Prediction for
Autonomous Vehicles" (CVPR 2022) and their code
(github.com/zqzqz/AdvTrajectoryPrediction, prediction/attack/{gradient,
constraint,loss}.py), adapted to the issue #18 metrics:

- perturb all 5 observed target positions; random init in [-1, 1] m,
  10 restarts x 100 plain gradient steps (lr 0.1, or 0.01 for ade/fde)
- after every step, scale the perturbation by the largest theta in
  {1.00, 0.99, ..., 0.02} that keeps every point within 1 m and the trace
  within the nuScenes physical bounds from the original code; the trace
  checked also includes the next 3 ground-truth points, as the paper text
  describes (theta = 0 if no theta passes)
- the perturbed current position becomes the new frame origin, so every
  node and lane-centreline point is re-centred and edge attributes are
  recomputed differentiably (the set of lanes within 50 m stays fixed)
- objectives: ade / fde (on minADE6 / minFDE6, squared as in the original
  loss) and left / right / front / rear deviation of the min-ADE candidate

Map-aware models read the *_map datasets, which hold the same samples in the
same order as the plain ones, so the same --num-samples draw is comparable.

Example usage:
    python scripts/adversarial_attack.py
    python scripts/adversarial_attack.py --models multimodal:42 --num-samples 100
    python scripts/adversarial_attack.py --models multimodal:42 --tag _seed42   # one run per process
    python scripts/adversarial_attack.py --models multimodal_map_gru:42 --tag _map_gru_seed42
    python scripts/adversarial_attack.py --merge data/processed/adversarial/results_val_b3_seed*.json
"""

import argparse
import copy
import json
import pickle
import sys
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch_geometric.loader import DataLoader


PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


from utils.experiment_utils import sha256
from utils.graph_utils import sample_to_graph
from utils.baselines import ConstantVelocityBaseline
from utils.models import (
    GraphNetworkAttention,
    GraphNetworkAttentionMultimodal,
    GraphNetworkAttentionMultimodalWithMap,
    GraphNetworkAttentionMultimodalWithMapGRU,
    TargetHistoryMLP,
)

OBJECTIVES = ("ade", "fde", "left", "right", "front", "rear")
DIRECTIONS = ("left", "right", "front", "rear")

# nuScenes bounds from the original code (prediction/dataset/nuscenes.py),
# in metres per 0.5 s step (velocity), per step^2 (acceleration), per step^3 (jerk)
PHYSICAL_BOUNDS = {
    "scalar_v": 8.599,
    "linear_a": 1.275,
    "rotate_a": 0.468,
    "linear_aa": 1.957,
    "rotate_aa": 0.535,
}
DEVIATION_BOUND = 1.0   # metres per observed point
HALF_LANE_WIDTH = 1.85  # metres; the paper's real-world-impact threshold
THETAS = torch.arange(100, 1, -1, dtype=torch.float32) / 100  # 1.00 ... 0.02

MODEL_CLASSES = {
    "multimodal": lambda: GraphNetworkAttentionMultimodal(num_modes=6),
    "attention": GraphNetworkAttention,
    "target_mlp": TargetHistoryMLP,
    "multimodal_map": GraphNetworkAttentionMultimodalWithMap,
    "multimodal_map_gru": GraphNetworkAttentionMultimodalWithMapGRU,
}
MAP_MODELS = {"multimodal_map", "multimodal_map_gru"}
CHECKPOINT_DIRS = [PROJECT_ROOT / d for d in ("checkpoints", "checkpoints_60ep", "checkpoints_gru_decoder")]

# Reference categorical palette (light mode) and chart ink
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
INK, INK_2, MUTED, GRID, AXIS = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
SURFACE = "#fcfcfb"

# ========================================================
# MODEL INPUTS
# ========================================================

def uses_map(specs):
    return any(spec.split(":")[0] in MAP_MODELS for spec in specs)


def checkpoint_path(spec, checkpoint_dirs):
    """The one '<name>_seed<seed>_best.pth' under the given dirs (any depth); None for 'cv'."""
    if spec == "cv":
        return None
    name, seed = spec.split(":")
    matches = [p for d in checkpoint_dirs if d.is_dir() for p in d.rglob(f"{name}_seed{seed}_best.pth")]
    if len(matches) != 1:
        raise FileNotFoundError(f"{spec}: expected one checkpoint, found {[str(p) for p in matches]}")
    return matches[0]


def load_model(spec, checkpoint_dirs):
    """spec is 'cv' or '<name>:<seed>', e.g. 'multimodal:42'."""
    if spec == "cv":
        return ConstantVelocityBaseline()
    model = MODEL_CLASSES[spec.split(":")[0]]()
    path = checkpoint_path(spec, checkpoint_dirs)
    model.load_state_dict(torch.load(path, map_location="cpu", weights_only=True))
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model


def perturbed_forward(model, batch, delta):
    """Predict with the target's observed positions shifted by delta [B, 5, 2].

    Returns predictions [B, K, 12, 2] in the original (clean) target frame.
    """
    pos = batch.x[:, :10].view(-1, 5, 2)
    target_idx = torch.where(batch.target_mask)[0]

    # Perturb the target nodes, then re-centre each graph on the perturbed current position
    add = torch.zeros_like(pos).index_add(0, target_idx, delta)
    shift = delta[:, -1, :][batch.batch]
    new_pos = pos + add - shift[:, None, :]

    current = new_pos[:, -1, :]
    src, dst = batch.edge_index
    rel = current[dst] - current[src]
    dist = torch.sqrt((rel ** 2).sum(dim=1, keepdim=True) + 1e-12)

    perturbed = copy.copy(batch)
    perturbed.x = torch.cat([new_pos.reshape(-1, 10), batch.x[:, 10:]], dim=1)
    perturbed.edge_attr = torch.cat([rel, dist], dim=1)

    # Lane points stay fixed in the world, so they move by the same re-centring shift
    lane_graph = torch.repeat_interleave(torch.arange(len(batch.num_map_lanes)), batch.num_map_lanes)
    perturbed.map_points = batch.map_points - delta[:, -1, :][lane_graph[batch.map_lane_index]]

    pred = model(perturbed)
    if pred.ndim == 3:
        pred = pred[:, None]
    return pred + delta[:, -1, None, None, :]

# ========================================================
# CONSTRAINTS (prediction/attack/constraint.py)
# ========================================================

def unit(v):
    return v / (v.norm(dim=-1, keepdim=True) + 0.001)


def physical_ok(trace):
    """Boolean [...] mask: does each trace [..., T, 2] satisfy PHYSICAL_BOUNDS?"""
    v = trace[..., 1:, :] - trace[..., :-1, :]
    a = v[..., 1:, :] - v[..., :-1, :]
    aa = a[..., 1:, :] - a[..., :-1, :]
    d = unit(v)
    d_r = torch.stack([d[..., 1], -d[..., 0]], dim=-1)

    checks = [
        (v.norm(dim=-1), "scalar_v"),
        ((d[..., :-1, :] * a).sum(-1).abs(), "linear_a"),
        ((d_r[..., :-1, :] * a).sum(-1).abs(), "rotate_a"),
        ((d[..., :-2, :] * aa).sum(-1).abs(), "linear_aa"),
        ((d_r[..., :-2, :] * aa).sum(-1).abs(), "rotate_aa"),
    ]
    ok = torch.ones(trace.shape[:-2], dtype=torch.bool)
    for values, key in checks:
        ok &= (values <= PHYSICAL_BOUNDS[key]).all(dim=-1)
    return ok


def constraint_theta(clean_past, delta, future_head):
    """Largest theta per sample keeping theta * delta feasible; 0 if none passes."""
    scaled = THETAS[:, None, None, None] * delta[None]                      # [G, B, 5, 2]
    merged = torch.cat([clean_past[None] + scaled,
                        future_head[None].expand(len(THETAS), -1, -1, -1)], dim=2)
    ok = physical_ok(merged) & (scaled.norm(dim=-1) <= DEVIATION_BOUND).all(dim=-1)
    first = torch.where(ok.any(dim=0), ok.float().argmax(dim=0), torch.full_like(ok[0], -1, dtype=torch.long))
    return torch.where(first >= 0, THETAS[first.clamp(min=0)], torch.zeros(()))

# ========================================================
# METRICS AND LOSSES (issue #18 / prediction/attack/loss.py)
# ========================================================

def motion_directions(future):
    """Unit longitudinal and left vectors of the ground-truth motion at each step."""
    prev = torch.cat([torch.zeros_like(future[:, :1]), future[:, :-1]], dim=1)
    forward = unit(future - prev)
    left = torch.stack([-forward[..., 1], forward[..., 0]], dim=-1)
    return forward, left


def sample_metrics(pred, future, forward, left):
    """Per-sample minADE, minFDE and directional deviations of the min-ADE mode."""
    err = (pred - future[:, None]).norm(dim=-1)                # [B, K, 12]
    ade_k = err.mean(dim=-1)
    best = ade_k.argmin(dim=1)
    offset = pred[torch.arange(len(pred)), best] - future     # [B, 12, 2]
    front = (offset * forward).sum(-1).mean(-1)
    lateral = (offset * left).sum(-1).mean(-1)
    return {
        "ade": ade_k.min(dim=1).values,
        "fde": err[..., -1].min(dim=1).values,
        "left": lateral, "right": -lateral,
        "front": front, "rear": -front,
    }


def attack_loss(pred, future, forward, left, objective):
    """Per-sample loss to minimise (negative attack goal)."""
    if objective == "ade":
        return -((pred - future[:, None]) ** 2).sum(-1).mean(-1).min(dim=1).values
    if objective == "fde":
        return -((pred[:, :, -1] - future[:, None, -1]) ** 2).sum(-1).min(dim=1).values
    return -sample_metrics(pred, future, forward, left)[objective]

# ========================================================
# PGD (prediction/attack/gradient.py)
# ========================================================

def pgd_attack(model, batch, clean_past, future, future_head, objective, restarts, iters, generator):
    forward, left = motion_directions(future)
    lr = 0.01 if objective in ("ade", "fde") else 0.1

    # Start from the unperturbed trajectory so an attack never ends below clean
    best_delta = torch.zeros_like(clean_past)
    with torch.no_grad():
        best_loss = attack_loss(perturbed_forward(model, batch, best_delta), future, forward, left, objective)

    for _ in range(restarts):
        value = (torch.rand(clean_past.shape, generator=generator) * 2 - 1) * DEVIATION_BOUND
        for _ in range(iters):
            theta = constraint_theta(clean_past, value, future_head)
            value.requires_grad_(True)
            delta = value * theta[:, None, None]
            loss = attack_loss(perturbed_forward(model, batch, delta), future, forward, left, objective)

            improved = loss.detach() < best_loss
            best_loss = torch.where(improved, loss.detach(), best_loss)
            best_delta[improved] = delta.detach()[improved]

            grad, = torch.autograd.grad(loss.sum(), value)
            value = (value - lr * grad).detach()

    return best_delta


def random_control(model, batch, clean_past, future, future_head, target_size, draws=20, seed=7):
    """Smooth random perturbations (offset + linear drift) of the attack's mean size.

    Same feasibility scaling as the attack; separates the effect of the
    adversarial direction from mere sensitivity to perturbations this large.
    """
    forward, left = motion_directions(future)
    steps = torch.linspace(-1, 0, clean_past.shape[1])[None, :, None]
    generator = torch.Generator().manual_seed(seed)

    def draw(scale):
        n = len(clean_past)
        raw = (torch.randn(n, 1, 2, generator=generator)
               + torch.randn(n, 1, 2, generator=generator) * steps) * scale
        return raw * constraint_theta(clean_past, raw, future_head)[:, None, None]

    scale = 0.025
    while scale < 2.0 and draw(scale).norm(dim=-1).mean() < target_size:
        scale += 0.025

    sizes, runs = [], []
    with torch.no_grad():
        for _ in range(draws):
            delta = draw(scale)
            sizes.append(float(delta.norm(dim=-1).mean()))
            runs.append(sample_metrics(perturbed_forward(model, batch, delta), future, forward, left))
    return {"mean_point_m": float(np.mean(sizes)), "draws": draws,
            "metrics": {k: float(torch.stack([r[k] for r in runs]).mean()) for k in runs[0]},
            "beyond_half_lane": {d: float(torch.stack([(r[d] > HALF_LANE_WIDTH).float() for r in runs]).mean())
                                 for d in DIRECTIONS}}

# ========================================================
# FIGURE
# ========================================================

def plot_examples(examples, path):
    fig, axes = plt.subplots(1, len(examples), figsize=(4.6 * len(examples), 4.6), facecolor=SURFACE)
    for ax, ex in zip(axes, examples):
        ax.set_facecolor(SURFACE)
        ax.plot(*ex["clean_past"].T, color=INK, linewidth=2, marker="o", markersize=4,
                label="Observed history")
        ax.plot(*ex["adv_past"].T, color=ORANGE, linewidth=2, marker="o", markersize=4,
                markeredgecolor=SURFACE, markeredgewidth=1, label="Adversarial history")
        ax.plot(*ex["future"].T, color=INK, linewidth=2, linestyle=(0, (1, 1.5)), label="Ground truth")
        ax.plot(*ex["clean_pred"].T, color=BLUE, linewidth=2, label="Prediction, clean (min-ADE mode)")
        ax.plot(*ex["adv_pred"].T, color=ORANGE, linewidth=2, linestyle="--",
                label="Prediction, attacked (min-ADE mode)")
        ax.set_aspect("equal", adjustable="datalim")
        ax.set_title(f"{ex['objective']} attack: {ex['clean']:+.2f} → {ex['adv']:+.2f} m",
                     color=INK, fontsize=9, loc="left")
        ax.set_xlabel("x (m), target heading along +x", color=INK_2, fontsize=8)
        ax.set_ylabel("y (m)", color=INK_2, fontsize=8)
        ax.tick_params(colors=MUTED, labelcolor=INK_2, labelsize=7, length=0)
        for side in ax.spines.values():
            side.set_color(GRID)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=5, frameon=False, labelcolor=INK_2,
               fontsize=8, bbox_to_anchor=(0.5, 1.0))
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    fig.savefig(path, dpi=160, facecolor=SURFACE)
    plt.close(fig)

# ========================================================
# MAIN
# ========================================================

def dataset_path(split, use_map=False):
    suffix = "_map" if use_map else ""
    return PROJECT_ROOT / "data" / "processed" / f"nuscenes_{split}_prediction_samples{suffix}.pkl"


def load_samples(split, use_map=False):
    with open(dataset_path(split, use_map), "rb") as f:
        return pickle.load(f)


def build_inputs(samples):
    """One batch of all samples, plus the clean target history and future."""
    batch = next(iter(DataLoader([sample_to_graph(s) for s in samples], batch_size=len(samples))))
    clean_past = torch.tensor(np.stack([s["target_past"] for s in samples]))
    future = torch.tensor(np.stack([s["target_future"] for s in samples]))
    return batch, clean_past, future


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", nargs="+",
                        default=["multimodal:42", "multimodal:123", "multimodal:2026"])
    parser.add_argument("--split", default="val", choices=["val", "test"])
    parser.add_argument("--num-samples", type=int, default=500)
    parser.add_argument("--restarts", type=int, default=10)
    parser.add_argument("--iters", type=int, default=100)
    parser.add_argument("--boundary-points", type=int, default=3,
                        help="ground-truth future points included in the physical check (0 = original code)")
    parser.add_argument("--checkpoint-dir", nargs="+", type=Path, default=CHECKPOINT_DIRS,
                        help="searched recursively for <name>_seed<seed>_best.pth")
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "data" / "processed" / "adversarial")
    parser.add_argument("--figure-dir", type=Path, default=PROJECT_ROOT / "doc" / "figures" / "adversarial")
    parser.add_argument("--tag", default="", help="suffix for output files, e.g. _seed42 for parallel runs")
    parser.add_argument("--merge", nargs="+", type=Path,
                        help="combine result JSONs from parallel runs into one report and exit")
    parser.add_argument("--control-only", type=Path,
                        help="add the random-perturbation control to an existing result JSON and exit")
    args = parser.parse_args()

    if args.control_only:
        results = json.loads(args.control_only.read_text())
        samples = load_samples(results["split"], results.get("map_data", False))
        batch, clean_past, future = build_inputs([samples[i] for i in results["sample_indices"]])
        future_head = future[:, :results["boundary_points"]]
        for spec, entry in results["models"].items():
            model = load_model(spec, args.checkpoint_dir)
            attack_size = np.mean([entry["perturbation"][o]["mean_point_m"] for o in OBJECTIVES])
            entry["random_control"] = random_control(model, batch, clean_past, future, future_head, attack_size)
            print(f"{spec}: random control done", flush=True)
        args.control_only.write_text(json.dumps(results, indent=2) + "\n")
        write_report(results, args.control_only.with_suffix(".md"))
        return

    if args.merge:
        merged = json.loads(args.merge[0].read_text())
        for path in args.merge[1:]:
            other = json.loads(path.read_text())
            for key in ("split", "sample_indices", "restarts", "iters", "boundary_points"):
                if other[key] != merged[key]:
                    raise ValueError(f"{path}: {key} differs from {args.merge[0]}")
            merged["models"].update(other["models"])
        name = f"results_{merged['split']}_b{merged['boundary_points']}{args.tag}"
        (args.output_dir / f"{name}.json").write_text(json.dumps(merged, indent=2) + "\n")
        write_report(merged, args.output_dir / f"{name}.md")
        return

    torch.manual_seed(0)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.figure_dir.mkdir(parents=True, exist_ok=True)

    use_map = uses_map(args.models)
    samples = load_samples(args.split, use_map)
    chosen = np.sort(np.random.default_rng(0).choice(len(samples), args.num_samples, replace=False))
    samples = [samples[i] for i in chosen]
    batch, clean_past, future = build_inputs(samples)
    future_head = future[:, :args.boundary_points]
    forward, left = motion_directions(future)

    attackable = physical_ok(torch.cat([clean_past, future_head], dim=1))
    print(f"{args.split}: {len(samples)} samples; clean trace within physical bounds: {attackable.float().mean():.1%}")

    results = {
        "split": args.split, "map_data": use_map, "dataset": dataset_path(args.split, use_map).name,
        "dataset_sha256": sha256(dataset_path(args.split, use_map)),
        "num_samples": len(samples), "sample_indices": chosen.tolist(),
        "restarts": args.restarts, "iters": args.iters, "boundary_points": args.boundary_points,
        "deviation_bound_m": DEVIATION_BOUND, "physical_bounds": PHYSICAL_BOUNDS,
        "half_lane_width_m": HALF_LANE_WIDTH,
        "clean_trace_feasible_share": float(attackable.float().mean()),
        "models": {},
    }

    for spec in args.models:
        model = load_model(spec, args.checkpoint_dir)
        generator = torch.Generator().manual_seed(1)
        start = time.time()

        with torch.no_grad():
            clean = sample_metrics(perturbed_forward(model, batch, torch.zeros_like(clean_past)),
                                   future, forward, left)

        path = checkpoint_path(spec, args.checkpoint_dir)
        entry = {"checkpoint": str(path.relative_to(PROJECT_ROOT)) if path else None,
                 "checkpoint_sha256": sha256(path) if path else None,
                 "clean": {k: float(v.mean()) for k, v in clean.items()}, "attacked": {},
                 "clean_beyond_half_lane": {d: float((clean[d] > HALF_LANE_WIDTH).float().mean()) for d in DIRECTIONS},
                 "attacked_beyond_half_lane": {}, "perturbation": {}, "all_metrics_under_attack": {}}
        deltas = {}
        for objective in OBJECTIVES:
            delta = pgd_attack(model, batch, clean_past, future, future_head, objective,
                               args.restarts, args.iters, generator)
            deltas[objective] = delta
            with torch.no_grad():
                adv = sample_metrics(perturbed_forward(model, batch, delta), future, forward, left)
            entry["attacked"][objective] = float(adv[objective].mean())
            entry["all_metrics_under_attack"][objective] = {k: float(v.mean()) for k, v in adv.items()}
            if objective in DIRECTIONS:
                beyond = adv[objective] > HALF_LANE_WIDTH
                entry["attacked_beyond_half_lane"][objective] = float(beyond.float().mean())
                entry.setdefault("attackable_only", {})[objective] = {
                    "clean": float((clean[objective] > HALF_LANE_WIDTH)[attackable].float().mean()),
                    "attacked": float(beyond[attackable].float().mean()),
                }
            norms = delta.norm(dim=-1)
            entry["perturbation"][objective] = {"mean_point_m": float(norms.mean()),
                                                "max_point_m": float(norms.max()),
                                                "zero_share": float((norms.max(dim=1).values == 0).float().mean())}
            print(f"{spec} {objective:5s}: clean {entry['clean'][objective]:+.3f} -> attacked "
                  f"{entry['attacked'][objective]:+.3f} m | {time.time() - start:.0f}s", flush=True)

        entry["directional_impact_share"] = float(np.mean(list(entry["attacked_beyond_half_lane"].values())))
        entry["directional_clean_share"] = float(np.mean(list(entry["clean_beyond_half_lane"].values())))
        attack_size = np.mean([entry["perturbation"][o]["mean_point_m"] for o in OBJECTIVES])
        entry["random_control"] = random_control(model, batch, clean_past, future, future_head, attack_size)
        results["models"][spec] = entry

        # Illustrative examples: median attacked sample for each direction (first model only)
        if spec == args.models[0]:
            examples = []
            for objective in DIRECTIONS:
                with torch.no_grad():
                    clean_pred = perturbed_forward(model, batch, torch.zeros_like(clean_past))
                    adv_pred = perturbed_forward(model, batch, deltas[objective])
                adv = sample_metrics(adv_pred, future, forward, left)[objective]
                i = int(torch.argsort(adv)[len(adv) // 2])
                best_c = (clean_pred[i] - future[i]).norm(dim=-1).mean(-1).argmin()
                best_a = (adv_pred[i] - future[i]).norm(dim=-1).mean(-1).argmin()
                examples.append({
                    "objective": objective,
                    "clean_past": clean_past[i].numpy(),
                    "adv_past": (clean_past[i] + deltas[objective][i]).numpy(),
                    "future": future[i].numpy(),
                    "clean_pred": clean_pred[i, best_c].numpy(),
                    "adv_pred": adv_pred[i, best_a].numpy(),
                    "clean": float(clean[objective][i]),
                    "adv": float(adv[i]),
                })
            plot_examples(examples, args.figure_dir / f"attack_examples{args.tag}.png")

    name = f"results_{args.split}_b{args.boundary_points}{args.tag}"
    (args.output_dir / f"{name}.json").write_text(json.dumps(results, indent=2) + "\n")
    write_report(results, args.output_dir / f"{name}.md")


def write_report(results, md_path):
    dataset = f" (`{results['dataset']}`)" if "dataset" in results else ""
    md = ["# White-box PGD attack: results", "",
          f"{results['split']} split{dataset}, {results['num_samples']} random samples; perturbation on all 5 observed "
          f"target positions, ≤ {DEVIATION_BOUND} m per point, nuScenes physical bounds from the original code, "
          f"{results['boundary_points']} ground-truth future points included in the physical check; "
          f"{results['restarts']} restarts × {results['iters']} steps. Each column is attacked with its own "
          f"objective. Deviations use the min-ADE candidate (issue #18). "
          f"{results['clean_trace_feasible_share']:.1%} of clean traces satisfy the physical bounds; for the "
          f"others a perturbation is only accepted if it brings the trace within bounds, so most stay unperturbed. "
          f"\"random, same size\" = smooth random perturbations (offset + linear drift) scaled to the attack's mean "
          f"size per point, with the same feasibility scaling, averaged over 20 draws.", "",
          "| Model | | minADE6 | minFDE6 | Left | Right | Front | Rear |", "|---|---|---:|---:|---:|---:|---:|---:|"]

    def metric_row(label, values):
        return f"| {label} | " + " | ".join(f"{values[o]:+.3f}" if o in DIRECTIONS
                                           else f"{values[o]:.3f}" for o in OBJECTIVES) + " |"

    for spec, e in results["models"].items():
        md.append(metric_row(f"{spec} | clean", e["clean"]))
        if "random_control" in e:
            md.append(metric_row(f" | random, same size ({e['random_control']['mean_point_m']:.2f} m)",
                                 e["random_control"]["metrics"]))
        md.append(metric_row(" | attacked", e["attacked"]))
    md += ["", f"Share of samples whose directional deviation exceeds half a lane width ({HALF_LANE_WIDTH} m):", "",
           "| Model | | Left | Right | Front | Rear | Mean of 4 |", "|---|---|---:|---:|---:|---:|---:|"]
    for spec, e in results["models"].items():
        md.append(f"| {spec} | clean | " + " | ".join(f"{e['clean_beyond_half_lane'][d]:.1%}" for d in DIRECTIONS)
                  + f" | {e['directional_clean_share']:.1%} |")
        if "random_control" in e:
            shares = e["random_control"]["beyond_half_lane"]
            md.append(f"| | random, same size | " + " | ".join(f"{shares[d]:.1%}" for d in DIRECTIONS)
                      + f" | {np.mean([shares[d] for d in DIRECTIONS]):.1%} |")
        md.append(f"| | attacked | " + " | ".join(f"{e['attacked_beyond_half_lane'][d]:.1%}" for d in DIRECTIONS)
                  + f" | {e['directional_impact_share']:.1%} |")
        sub = e["attackable_only"]
        md.append(f"| | attacked, attackable samples only | " + " | ".join(f"{sub[d]['attacked']:.1%}" for d in DIRECTIONS)
                  + f" | {np.mean([sub[d]['attacked'] for d in DIRECTIONS]):.1%} |")
    md += ["", "Mean perturbation per observed point (m):", "",
           "| Model | " + " | ".join(OBJECTIVES) + " |", "|---|" + "---:|" * len(OBJECTIVES)]
    for spec, e in results["models"].items():
        md.append(f"| {spec} | " + " | ".join(f"{e['perturbation'][o]['mean_point_m']:.2f}" for o in OBJECTIVES) + " |")
    md.append("")

    md_path.write_text("\n".join(md))
    print("\n".join(md))


if __name__ == "__main__":
    main()
