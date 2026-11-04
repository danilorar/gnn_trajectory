"""Animate one white-box PGD attack on the map-aware GRU: clean vs attacked, plus the front camera.

Example:
    python tools/animate_attack_example.py --seed 42 --objective right --val-index 4640 \
        --nuscenes-root /path/to/nuscenes

Uses the perturbation that scripts/adversarial_attack.py saved for this sample, so the
attacked prediction is exactly the one behind the reported results. Agent motion and
camera geometry come from the nuScenes v1.0-trainval metadata, streamed so the large
tables are never loaded whole. The camera panel needs the scene's CAM_FRONT keyframes
under <nuscenes-root>/samples/CAM_FRONT/; missing images are listed and left blank.
"""

import argparse
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import torch
from PIL import Image
from pyquaternion import Quaternion

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.adversarial_attack import (
    CHECKPOINT_DIRS, HALF_LANE_WIDTH, build_inputs, checkpoint_path, dataset_path,
    load_model, load_samples, motion_directions, perturbed_forward, sample_metrics,
)
from utils.experiment_utils import sha256, write_json
from utils.graph_utils import general_category

PAST, FUTURE, DT = 4, 12, 0.5  # keyframes before / after the current one, seconds per keyframe
FRAME_DT, HOLD_NOW, HOLD_END, FPS = 0.1, 10, 15, 10

PAST_C, CRAFTED_C, TRUTH_C, BEST_C, MODE_C = "#2563b0", "#eb6834", "#2e8b57", "#d62728", "#9a9a9a"
LANE_C, INK, MUTED, SURFACE = "#cfcfcf", "#1a1a1a", "#6b6b6b", "#fcfcfb"
AGENT_STYLE = {"car": ("s", "#4a6984", "C"), "other_vehicle": ("s", "#4a6984", "V"),
               "large_vehicle": ("s", "#b8731a", "T"), "two_wheeler": ("^", "#2a9d8f", "B"),
               "pedestrian": ("o", "#8e4fb0", "P")}

# ========================================================
# METADATA
# ========================================================

def stream_records(path, keep):
    """Yield the records of a nuScenes table whose raw text passes keep(text).

    The tables hold one record per '{' ... '}' block at the top level with only
    lists nested inside, so a record can be filtered before it is parsed.
    """
    lines = []
    with open(path) as handle:
        for line in handle:
            if line == "{\n":
                lines = [line]
            elif lines:
                lines.append(line)
                if line in ("}\n", "},\n"):
                    text, lines = "".join(lines), []
                    if keep(text):
                        yield json.loads(text.rstrip().rstrip(","))


def token_field(text, key):
    start = text.index(f'"{key}": "') + len(key) + 5
    return text[start:start + 32]


def load_metadata(meta_dir, sample, instances):
    """CAM_FRONT records, ego poses and annotations for the 17 keyframes around the sample."""
    samples = {r["token"]: r for r in json.loads((meta_dir / "sample.json").read_text())}
    window = [sample["sample_token"]]
    for _ in range(PAST):
        window.insert(0, samples[window[0]]["prev"])
    for _ in range(FUTURE):
        window.append(samples[window[-1]]["next"])
    if "" in window:
        raise ValueError("The scene does not cover 2 s before and 6 s after this sample")
    keyframes = set(window)
    scene = next(r for r in json.loads((meta_dir / "scene.json").read_text())
                 if r["token"] == samples[window[PAST]]["scene_token"])
    calibration = {r["token"]: r for r in json.loads((meta_dir / "calibrated_sensor.json").read_text())}
    cameras = {r["sample_token"]: r for r in stream_records(
        meta_dir / "sample_data.json",
        lambda t: "samples/CAM_FRONT/" in t and token_field(t, "sample_token") in keyframes)}
    pose_tokens = {r["ego_pose_token"] for r in cameras.values()}
    poses = {r["token"]: r for r in stream_records(
        meta_dir / "ego_pose.json", lambda t: token_field(t, "token") in pose_tokens)}
    annotations = {(r["instance_token"], r["sample_token"]): r for r in stream_records(
        meta_dir / "sample_annotation.json",
        lambda t: token_field(t, "sample_token") in keyframes and token_field(t, "instance_token") in instances)}
    return window, scene, calibration, cameras, poses, annotations

# ========================================================
# GEOMETRY
# ========================================================

def rotation(yaw):
    return np.array([[np.cos(yaw), -np.sin(yaw)], [np.sin(yaw), np.cos(yaw)]])


def track(annotations, instance, window, origin, yaw):
    """Keyframe positions [17, 2] of one agent in the target-local frame; NaN where unannotated."""
    out = np.full((len(window), 2), np.nan)
    for i, token in enumerate(window):
        ann = annotations.get((instance, token))
        if ann is not None:
            out[i] = (np.asarray(ann["translation"][:2]) - origin) @ rotation(yaw)
    return out


def at_time(keyframes, t):
    """Linear interpolation of a keyframe track at time t (keyframe 0 is t = -2 s)."""
    x = min(max((t + PAST * DT) / DT, 0), len(keyframes) - 1)
    i = min(int(np.floor(x)), len(keyframes) - 2)
    f = x - i
    return keyframes[i] * (1 - f) + keyframes[i + 1] * f


def camera_coords(points, pose, sensor):
    """Global xyz points [N, 3] in camera coordinates (z forward)."""
    points = (points - pose["translation"]) @ Quaternion(pose["rotation"]).rotation_matrix
    return (points - sensor["translation"]) @ Quaternion(sensor["rotation"]).rotation_matrix


def pixels(points, sensor):
    projected = points @ np.asarray(sensor["camera_intrinsic"]).T
    return projected[:, :2] / projected[:, 2:]


def box_in_image(ann, pose, sensor, size):
    """2D bounding box of a 3D annotation box, or None if it is behind or outside the image."""
    w, l, h = ann["size"]
    corners = np.array([[x, y, z] for x in (-l / 2, l / 2) for y in (-w / 2, w / 2) for z in (-h / 2, h / 2)])
    corners = corners @ Quaternion(ann["rotation"]).rotation_matrix.T + ann["translation"]
    cam = camera_coords(corners, pose, sensor)
    if (cam[:, 2] < 0.5).any():
        return None
    pix = pixels(cam, sensor)
    lo, hi = pix.min(0), pix.max(0)
    inside = (np.minimum(hi, size) - np.maximum(lo, 0)).clip(min=0).prod() / max((hi - lo).prod(), 1e-6)
    return (lo, hi) if inside > 0.5 else None


def ground_path_pixels(local, origin, yaw, ground_z, pose, sensor):
    """Project a target-local ground path; returns contiguous pixel segments in front of the camera."""
    world = np.column_stack([local @ rotation(yaw).T + origin, np.full(len(local), ground_z)])
    cam = camera_coords(world, pose, sensor)
    pix = pixels(cam, sensor)
    segments, current = [], []
    for point, depth in zip(pix, cam[:, 2]):
        if depth > 1.0:
            current.append(point)
        elif current:
            segments.append(np.array(current)); current = []
    if current:
        segments.append(np.array(current))
    return [s for s in segments if len(s) > 1]

# ========================================================
# MAIN
# ========================================================

def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--objective", default="right", choices=("ade", "fde", "left", "right", "front", "rear"))
    parser.add_argument("--val-index", type=int, default=4640, help="index into the val split (must be one of the attacked samples)")
    parser.add_argument("--nuscenes-root", type=Path, default=PROJECT_ROOT / "data",
                        help="folder with v1.0-trainval/ and, for the camera panel, samples/CAM_FRONT/")
    parser.add_argument("--attack-dir", type=Path, default=PROJECT_ROOT / "data/processed/adversarial_modes")
    parser.add_argument("--output", type=Path, help="new GIF path; default under doc/figures/adversarial/")
    parser.add_argument("--dpi", type=int, default=70)
    args = parser.parse_args()

    spec = f"multimodal_map_gru:{args.seed}"
    stem = f"results_val_b3_multimodal_map_gru_seed{args.seed}"
    results = json.loads((args.attack_dir / f"{stem}.json").read_text())
    entry = results["models"][spec]
    deltas = np.load(args.attack_dir / f"deltas_val_b3_multimodal_map_gru_seed{args.seed}.npz")
    if deltas["sample_indices"].tolist() != results["sample_indices"]:
        raise ValueError("Saved perturbations and results cover different samples")
    if args.val_index not in results["sample_indices"]:
        parser.error("--val-index is not one of the attacked samples")
    checkpoint = checkpoint_path(spec, CHECKPOINT_DIRS)
    if sha256(checkpoint) != entry["checkpoint_sha256"]:
        raise ValueError("Checkpoint differs from the one that was attacked")
    if sha256(dataset_path("val", use_map=True)) != results["dataset_sha256"]:
        raise ValueError("Dataset differs from the one that was attacked")
    if args.output is None:
        args.output = PROJECT_ROOT / f"doc/figures/adversarial/attack_animation_gru_seed{args.seed}_{args.objective}_val{args.val_index}.gif"
    for path in (args.output, args.output.with_suffix(".json")):
        if path.exists():
            raise FileExistsError(f"Refusing to overwrite existing output: {path}")

    # Predictions through the attack's own differentiable input path
    sample = load_samples("val", use_map=True)[args.val_index]
    k = results["sample_indices"].index(args.val_index)
    delta = torch.tensor(deltas[args.objective][k:k + 1])
    batch, past, future = build_inputs([sample])
    model = load_model(spec, CHECKPOINT_DIRS)
    forward, left = motion_directions(future)
    with torch.no_grad():
        preds = {"clean": perturbed_forward(model, batch, torch.zeros_like(delta)),
                 "attacked": perturbed_forward(model, batch, delta)}
        metrics = {name: {m: float(v[0]) for m, v in sample_metrics(p, future, forward, left).items()}
                   for name, p in preds.items()}
    preds = {name: p[0].numpy() for name, p in preds.items()}
    truth, history = future[0].numpy(), past[0].numpy()
    crafted = history + delta[0].numpy()
    best = {name: int(np.linalg.norm(p - truth, axis=-1).mean(-1).argmin()) for name, p in preds.items()}

    # Scene metadata; reconstruct the processed trajectories to confirm the frame
    meta_dir = args.nuscenes_root / "v1.0-trainval"
    neighbours = list(zip(sample["neighbor_instance_tokens"], sample["neighbors_categories"]))
    window, scene, calibration, cameras, poses, annotations = load_metadata(
        meta_dir, sample, {sample["target_instance_token"], *[n for n, _ in neighbours]})
    now = annotations[(sample["target_instance_token"], window[PAST])]
    origin, yaw = np.asarray(now["translation"][:2]), Quaternion(now["rotation"]).yaw_pitch_roll[0]
    target = track(annotations, sample["target_instance_token"], window, origin, yaw)
    if not (np.allclose(target[:PAST + 1], history, atol=1e-3) and np.allclose(target[PAST + 1:], truth, atol=1e-3)):
        raise ValueError("Metadata does not reproduce the processed target trajectory")
    agents = []
    for (instance, category), processed in zip(neighbours, sample["neighbors_pasts"]):
        keyframes = track(annotations, instance, window, origin, yaw)
        if not np.allclose(keyframes[:PAST + 1], processed, atol=1e-3):
            raise ValueError("Metadata does not reproduce a processed neighbour trajectory")
        agents.append((general_category(category), keyframes))
    order = np.argsort([np.linalg.norm(a[1][PAST]) for a in agents])
    counters, labelled = {}, []
    for i in order:
        kind, keyframes = agents[i]
        marker, color, letter = AGENT_STYLE[kind]
        counters[letter] = counters.get(letter, 0) + 1
        labelled.append((kind, keyframes, marker, color, f"{letter}{counters[letter]}" if kind != "pedestrian" else None))

    images = {token: args.nuscenes_root / cameras[token]["filename"] for token in window}
    missing = [p for p in images.values() if not p.is_file() or p.stat().st_size == 0]
    if missing:
        print(f"{len(missing)}/{len(images)} CAM_FRONT images missing under {args.nuscenes_root}; "
              "the camera panel stays blank for them:")
        for p in missing:
            print(f"  {p.relative_to(args.nuscenes_root)}")
    ground_z = now["translation"][2] - now["size"][2] / 2

    # Shared view for both bird's-eye panels, framed on the target and the two best modes
    points = np.vstack([history, crafted, truth, preds["clean"][best["clean"]], preds["attacked"][best["attacked"]]])
    lo, hi = points.min(0) - 12, points.max(0) + 12
    centre, half = (lo + hi) / 2, (hi - lo) / 2
    half = np.maximum(half, [half[1] * 1.25, half[0] / 1.25])  # keep a 1.25 : 1 panel
    xlim, ylim = (centre[0] - half[0], centre[0] + half[0]), (centre[1] - half[1], centre[1] + half[1])

    fig = plt.figure(figsize=(15, 11.5), facecolor=SURFACE, dpi=args.dpi)
    grid = fig.add_gridspec(2, 1, height_ratios=[1, 0.9], hspace=0.24, left=0.05, right=0.97, top=0.895, bottom=0.135)
    top = grid[0].subgridspec(1, 2, wspace=0.12)
    bottom = grid[1].subgridspec(1, 2, width_ratios=[0.75, 1.25], wspace=0.08)
    ax_clean, ax_attacked = fig.add_subplot(top[0]), fig.add_subplot(top[1])
    ax_history, ax_cam = fig.add_subplot(bottom[0]), fig.add_subplot(bottom[1])
    deviation = {name: metrics[name][args.objective] for name in preds}
    share = entry["attacked_beyond_half_lane"].get(args.objective)
    perturbation = np.linalg.norm(delta[0].numpy(), axis=-1)
    endpoint_gap = float(np.linalg.norm(preds["attacked"][best["attacked"]][-1] - preds["clean"][best["clean"]][-1]))

    def draw_bev(ax, name, t):
        ax.clear()
        ax.set_facecolor(SURFACE)
        for lane in sample["map_lanes"]:
            ax.plot(*lane.T, color=LANE_C, lw=0.9, zorder=1)
        for kind, keyframes, marker, color, label in labelled:
            p = at_time(keyframes, t)
            if np.isnan(p).any() or not (xlim[0] < p[0] < xlim[1] and ylim[0] < p[1] < ylim[1]):
                continue
            ax.scatter(*p, marker=marker, s=28 if kind != "pedestrian" else 14, color=color, zorder=3)
            if label:
                ax.annotate(label, p, xytext=(4, 4), textcoords="offset points", fontsize=7, color=color)
        observed = crafted if name == "attacked" else history
        full = np.vstack([observed, truth])
        if name == "attacked":
            ax.plot(*history[:min(int((t + 2) / DT) + 1, PAST + 1)].T, color=PAST_C, lw=1.2, alpha=0.35, zorder=4)
        shown = [p for i, p in enumerate(observed) if (i - PAST) * DT <= t]
        if t < 0:
            shown.append(at_time(full, t))
        ax.plot(*np.array(shown).T, "-o", color=CRAFTED_C if name == "attacked" else PAST_C, lw=2.2, ms=3, zorder=5)
        if t >= 0:
            for m, mode in enumerate(preds[name]):
                if m != best[name]:
                    ax.plot(*mode.T, "--", color=MODE_C, lw=1, zorder=4)
            ax.plot(*preds[name][best[name]].T, "--", color=BEST_C, lw=2.4, zorder=6)
            revealed = [observed[-1]] + [p for i, p in enumerate(truth) if (i + 1) * DT <= t] + [at_time(full, t)]
            ax.plot(*np.array(revealed).T, color=TRUTH_C, lw=2.6, zorder=7)
            expected = at_time(np.vstack([observed, preds[name][best[name]]]), t)
            ax.scatter(*expected, s=60, facecolors="none", edgecolors=BEST_C, lw=1.6, zorder=8)
            ax.text(0.02, 0.97, f"minADE6 = {metrics[name]['ade']:.2f} m\n{args.objective} deviation of best mode "
                    f"= {deviation[name]:+.2f} m", transform=ax.transAxes, va="top", fontsize=9, color=INK,
                    bbox=dict(boxstyle="round,pad=0.3", fc=SURFACE, ec=LANE_C))
        ax.scatter(*at_time(full, t), s=70, color=CRAFTED_C if name == "attacked" else PAST_C,
                   edgecolors="white", lw=1, zorder=9)
        ax.set_xlim(xlim); ax.set_ylim(ylim); ax.set_aspect("equal")
        ax.tick_params(labelsize=8, colors=MUTED)
        for side in ax.spines.values():
            side.set_color(LANE_C)
        ax.set_xlabel("x (m), target heading along +x at t = 0", fontsize=8, color=MUTED)
        ax.set_title("(a) Clean history" if name == "clean"
                     else f"(b) Attacked history: {perturbation.mean():.2f} m per point (max {perturbation.max():.2f} m)",
                     fontsize=11, color=INK)

    def draw_history(t):
        """The observed history only, with the lateral axis stretched so centimetre offsets show."""
        ax_history.clear()
        ax_history.set_facecolor(SURFACE)
        seen = min(int(np.floor((t + PAST * DT) / DT + 1e-6)) + 1, PAST + 1)
        for i in range(seen):
            ax_history.plot(*np.stack([history[i], crafted[i]]).T, color=MUTED, lw=0.8, zorder=2)
            below = crafted[i, 1] < history[i, 1]
            ax_history.annotate(f"{perturbation[i] * 100:.0f} cm", crafted[i], xytext=(0, -13 if below else 9),
                                textcoords="offset points", ha="center", fontsize=8, color=CRAFTED_C)
            ax_history.annotate(f"{(i - PAST) * DT:+.1f} s", history[i], xytext=(0, 9 if below else -13),
                                textcoords="offset points", ha="center", fontsize=7, color=MUTED)
        ax_history.plot(*history[:seen].T, "-o", color=PAST_C, lw=2, ms=5, zorder=3)
        ax_history.plot(*crafted[:seen].T, "-o", color=CRAFTED_C, lw=2, ms=5, zorder=4)
        xs, ys = np.r_[history[:, 0], crafted[:, 0]], np.r_[history[:, 1], crafted[:, 1]]
        centre_y, half_y = (ys.min() + ys.max()) / 2, max((ys.max() - ys.min()) * 0.9, 0.25)
        ax_history.set_xlim(xs.min() - 1.5, xs.max() + 1.5)
        ax_history.set_ylim(centre_y - half_y, centre_y + half_y)
        box = ax_history.get_position()
        stretch = ((xs.max() - xs.min() + 3) / (box.width * fig.get_figwidth())) / (2 * half_y / (box.height * fig.get_figheight()))
        if t >= 0:
            ax_history.text(0.02, 0.04, f"Input: history moved {perturbation.mean():.2f} m per point (max {perturbation.max():.2f} m)\n"
                            f"Output: best-mode endpoint moved {endpoint_gap:.1f} m", transform=ax_history.transAxes,
                            fontsize=9, color=INK, bbox=dict(boxstyle="round,pad=0.3", fc=SURFACE, ec=LANE_C))
        ax_history.tick_params(labelsize=8, colors=MUTED)
        for side in ax_history.spines.values():
            side.set_color(LANE_C)
        ax_history.set_xlabel("x (m)", fontsize=8, color=MUTED)
        ax_history.set_ylabel(f"y (m), stretched ×{stretch:.0f}", fontsize=8, color=MUTED)
        ax_history.set_title("(c) Observed past 2 s, zoomed: original vs crafted", fontsize=11, color=INK)

    def draw_camera(t):
        ax_cam.clear()
        index = int(np.clip(np.floor((t + PAST * DT) / DT + 1e-6), 0, len(window) - 1))
        token = window[index]
        record = cameras[token]
        pose, sensor = poses[record["ego_pose_token"]], calibration[record["calibrated_sensor_token"]]
        size = np.array([record["width"], record["height"]])
        if images[token] in missing:
            ax_cam.add_patch(plt.Rectangle((0, 0), *size, color="#e9e9e6"))
            ax_cam.text(*size / 2, f"CAM_FRONT image not available locally\n{record['filename']}",
                        ha="center", va="center", fontsize=9, color=MUTED)
        else:
            with Image.open(images[token]) as image:
                ax_cam.imshow(image.copy())
            # The prediction exists from t = 0 on, as in the bird's-eye panels
            paths = ((preds["attacked"][best["attacked"]], BEST_C, "--"), (truth, TRUTH_C, "-")) if t >= 0 else ()
            for path, color, style in paths:
                for segment in ground_path_pixels(np.vstack([crafted[-1:], path]) if color == BEST_C else
                                                  np.vstack([history[-1:], path]), origin, yaw, ground_z, pose, sensor):
                    ax_cam.plot(*segment.T, style, color=color, lw=2.4)
            ann = annotations.get((sample["target_instance_token"], token))
            box = box_in_image(ann, pose, sensor, size) if ann else None
            if box is not None:
                (x0, y0), (x1, y1) = np.clip(box[0], 0, size), np.clip(box[1], 0, size)
                ax_cam.add_patch(plt.Rectangle((x0, y0), x1 - x0, y1 - y0, fill=False, ec=PAST_C, lw=2))
                ax_cam.text(x0 + 4, max(y0 - 6, 30), "Target", color="white", fontsize=9, va="bottom",
                            bbox=dict(fc=PAST_C, ec="none", pad=1.5))
            else:
                ax_cam.text(size[0] - 20, 40, "target outside CAM_FRONT", ha="right", fontsize=9, color="white",
                            bbox=dict(fc="#00000088", ec="none", pad=2))
        ax_cam.set_xlim(0, size[0]); ax_cam.set_ylim(size[1], 0); ax_cam.set_aspect("equal")
        ax_cam.axis("off")
        ax_cam.set_title(f"(d) Front camera, keyframe t = {(index - PAST) * DT:+.1f} s: recorded scene; ground truth\n"
                         "and attacked best mode projected on the road", fontsize=11, color=INK)

    times = list(np.round(np.arange(-PAST * DT, FUTURE * DT + 1e-9, FRAME_DT), 2))
    zero = times.index(0.0)
    frames = times[:zero + 1] + [0.0] * HOLD_NOW + times[zero + 1:] + [times[-1]] * HOLD_END
    header = fig.text(0.05, 0.955, "", fontsize=14, color=INK, weight="bold")
    fig.text(0.05, 0.925, f"White-box PGD attack ({args.objective}) on the map-aware GRU (#19), seed {args.seed}, "
             f"val sample {args.val_index}. Same real future in both panels; only the observed history differs.",
             fontsize=10, color=MUTED)
    if share is not None:
        fig.text(0.05, 0.015, f"Example chosen for visual clarity, not typical: across the 500 attacked val samples, "
                 f"the {args.objective} attack moves the best mode more than {HALF_LANE_WIDTH} m for "
                 f"{share:.1%} of samples (seed {args.seed}).", fontsize=9, color=MUTED)
    legend = [Line2D([], [], color=PAST_C, lw=2.2, marker="o", ms=3, label="Observed past"),
              Line2D([], [], color=CRAFTED_C, lw=2.2, marker="o", ms=3, label="Crafted (attacked) past"),
              Line2D([], [], color=TRUTH_C, lw=2.6, label="Ground truth"),
              Line2D([], [], color=BEST_C, lw=2.4, ls="--", label="Best of 6 (oracle)"),
              Line2D([], [], color=MODE_C, lw=1, ls="--", label="Other modes"),
              Line2D([], [], color=BEST_C, marker="o", mfc="none", ls="none", label="Where the best mode expects the car")]
    legend += [Line2D([], [], color=color, marker=marker, ls="none", label=f"{letter}: {kind.replace('_', ' ')}")
               for kind, (marker, color, letter) in AGENT_STYLE.items() if any(a[0] == kind for a in labelled)]
    fig.legend(handles=legend, loc="lower center", ncol=6, frameon=False, fontsize=9, bbox_to_anchor=(0.5, 0.03))

    def update(t):
        header.set_text(f"{scene['name']}    t = {t:+.1f} s")
        draw_bev(ax_clean, "clean", t)
        draw_bev(ax_attacked, "attacked", t)
        draw_history(t)
        draw_camera(t)

    rendered = []
    for t in frames:
        update(t)
        fig.canvas.draw()
        rendered.append(np.asarray(fig.canvas.buffer_rgba())[..., :3].copy())
    plt.close(fig)
    # One palette for all frames: the exact chart colours first, the rest fitted to the frames
    # (photos included); no dithering, so flat chart colours stay clean
    chart = [PAST_C, CRAFTED_C, TRUTH_C, BEST_C, MODE_C, LANE_C, INK, MUTED, SURFACE, "#ffffff"]
    chart += [color for _, color, _ in AGENT_STYLE.values()]
    fitted = Image.fromarray(np.concatenate(rendered[::8], axis=0)).quantize(
        colors=255 - len(chart), method=Image.Quantize.MEDIANCUT)
    palette = Image.new("P", (1, 1))
    palette.putpalette([round(v * 255) for c in chart for v in matplotlib.colors.to_rgb(c)]
                       + fitted.getpalette()[:3 * (255 - len(chart))])
    gif = [Image.fromarray(frame).quantize(palette=palette, dither=Image.Dither.NONE) for frame in rendered]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    gif[0].save(args.output, save_all=True, append_images=gif[1:], duration=1000 // FPS, loop=0)

    write_json(args.output.with_suffix(".json"), {
        "model": spec, "checkpoint": str(checkpoint.relative_to(PROJECT_ROOT)), "checkpoint_sha256": entry["checkpoint_sha256"],
        "dataset_sha256": results["dataset_sha256"], "attack_results": str((args.attack_dir / f"{stem}.json").relative_to(PROJECT_ROOT)),
        "objective": args.objective, "val_index": args.val_index, "sample_token": sample["sample_token"],
        "scene": scene["name"], "scene_description": scene["description"],
        "perturbation_mean_point_m": float(perturbation.mean()), "perturbation_max_point_m": float(perturbation.max()),
        "metrics": metrics, "best_mode": best, "camera_images_missing": [str(p.relative_to(args.nuscenes_root)) for p in missing],
        "selection": "Chosen for visual clarity among the attacked val samples where the lateral attack moves the "
                     "min-ADE mode more than half a lane width, the clean prediction is accurate and the target is "
                     "visible in CAM_FRONT at t = 0; not representative of all samples.",
    })
    print(f"{scene['name']} | clean minADE6 {metrics['clean']['ade']:.3f} m, {args.objective} {deviation['clean']:+.2f} m -> "
          f"attacked minADE6 {metrics['attacked']['ade']:.3f} m, {args.objective} {deviation['attacked']:+.2f} m")
    print(f"Saved {args.output} ({args.output.stat().st_size / 1e6:.1f} MB) and provenance JSON")


if __name__ == "__main__":
    main()
