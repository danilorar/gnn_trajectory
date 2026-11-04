"""Static poster version of the attack animation: val sample 4640, right attack, GRU seed 42.

Run from the project root with the nuScenes metadata and CAM_FRONT images available:
    python doc/poster/figures/hero_example.py --nuscenes-root /path/to/nuscenes

Reuses the helpers of tools/animate_attack_example.py and the saved PGD perturbation, so the
attacked prediction is exactly the one behind the reported results. Shows the scene at the
moment of prediction (t = 0) with the full 6 s predictions and ground truth.
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

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.adversarial_attack import (
    CHECKPOINT_DIRS, build_inputs, checkpoint_path, dataset_path, load_model, load_samples,
    motion_directions, perturbed_forward, sample_metrics,
)
from tools.animate_attack_example import (
    AGENT_STYLE, BEST_C, CRAFTED_C, INK, LANE_C, MODE_C, MUTED, PAST, PAST_C, TRUTH_C,
    box_in_image, ground_path_pixels, load_metadata, track,
)
from utils.experiment_utils import sha256
from utils.graph_utils import general_category

OUT_DIR = Path(__file__).resolve().parent
SEED, OBJECTIVE, VAL_INDEX = 42, "right", 4640
WHITE = "#ffffff"

plt.rcParams.update({"font.family": "Arial", "font.size": 13.5, "axes.titlesize": 15,
                     "axes.titleweight": "bold", "axes.titlelocation": "left", "axes.titlepad": 5})


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--nuscenes-root", type=Path, required=True)
    args = parser.parse_args()

    spec = f"multimodal_map_gru:{SEED}"
    attack_dir = PROJECT_ROOT / "data/processed/adversarial_modes"
    results = json.loads((attack_dir / f"results_val_b3_multimodal_map_gru_seed{SEED}.json").read_text())
    entry = results["models"][spec]
    deltas = np.load(attack_dir / f"deltas_val_b3_multimodal_map_gru_seed{SEED}.npz")
    assert deltas["sample_indices"].tolist() == results["sample_indices"]
    checkpoint = checkpoint_path(spec, CHECKPOINT_DIRS)
    assert sha256(checkpoint) == entry["checkpoint_sha256"], "checkpoint differs from the attacked one"
    assert sha256(dataset_path("val", use_map=True)) == results["dataset_sha256"], "dataset differs"

    sample = load_samples("val", use_map=True)[VAL_INDEX]
    k = results["sample_indices"].index(VAL_INDEX)
    delta = torch.tensor(deltas[OBJECTIVE][k:k + 1])
    batch, past, future = build_inputs([sample])
    model = load_model(spec, CHECKPOINT_DIRS)
    forward, left = motion_directions(future)
    with torch.no_grad():
        preds = {"clean": perturbed_forward(model, batch, torch.zeros_like(delta)),
                 "attacked": perturbed_forward(model, batch, delta)}
        metrics = {n: {m: float(v[0]) for m, v in sample_metrics(p, future, forward, left).items()}
                   for n, p in preds.items()}
    preds = {n: p[0].numpy() for n, p in preds.items()}
    truth, history = future[0].numpy(), past[0].numpy()
    crafted = history + delta[0].numpy()
    best = {n: int(np.linalg.norm(p - truth, axis=-1).mean(-1).argmin()) for n, p in preds.items()}
    size_cm = np.linalg.norm(delta[0].numpy(), axis=-1) * 100

    # The animation's JSON records the same metrics; refuse to draw anything else
    reference = json.loads((PROJECT_ROOT / "doc/figures/adversarial/"
                            f"attack_animation_gru_seed{SEED}_{OBJECTIVE}_val{VAL_INDEX}.json").read_text())
    for n in preds:
        assert abs(metrics[n]["ade"] - reference["metrics"][n]["ade"]) < 1e-5

    # Scene metadata at the moment of prediction
    meta_dir = args.nuscenes_root / "v1.0-trainval"
    neighbours = list(zip(sample["neighbor_instance_tokens"], sample["neighbors_categories"]))
    window, scene, calibration, cameras, poses, annotations = load_metadata(
        meta_dir, sample, {sample["target_instance_token"], *[n for n, _ in neighbours]})
    from pyquaternion import Quaternion
    now = annotations[(sample["target_instance_token"], window[PAST])]
    origin, yaw = np.asarray(now["translation"][:2]), Quaternion(now["rotation"]).yaw_pitch_roll[0]
    target = track(annotations, sample["target_instance_token"], window, origin, yaw)
    assert np.allclose(target[:PAST + 1], history, atol=1e-3) and np.allclose(target[PAST + 1:], truth, atol=1e-3)
    agents = [(general_category(c), track(annotations, i, window, origin, yaw)[PAST]) for i, c in neighbours]
    ground_z = now["translation"][2] - now["size"][2] / 2

    points = np.vstack([history, crafted, truth, preds["clean"][best["clean"]], preds["attacked"][best["attacked"]]])
    lo, hi = points.min(0) - np.array([6, 9]), points.max(0) + np.array([6, 9])
    centre, half = (lo + hi) / 2, (hi - lo) / 2
    half = np.maximum(half, [half[1] * 1.45, half[0] / 1.45])
    xlim, ylim = (centre[0] - half[0], centre[0] + half[0]), (centre[1] - half[1], centre[1] + half[1])

    fig = plt.figure(figsize=(8.3, 6.55))
    grid = fig.add_gridspec(2, 1, height_ratios=[1, 1.02], hspace=0.24, left=0.01, right=0.99, top=0.955, bottom=0.165)
    top = grid[0].subgridspec(1, 2, wspace=0.04)
    bottom = grid[1].subgridspec(1, 2, width_ratios=[0.8, 1.2], wspace=0.10)
    ax = {"clean": fig.add_subplot(top[0]), "attacked": fig.add_subplot(top[1])}
    ax_hist, ax_cam = fig.add_subplot(bottom[0]), fig.add_subplot(bottom[1])

    for name, a in ax.items():
        for lane in sample["map_lanes"]:
            a.plot(*lane.T, color=LANE_C, lw=0.8, zorder=1)
        for kind, p in agents:
            if np.isnan(p).any() or not (xlim[0] < p[0] < xlim[1] and ylim[0] < p[1] < ylim[1]):
                continue
            marker, color, _ = AGENT_STYLE[kind]
            a.scatter(*p, marker=marker, s=26 if kind != "pedestrian" else 10, color=color, zorder=3, lw=0)
        observed = crafted if name == "attacked" else history
        for m, mode in enumerate(preds[name]):
            if m != best[name]:
                a.plot(*np.vstack([observed[-1:], mode]).T, "--", color=MODE_C, lw=0.9, zorder=4)
        a.plot(*np.vstack([observed[-1:], truth]).T, color=TRUTH_C, lw=2.4, zorder=6, solid_capstyle="round")
        a.plot(*np.vstack([observed[-1:], preds[name][best[name]]]).T, "--", color=BEST_C, lw=2.2, zorder=7)
        if name == "attacked":
            a.plot(*history.T, color=PAST_C, lw=1.2, alpha=0.4, zorder=5)
        a.plot(*observed.T, "-o", color=CRAFTED_C if name == "attacked" else PAST_C, lw=2, ms=3, zorder=8)
        a.scatter(*observed[-1], s=55, color=CRAFTED_C if name == "attacked" else PAST_C,
                  edgecolors=WHITE, lw=1, zorder=9)
        a.text(0.03, 0.05, f"minADE$_6$ {metrics[name]['ade']:.2f} m\nbest mode "
               f"{metrics[name][OBJECTIVE]:.2f} m right", transform=a.transAxes, va="bottom", fontsize=13,
               color=INK, bbox=dict(boxstyle="round,pad=0.3", fc=WHITE, ec=LANE_C, lw=0.8), zorder=10)
        a.set_xlim(xlim); a.set_ylim(ylim); a.set_aspect("equal")
        a.set_xticks([]); a.set_yticks([])
        for side in a.spines.values():
            side.set_color(LANE_C)
    ax["clean"].set_title("(a) Real past 2 s → prediction")
    ax["attacked"].set_title(f"(b) Crafted past, {size_cm.mean():.0f} cm per point")
    scale_x = xlim[0] + 0.72 * (xlim[1] - xlim[0])
    scale_y = ylim[0] + 0.08 * (ylim[1] - ylim[0])
    ax["clean"].plot([scale_x, scale_x + 10], [scale_y, scale_y], color=MUTED, lw=1.5)
    ax["clean"].text(scale_x + 5, scale_y + 0.6, "10 m", ha="center", va="bottom", fontsize=12, color=MUTED)

    # (c) the observed history only, with the lateral axis stretched so centimetre offsets show
    for i in range(PAST + 1):
        ax_hist.plot(*np.stack([history[i], crafted[i]]).T, color=MUTED, lw=0.8, zorder=2)
        below = crafted[i, 1] < history[i, 1]
        side = 8 if 0 < i < PAST and below else 0  # keep inner labels clear of the incoming segment
        ax_hist.annotate(f"{size_cm[i]:.0f} cm", crafted[i], xytext=(side, -15 if below else 7),
                         textcoords="offset points", ha="left" if side else "center", fontsize=12.5, color=INK)
    ax_hist.plot(*history.T, "-o", color=PAST_C, lw=2, ms=5, zorder=3, label="real")
    ax_hist.plot(*crafted.T, "-o", color=CRAFTED_C, lw=2, ms=5, zorder=4, label="crafted")
    for i in (0, PAST):
        ax_hist.annotate("now" if i == PAST else f"{(i - PAST) * 0.5:+.0f} s", history[i], xytext=(0, 8),
                         textcoords="offset points", ha="center", fontsize=12, color=MUTED)
    ys = np.r_[history[:, 1], crafted[:, 1]]
    ax_hist.set_ylim(ys.min() - 0.12, ys.max() + 0.12)
    ax_hist.set_xlim(history[:, 0].min() - 1.5, history[:, 0].max() + 1.5)
    ax_hist.set_yticks([]); ax_hist.tick_params(labelsize=12, colors=MUTED, length=2)
    ax_hist.set_xlabel("metres along heading\n(lateral offsets exaggerated)", fontsize=12, color=MUTED, labelpad=2)
    for side in ax_hist.spines.values():
        side.set_color(LANE_C)
    ax_hist.set_title("(c) Past: real vs crafted")

    # (d) front camera at t = 0 with ground truth and the attacked best mode projected on the road
    token = window[PAST]
    record = cameras[token]
    pose, sensor = poses[record["ego_pose_token"]], calibration[record["calibrated_sensor_token"]]
    with Image.open(args.nuscenes_root / record["filename"]) as image:
        ax_cam.imshow(image.copy())
    img_size = np.array([record["width"], record["height"]])
    for path, start, color, style in ((truth, history, TRUTH_C, "-"), (preds["attacked"][best["attacked"]], crafted, BEST_C, "--")):
        for segment in ground_path_pixels(np.vstack([start[-1:], path]), origin, yaw, ground_z, pose, sensor):
            ax_cam.plot(*segment.T, style, color=color, lw=2.6)
    box = box_in_image(now, pose, sensor, img_size)
    if box is not None:
        (x0, y0), (x1, y1) = np.clip(box[0], 0, img_size), np.clip(box[1], 0, img_size)
        ax_cam.add_patch(plt.Rectangle((x0, y0), x1 - x0, y1 - y0, fill=False, ec=PAST_C, lw=2))
        ax_cam.text(x0, y0 - 12, "target", color=WHITE, fontsize=12, va="bottom",
                    bbox=dict(fc=PAST_C, ec="none", pad=1.5))
    ax_cam.set_xlim(0, img_size[0]); ax_cam.set_ylim(img_size[1], 0)
    ax_cam.axis("off")
    ax_cam.set_title(f"(d) Front camera, {scene['name']}")

    handles = [Line2D([], [], color=PAST_C, lw=2, marker="o", ms=3, label="real past"),
               Line2D([], [], color=CRAFTED_C, lw=2, marker="o", ms=3, label="crafted past"),
               Line2D([], [], color=TRUTH_C, lw=2.4, label="true future"),
               Line2D([], [], color=BEST_C, lw=2.2, ls="--", label="best of 6 modes"),
               Line2D([], [], color=MODE_C, lw=0.9, ls="--", label="other modes")]
    handles += [Line2D([], [], color=c, marker=mk, ls="none", ms=6, label=kind.replace("_", " "))
                for kind, (mk, c, _) in AGENT_STYLE.items() if any(a[0] == kind for a in agents)]
    # Align the bottom row with the bird's-eye panels: (c), made smaller, under (a)'s left edge;
    # (d) under (b)'s right edge
    fig.canvas.draw()
    pa, pb = ax["clean"].get_position(), ax["attacked"].get_position()
    pc, pd = ax_hist.get_position(), ax_cam.get_position()
    ax_hist.set_position([pa.x0, pc.y1 - 0.85 * pc.height, 0.82 * pc.width, 0.85 * pc.height])
    ax_cam.set_anchor("NE")
    ax_cam.set_position([pd.x0, pd.y0, pb.x1 - pd.x0, pd.height])

    # Legend just below the bottom row; the tight bounding box then drops the empty margin
    fig.canvas.draw()
    to_fig = fig.transFigure.inverted()
    bottom = min(a.get_tightbbox(fig.canvas.get_renderer()).transformed(to_fig).y0 for a in (ax_hist, ax_cam))
    fig.legend(handles=handles, loc="upper center", ncol=5, frameon=False, fontsize=12.5,
               bbox_to_anchor=(0.5, bottom - 0.01), columnspacing=1.2, handlelength=2.2)

    for ext in ("pdf", "png"):
        fig.savefig(OUT_DIR / f"hero_example.{ext}", dpi=300, facecolor=WHITE, bbox_inches="tight", pad_inches=0.03)
    print({n: {k: round(v, 3) for k, v in m.items()} for n, m in metrics.items()},
          "perturbation cm", np.round(size_cm, 1), "box", box is not None)


if __name__ == "__main__":
    main()
