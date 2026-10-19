"""Recreate a labelled prediction-sequence GIF and three-panel PNG.

Uses the saved sequence metadata, processed test dataset, and trained GRU.
No temporary caches, raw nuScenes metadata, or training are needed.

Example:
    python tools/plot_prediction_sequence.py --sequence docs/figures/qualitative/scene0556_right_turn/sequence.json --output-dir /tmp/scene0556
"""

import argparse
import json
import pickle
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
from PIL import Image
import torch
from torch_geometric.data import Batch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from utils.graph_utils import sample_to_graph
from utils.models import GraphNetworkAttentionMultimodalWithMapGRU
from utils.training_utils import set_seed


# ========================================================
# Shared colours and agent styles
# ========================================================

SEQUENCE_FILE = PROJECT_ROOT / "docs/figures/qualitative/scene0556_right_turn/sequence.json"

BLUE = "#2463a6"
GREEN = "#288448"
RED = "#d43e37"
GRAY = "#8c96a2"

AGENT_STYLES = {
    "vehicle.car": {
        "color": "#536f83", "marker": "s", "size": 34,
        "legend_size": 5, "label": "C: car",
    },
    "vehicle.truck": {
        "color": "#b16a16", "marker": "s", "size": 64,
        "legend_size": 7, "label": "T: truck",
    },
    "human.pedestrian.adult": {
        "color": "#8951a8", "marker": "o", "size": 42,
        "legend_size": 6, "label": "P: pedestrian",
    },
    "vehicle.motorcycle": {
        "color": "#087e8b", "marker": "D", "size": 40,
        "legend_size": 5, "label": "M: motorcycle",
    },
    "vehicle.bicycle": {
        "color": "#8c564b", "marker": "D", "size": 32,
        "legend_size": 5, "label": "B: bicycle",
    },
    "vehicle.trailer": {
        "color": "#a47739", "marker": "s", "size": 72,
        "legend_size": 8, "label": "TR: trailer",
    },
}

# Each sequence JSON supplies its own title, panel choices and label offsets.


# ========================================================
# Draw one timestamp in the fixed scene coordinate frame
# ========================================================


def draw_frame(ax, frame, bounds, lanes, font_size=9):
    ax.set_facecolor("#fbfcfd")

    # The map stays stationary as the target moves through the scene.
    for lane in lanes:
        ax.plot(lane[:, 0], lane[:, 1], color="#cbd1d7", linewidth=1.05, zorder=1)

    # Faint lines show the other five prediction modes.
    for mode, prediction in enumerate(frame["predictions"]):
        if mode != frame["best_mode"]:
            trajectory = np.vstack((frame["current"], prediction))
            ax.plot(
                trajectory[:, 0], trajectory[:, 1],
                color=GRAY, linewidth=1, linestyle="--", alpha=0.5, zorder=2,
            )

    past = frame["past"]
    future = np.vstack((frame["current"], frame["future"]))
    best = np.vstack((frame["current"], frame["predictions"][frame["best_mode"]]))

    ax.plot(
        past[:, 0], past[:, 1], color=BLUE, linewidth=3,
        marker="o", markersize=3, zorder=4,
    )
    ax.plot(future[:, 0], future[:, 1], color=GREEN, linewidth=3, alpha=0.95, zorder=3)
    ax.scatter(future[::2, 0], future[::2, 1], color=GREEN, s=14, zorder=3)
    ax.plot(best[:, 0], best[:, 1], color=RED, linewidth=2.2, linestyle=(0, (4, 2)), zorder=4)
    ax.scatter(best[-1, 0], best[-1, 1], color=RED, marker="x", s=44, linewidth=1.8, zorder=4)

    # IDs are tied to instance tokens, so they stay with the same agent.
    for agent in frame["agents"]:
        ax.scatter(
            *agent["position"], s=agent["size"], color=agent["color"],
            marker=agent["marker"], edgecolor="white", linewidth=0.65, zorder=6,
        )
        # Dense scenes can label a subset while keeping every agent visible.
        if LABEL_IDS is not None and agent["id"] not in LABEL_IDS:
            continue
        ax.annotate(
            agent["id"], agent["position"],
            xytext=LABEL_OFFSETS[agent["id"]], textcoords="offset points",
            fontsize=font_size, fontweight="bold", color=agent["color"], zorder=7,
            bbox={
                "boxstyle": "round,pad=.12", "facecolor": "white",
                "alpha": 0.9, "edgecolor": "none",
            },
            arrowprops={"arrowstyle": "-", "color": agent["color"], "linewidth": 0.7},
        )

    current = frame["current"]
    heading = np.degrees(frame["heading_rad"])
    ax.scatter(
        *current, marker=(3, 0, heading - 90), s=150,
        color=BLUE, edgecolor="white", linewidth=1, zorder=8,
    )
    ax.annotate(
        "Target car", current, xytext=(-35, 15), textcoords="offset points",
        fontsize=font_size, color=BLUE, fontweight="bold", zorder=9,
        bbox={
            "boxstyle": "round,pad=.15", "facecolor": "white",
            "alpha": 0.9, "edgecolor": "none",
        },
        arrowprops={"arrowstyle": "-", "color": BLUE, "linewidth": 0.7},
    )

    lower, upper = bounds
    ax.set_xlim(lower[0], upper[0])
    ax.set_ylim(lower[1], upper[1])
    ax.set_aspect("equal", adjustable="box")
    ax.grid(color="#e7ebef", linewidth=0.7, zorder=0)
    ax.set_xlabel("Forward from first frame (m)")
    ax.set_ylabel("Left from first frame (m)")


# ========================================================
# Shared legend and static report figure
# ========================================================


def add_legend(fig, categories, y=0.085, font_size=9):
    entries = [
        Line2D([], [], color=BLUE, linewidth=3, label="Target car / observed past"),
        Line2D([], [], color=GREEN, linewidth=3, label="Ground-truth future"),
        Line2D([], [], color=RED, linewidth=2.2, linestyle="--", label="ADE-best of 6 (oracle)"),
        Line2D([], [], color=GRAY, linewidth=1, linestyle="--", label="Other modes"),
    ]
    for category in categories:
        style = AGENT_STYLES[category]
        entries.append(Line2D(
            [], [], color=style["color"], marker=style["marker"], linewidth=0,
            markersize=style["legend_size"], label=style["label"],
        ))
    fig.legend(
        handles=entries, loc="lower center", bbox_to_anchor=(0.5, y),
        ncol=4, frameon=False, fontsize=font_size, columnspacing=1.5,
    )


def save_contact_sheet(frames, bounds, output_path, metadata, categories):
    fig, axes = plt.subplots(1, 3, figsize=(14.8, 6.9))

    for ax, index in zip(axes, PANEL_INDICES):
        frame = frames[index]
        draw_frame(ax, frame, bounds, frames[0]["lanes"], font_size=8.5)
        ax.set_title(
            f"t = {frame['time_s']:.1f} s  |  minADE6 = {frame['minADE6']:.2f} m",
            fontsize=11, pad=12,
        )

    fig.suptitle(metadata["figure"]["title"], fontsize=18, y=0.975, color="#172b43")
    fig.text(
        0.5, 0.90,
        f"{metadata['scene_name']}  |  Same car, fixed reference frame  |  Seed {metadata['seed']}  |  6-second forecasts",
        ha="center", fontsize=11, color="#475569",
    )
    add_legend(fig, categories, y=0.077)
    fig.text(
        0.5, 0.050,
        metadata["figure"].get(
            "agent_note",
            "Agent types follow nuScenes annotations; each ID stays with the same agent across frames.",
        ),
        ha="center", fontsize=9, color="#475569",
    )
    fig.text(
        0.5, 0.026,
        "Highlighted mode selected using ground truth (oracle). Curated example; "
        "agents shown when present in the processed neighbor set.",
        ha="center", fontsize=8.7, color="#475569",
    )
    fig.subplots_adjust(left=0.055, right=0.985, bottom=0.26, top=0.80, wspace=0.22)
    fig.savefig(output_path, dpi=200, facecolor="white")
    plt.close(fig)


# ========================================================
# Load the saved sequence and its trained model
# ========================================================


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sequence", type=Path, default=SEQUENCE_FILE, help="Sequence JSON to reproduce")
    parser.add_argument("--output-dir", type=Path, help="Output folder; default is beside the sequence JSON")
    args = parser.parse_args()

    metadata = json.loads(args.sequence.read_text())
    figure = metadata["figure"]
    PANEL_INDICES = tuple(figure["panel_indices"])
    LABEL_OFFSETS = figure["label_offsets"]
    LABEL_IDS = figure.get("label_ids")
    if args.output_dir is None:
        args.output_dir = args.sequence.parent
    png_path = args.output_dir / f"{figure['output_stem']}.png"
    gif_path = args.output_dir / f"{figure['output_stem']}.gif"
    for path in (png_path, gif_path):
        if path.exists():
            parser.error(f"Output already exists: {path}. Choose a new --output-dir.")

    categories = []
    for agent in metadata["agent_identification"]["neighbors"]:
        if agent["category"] not in categories:
            categories.append(agent["category"])

    with (PROJECT_ROOT / metadata["dataset"]).open("rb") as handle:
        samples = pickle.load(handle)
    selected_samples = [samples[record["index"]] for record in metadata["frames"]]

    # Predict only these ten samples; no full-dataset search or retraining.
    set_seed(metadata["seed"])
    torch.set_num_threads(2)
    model = GraphNetworkAttentionMultimodalWithMapGRU(num_modes=6)
    model.load_state_dict(
        torch.load(PROJECT_ROOT / metadata["checkpoint"], map_location="cpu", weights_only=True),
        strict=True,
    )
    model.eval()
    batch = Batch.from_data_list([sample_to_graph(sample) for sample in selected_samples])
    with torch.no_grad():
        predictions = model(batch).numpy()

    # The first target pose is the origin and heading of the entire animation.
    # Stored poses come from nuScenes annotations, not estimated motion.
    first_pose = metadata["frames"][0]["target_pose"]
    anchor_xy = np.asarray(first_pose["global_xy"])
    anchor_yaw = first_pose["yaw_rad"]
    anchor_rotation = np.array([
        [np.cos(anchor_yaw), -np.sin(anchor_yaw)],
        [np.sin(anchor_yaw), np.cos(anchor_yaw)],
    ])
    start_timestamp = metadata["frames"][0]["timestamp_us"]
    agent_ids = {
        agent["instance_token"]: agent
        for agent in metadata["agent_identification"]["neighbors"]
    }

    # ========================================================
    # Place each timestamp in the same fixed coordinate frame
    # ========================================================

    frames = []
    for index, (record, sample) in enumerate(zip(metadata["frames"], selected_samples)):
        if (sample["sample_token"] != record["sample_token"]
                or sample["target_instance_token"] != metadata["target_instance_token"]):
            raise ValueError("Selected sample does not match the saved sequence metadata")

        pose = record["target_pose"]
        yaw = pose["yaw_rad"]
        rotation = np.array([
            [np.cos(yaw), -np.sin(yaw)],
            [np.sin(yaw), np.cos(yaw)],
        ])

        # Local -> global -> initial target frame (row-vector convention).
        view_rotation = rotation.T @ anchor_rotation
        offset = (np.asarray(pose["global_xy"]) - anchor_xy) @ anchor_rotation

        future = np.asarray(sample["target_future"])
        errors = np.linalg.norm(predictions[index] - future, axis=-1)
        ade_per_mode = errors.mean(axis=-1)
        fde_per_mode = errors[:, -1]

        agents = []
        for token, category, history in zip(
            sample["neighbor_instance_tokens"],
            sample["neighbors_categories"],
            sample["neighbors_pasts"],
        ):
            identity = agent_ids[token]
            if category != identity["category"]:
                raise ValueError(f"Agent category differs from the saved label: {token}")
            style = AGENT_STYLES[category]
            agents.append({
                "id": identity["id"],
                "position": history[-1] @ view_rotation + offset,
                "color": style["color"],
                "marker": style["marker"],
                "size": style["size"],
            })

        frames.append({
            "time_s": (record["timestamp_us"] - start_timestamp) / 1e6,
            "past": sample["target_past"] @ view_rotation + offset,
            "future": future @ view_rotation + offset,
            "predictions": predictions[index] @ view_rotation + offset,
            "lanes": [lane @ view_rotation + offset for lane in sample["map_lanes"] if len(lane)],
            "current": offset,
            "heading_rad": yaw - anchor_yaw,
            "minADE6": float(ade_per_mode.min()),
            "minFDE6": float(fde_per_mode.min()),
            "best_mode": int(ade_per_mode.argmin()),
            "agents": agents,
        })

    # One square view contains all target trajectories and nearby agents.
    visible_points = []
    for frame in frames:
        visible_points.extend((frame["past"], frame["future"], frame["predictions"][frame["best_mode"]]))
        visible_points.append(np.array([agent["position"] for agent in frame["agents"]]))
    points = np.vstack(visible_points)
    lower = points.min(axis=0) - 8
    upper = points.max(axis=0) + 8
    center = (lower + upper) / 2
    side = max(upper - lower)
    bounds = (center - side / 2, center + side / 2)

    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 10,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.titleweight": "normal",
        "axes.labelcolor": "#334155",
        "xtick.color": "#475569",
        "ytick.color": "#475569",
    })

    # ========================================================
    # Save the three-panel PNG and draw the animation frames
    # ========================================================

    args.output_dir.mkdir(parents=True, exist_ok=True)
    save_contact_sheet(frames, bounds, png_path, metadata, categories)

    images = []
    for index, frame in enumerate(frames):
        fig, ax = plt.subplots(figsize=(8.6, 9.3), dpi=120)
        draw_frame(ax, frame, bounds, frames[0]["lanes"])
        fig.suptitle(figure["title"], fontsize=17, y=0.975, color="#172b43")
        fig.text(
            0.5, 0.93,
            f"{metadata['scene_name']}  |  Same target car  |  Seed {metadata['seed']}  |  6-second forecasts",
            ha="center", fontsize=10, color="#475569",
        )
        ax.set_title(
            f"t = {frame['time_s']:.1f} s  |  minADE6 = {frame['minADE6']:.2f} m  |  {index + 1}/{len(frames)}",
            fontsize=11, pad=12,
        )
        add_legend(fig, categories, y=0.082, font_size=8.8)
        fig.text(
            0.5, 0.055,
            figure.get(
                "agent_note",
                "Agent categories are shown in the legend; IDs remain consistent across frames.",
            ),
            ha="center", fontsize=9, color="#475569",
        )
        fig.text(
            0.5, 0.035,
            "Highlighted mode selected using ground truth (oracle). Curated test example.",
            ha="center", fontsize=9, color="#475569",
        )
        fig.text(
            0.5, 0.015,
            "Agents shown when present in the processed neighbor set; no interpolated motion.",
            ha="center", fontsize=9, color="#475569",
        )
        fig.subplots_adjust(left=0.11, right=0.96, bottom=0.26, top=0.855)
        fig.canvas.draw()
        images.append(Image.fromarray(np.asarray(fig.canvas.buffer_rgba())[:, :, :3].copy()))
        plt.close(fig)

    # ========================================================
    # Encode the GIF using the recorded timing and one palette
    # ========================================================

    times = [frame["time_s"] for frame in frames]
    intervals_ms = np.diff(times) * 1000
    durations = [max(10, int(round(interval / 10) * 10)) for interval in intervals_ms]
    durations.append(1200)  # Pause at the last frame before looping.
    durations[0] += 600     # Allow time to inspect the initial forecast.

    # Sample the beginning, middle, and end so all agent colours are included.
    montage = Image.new("RGB", (images[0].width * 3, images[0].height))
    for column, index in enumerate(PANEL_INDICES):
        montage.paste(images[index], (column * images[0].width, 0))
    palette = montage.quantize(colors=192)
    gif_frames = [image.quantize(palette=palette, dither=Image.Dither.NONE) for image in images]
    gif_frames[0].save(
        gif_path, save_all=True, append_images=gif_frames[1:],
        duration=durations, loop=0, optimize=False, disposal=2,
    )

    print(f"Saved {png_path}")
    print(f"Saved {gif_path}")
    print("Highlighted mode: minimum ground-truth ADE of six modes at each timestamp.")
