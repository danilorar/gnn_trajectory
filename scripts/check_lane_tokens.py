"""Visual check of utils/lane_tokens.py on one nuScenes prediction target.

Checks:
- arrows near the target point roughly along +x
- arrows follow the traffic direction
- edges connect the end of one segment to the start of the next (gap printed below)
"""

import argparse
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np
from pyquaternion import Quaternion

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from nuscenes.nuscenes import NuScenes
from nuscenes.eval.prediction.splits import get_prediction_challenge_split

from scripts.build_processed_dataset import get_map_for_sample
from utils.lane_tokens import build_lane_tokens


def get_target_pose(nusc, target_string):
    """Same origin/yaw as process_prediction_target in build_processed_dataset.py."""
    instance_token, sample_token = target_string.split("_")
    sample_record = nusc.get("sample", sample_token)

    for ann_token in sample_record["anns"]:
        ann = nusc.get("sample_annotation", ann_token)
        if ann["instance_token"] == instance_token:
            origin = np.asarray(ann["translation"][:2], dtype=np.float32)
            yaw = Quaternion(ann["rotation"]).yaw_pitch_roll[0]
            return sample_record, origin, yaw

    raise ValueError(f"target not found: {target_string}")


def plot_lane_tokens(lane_points, lane_is_connector, lane_edges, radius, arrow_every=5):
    fig, ax = plt.subplots(figsize=(9, 9))
    colors = plt.cm.tab20(np.linspace(0, 1, 20))

    # Segments, each in its own colour (connectors dashed)
    for i, pts in enumerate(lane_points):
        ax.plot(pts[:, 0], pts[:, 1], color=colors[i % 20], lw=2,
                ls="--" if lane_is_connector[i] else "-")

    # A few direction arrows per segment from cos(psi), sin(psi)
    arrows = np.concatenate([pts[::arrow_every] for pts in lane_points])
    ax.quiver(arrows[:, 0], arrows[:, 1], arrows[:, 2], arrows[:, 3],
              angles="xy", scale_units="xy", scale=0.4, width=0.003, color="k")

    # Edges: end of the from-segment -> start of the to-segment
    for a, b in lane_edges:
        p, q = lane_points[a][-1], lane_points[b][0]
        ax.plot([p[0], q[0]], [p[1], q[1]], color="red", lw=1)
        ax.plot(q[0], q[1], "o", ms=3, color="red")

    # Target at the origin, heading along +x
    ax.quiver(0, 0, 1, 0, angles="xy", scale_units="xy", scale=0.2, color="blue", width=0.008)
    ax.set_xlim(-radius, radius)
    ax.set_ylim(-radius, radius)
    ax.set_aspect("equal")
    ax.grid(alpha=0.3)
    ax.set_title(f"{len(lane_points)} segments, {len(lane_edges)} edges "
                 "(dashed = connector, red = edges, blue = target)")
    return fig


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot lane tokens for one prediction target.")
    parser.add_argument("--version", default="v1.0-trainval")
    parser.add_argument("--split", default="train")
    parser.add_argument("--index", type=int, default=0, help="target index in the split")
    parser.add_argument("--radius", type=float, default=50.0)
    parser.add_argument("--seg-len", type=int, default=20)
    parser.add_argument("--out", default=str(PROJECT_ROOT / "outputs" / "check_lane_tokens.png"))
    args = parser.parse_args()

    data_dir = str(PROJECT_ROOT / "data")
    nusc = NuScenes(version=args.version, dataroot=data_dir, verbose=False)
    target = get_prediction_challenge_split(args.split, dataroot=data_dir)[args.index]

    sample_record, origin, yaw = get_target_pose(nusc, target)
    nusc_map = get_map_for_sample(nusc, sample_record)

    lane_points, lane_is_connector, lane_edges = build_lane_tokens(
        nusc_map, origin, yaw, radius=args.radius, seg_len=args.seg_len)

    # Numeric checks
    gaps = [np.linalg.norm(lane_points[a][-1, :2] - lane_points[b][0, :2]) for a, b in lane_edges]
    near = [p[np.linalg.norm(p[:, :2], axis=1) < 5] for p in lane_points]
    near = np.concatenate([n for n in near if len(n)]) if any(len(n) for n in near) else np.zeros((0, 4))

    print(f"target: {target}")
    print(f"segments: {len(lane_points)} ({lane_is_connector.sum()} connectors), edges: {len(lane_edges)}")
    if gaps:
        print(f"edge gap end->start: max {max(gaps):.2f} m, mean {np.mean(gaps):.2f} m")
    if len(near):
        print(f"lane points within 5 m of target: {len(near)}, "
              f"max cos(psi) = {near[:, 2].max():.2f} (close to 1 = along +x)")

    fig = plot_lane_tokens(lane_points, lane_is_connector, lane_edges, args.radius)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=150, bbox_inches="tight")
    print(f"saved {args.out}")
    plt.show()
