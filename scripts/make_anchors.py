"""Endpoint anchors from train-split target endpoints (target frame, 6 s ahead).

--method kmeans (default): plain k-means, saves assets/anchors.npy.
--method angle: one anchor per turn direction so rare sharp turns get their own mode.
    Endpoints are grouped by angle (sharp right < -60, right -60..-15, straight |a| < 15,
    left 15..60, sharp left > 60 deg; |a| > 135 deg, i.e. reversing, is left out of the sharp groups).
    Straight gets K - 4 k-means anchors (speed spread), each turn group its median endpoint.
    Saves assets/anchors_angle.npy.
"""

import argparse
import pickle
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import KMeans

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Make k-means endpoint anchors.")
    parser.add_argument("--k", type=int, default=6)
    parser.add_argument("--method", choices=("kmeans", "angle"), default="kmeans")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--data", default=str(PROJECT_ROOT / "data/processed_v2/nuscenes_train_prediction_samples_map.pkl"))
    parser.add_argument("--out", help="Default: assets/anchors.npy or assets/anchors_angle.npy")
    parser.add_argument("--plot", help="Default: outputs/anchors.png or outputs/anchors_angle.png")
    args = parser.parse_args()
    suffix = "" if args.method == "kmeans" else "_angle"
    args.out = args.out or str(PROJECT_ROOT / f"assets/anchors{suffix}.npy")
    args.plot = args.plot or str(PROJECT_ROOT / f"outputs/anchors{suffix}.png")

    with open(args.data, "rb") as f:
        samples = pickle.load(f)
    endpoints = np.stack([s["target_future"][-1] for s in samples])  # (N, 2)

    if args.method == "kmeans":
        kmeans = KMeans(n_clusters=args.k, n_init=10, random_state=args.seed).fit(endpoints)
        anchors = kmeans.cluster_centers_.astype(np.float32)
        anchors = anchors[np.argsort(anchors[:, 0])]  # sort by forward distance for readability
    else:
        if args.k < 5:
            parser.error("--method angle needs --k >= 5")
        ang = np.degrees(np.arctan2(endpoints[:, 1], endpoints[:, 0]))
        straight = np.abs(ang) < 15
        groups = [(ang <= -60) & (ang > -135), (ang <= -15) & (ang > -60),
                  (ang >= 15) & (ang < 60), (ang >= 60) & (ang < 135)]
        turn = [np.median(endpoints[g], axis=0) for g in groups]
        km = KMeans(n_clusters=args.k - 4, n_init=10, random_state=args.seed).fit(endpoints[straight])
        fwd = km.cluster_centers_[np.argsort(km.cluster_centers_[:, 0])]
        anchors = np.concatenate([fwd, turn]).astype(np.float32)
    np.save(args.out, anchors)

    nearest = np.linalg.norm(endpoints[:, None] - anchors[None], axis=-1).argmin(1)
    counts = np.bincount(nearest, minlength=args.k)
    print(f"{len(endpoints)} endpoints -> {args.k} anchors, saved {args.out}")
    for k, (a, n) in enumerate(zip(anchors, counts)):
        angle = np.degrees(np.arctan2(a[1], a[0]))
        print(f"  anchor {k}: x {a[0]:6.1f} m, y {a[1]:6.1f} m, angle {angle:6.1f} deg, share {n / len(endpoints):5.1%}")

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.scatter(endpoints[:, 0], endpoints[:, 1], s=1, alpha=0.05, color="gray", label="target endpoints (train)")
    ax.scatter(anchors[:, 0], anchors[:, 1], s=120, marker="*", color="red", edgecolor="k", zorder=3, label="anchors")
    for k, a in enumerate(anchors):
        ax.annotate(str(k), a, xytext=(6, 6), textcoords="offset points", fontsize=11, weight="bold")
    ax.quiver(0, 0, 1, 0, angles="xy", scale_units="xy", scale=0.1, color="blue", width=0.006, label="target heading")
    ax.set_xlim(-30, 90)
    ax.set_ylim(-60, 60)
    ax.set_aspect("equal")
    ax.grid(alpha=0.3)
    ax.set_xlabel("x (m, forward)")
    ax.set_ylabel("y (m, left)")
    ax.set_title(f"K = {args.k} k-means anchors on 6 s target endpoints")
    ax.legend(loc="upper left")
    Path(args.plot).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.plot, dpi=150, bbox_inches="tight")
    print(f"saved {args.plot}")
