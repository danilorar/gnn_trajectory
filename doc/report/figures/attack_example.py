"""Report version of the poster's attack example: val sample 4640, right attack, GRU seed 42.

Run from the project root:
    python doc/report/figures/attack_example.py

Same scene and data as doc/poster/figures/hero_example.py, restyled for the IEEE paper:
Times at caption size, subfigure labels below the panels, no camera panel. Neighbour
positions come from the processed sample (target frame, t = 0), so no nuScenes metadata
is needed.
"""

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.adversarial_attack import (
    CHECKPOINT_DIRS, build_inputs, checkpoint_path, dataset_path, load_model, load_samples,
    motion_directions, perturbed_forward, sample_metrics,
)
from tools.animate_attack_example import (
    AGENT_STYLE, BEST_C, CRAFTED_C, INK, LANE_C, MODE_C, MUTED, PAST, PAST_C, TRUTH_C,
)
from utils.experiment_utils import sha256
from utils.graph_utils import general_category

OUT_DIR = Path(__file__).resolve().parent
SEED, OBJECTIVE, VAL_INDEX = 42, "right", 4640
WHITE = "#ffffff"
WIDTH = 7.16  # IEEE conference \textwidth in inches

plt.rcParams.update({"font.family": "serif", "font.serif": ["Times New Roman", "Times", "STIXGeneral"],
                     "mathtext.fontset": "stix", "font.size": 7, "pdf.fonttype": 42})


def main():
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

    agents = [(general_category(c), p[-1]) for p, c in zip(sample["neighbors_pasts"], sample["neighbors_categories"])]

    points = np.vstack([history, crafted, truth, preds["clean"][best["clean"]], preds["attacked"][best["attacked"]]])
    lo, hi = points.min(0) - np.array([6, 9]), points.max(0) + np.array([6, 9])
    centre, half = (lo + hi) / 2, (hi - lo) / 2
    half = np.maximum(half, [half[1] * 1.45, half[0] / 1.45])
    xlim, ylim = (centre[0] - half[0], centre[0] + half[0]), (centre[1] - half[1], centre[1] + half[1])

    fig = plt.figure(figsize=(WIDTH, 1.95))
    grid = fig.add_gridspec(1, 3, width_ratios=[1.45, 1.45, 1.0], wspace=0.06,
                            left=0.005, right=0.995, top=0.99, bottom=0.30)
    ax = {"clean": fig.add_subplot(grid[0]), "attacked": fig.add_subplot(grid[1])}
    ax_hist = fig.add_subplot(grid[2])

    for name, a in ax.items():
        for lane in sample["map_lanes"]:
            a.plot(*lane.T, color=LANE_C, lw=0.5, zorder=1)
        for kind, p in agents:
            if not (xlim[0] < p[0] < xlim[1] and ylim[0] < p[1] < ylim[1]):
                continue
            marker, color, _ = AGENT_STYLE[kind]
            a.scatter(*p, marker=marker, s=9 if kind != "pedestrian" else 3, color=color, zorder=3, lw=0)
        observed = crafted if name == "attacked" else history
        for m, mode in enumerate(preds[name]):
            if m != best[name]:
                a.plot(*np.vstack([observed[-1:], mode]).T, "--", color=MODE_C, lw=0.5, zorder=4)
        a.plot(*np.vstack([observed[-1:], truth]).T, color=TRUTH_C, lw=1.3, zorder=6, solid_capstyle="round")
        a.plot(*np.vstack([observed[-1:], preds[name][best[name]]]).T, "--", color=BEST_C, lw=1.2, zorder=7)
        if name == "attacked":
            a.plot(*history.T, color=PAST_C, lw=0.7, alpha=0.4, zorder=5)
        a.plot(*observed.T, "-o", color=CRAFTED_C if name == "attacked" else PAST_C, lw=1.1, ms=1.6, zorder=8)
        a.scatter(*observed[-1], s=14, color=CRAFTED_C if name == "attacked" else PAST_C,
                  edgecolors=WHITE, lw=0.5, zorder=9)
        a.text(0.03, 0.05, f"minADE$_6$ {metrics[name]['ade']:.2f} m\nbest mode "
               f"{metrics[name][OBJECTIVE]:.2f} m right", transform=a.transAxes, va="bottom",
               color=INK, bbox=dict(boxstyle="round,pad=0.25", fc=WHITE, ec=LANE_C, lw=0.5), zorder=10)
        a.set_xlim(xlim); a.set_ylim(ylim); a.set_aspect("equal")
        a.set_xticks([]); a.set_yticks([])
        for side in a.spines.values():
            side.set_color(LANE_C); side.set_linewidth(0.6)
    scale_x = xlim[0] + 0.72 * (xlim[1] - xlim[0])
    scale_y = ylim[0] + 0.08 * (ylim[1] - ylim[0])
    ax["clean"].plot([scale_x, scale_x + 10], [scale_y, scale_y], color=MUTED, lw=0.9)
    ax["clean"].text(scale_x + 5, scale_y + 0.6, "10 m", ha="center", va="bottom", color=MUTED)

    # (c) the observed history only, with the lateral axis stretched so centimetre offsets show
    for i in range(PAST + 1):
        ax_hist.plot(*np.stack([history[i], crafted[i]]).T, color=MUTED, lw=0.5, zorder=2)
        below = crafted[i, 1] < history[i, 1]
        side = 5 if 0 < i < PAST and below else 0  # keep inner labels clear of the incoming segment
        ax_hist.annotate(f"{size_cm[i]:.0f} cm", crafted[i], xytext=(side, -9 if below else 4),
                         textcoords="offset points", ha="left" if side else "center", color=INK)
    ax_hist.plot(*history.T, "-o", color=PAST_C, lw=1.1, ms=2.5, zorder=3)
    ax_hist.plot(*crafted.T, "-o", color=CRAFTED_C, lw=1.1, ms=2.5, zorder=4)
    for i in (0, PAST):
        ax_hist.annotate("now" if i == PAST else f"{(i - PAST) * 0.5:+.0f} s", history[i], xytext=(0, 4),
                         textcoords="offset points", ha="center", color=MUTED)
    ys = np.r_[history[:, 1], crafted[:, 1]]
    ax_hist.set_ylim(ys.min() - 0.12, ys.max() + 0.12)
    ax_hist.set_xlim(history[:, 0].min() - 1.5, history[:, 0].max() + 1.5)
    ax_hist.set_yticks([]); ax_hist.tick_params(colors=MUTED, length=2, width=0.5, pad=1.5)
    ax_hist.set_xlabel("metres along heading (lateral offsets exaggerated)", color=INK, labelpad=1.5)
    for side in ax_hist.spines.values():
        side.set_color(LANE_C); side.set_linewidth(0.6)

    # Subfigure labels below the panels, in the caption font
    fig.canvas.draw()
    renderer, to_fig = fig.canvas.get_renderer(), fig.transFigure.inverted()
    labels = {ax["clean"]: "(a) Real history", ax["attacked"]: f"(b) Crafted history, {size_cm.mean():.0f} cm per point",
              ax_hist: "(c) Observed points, real and crafted"}
    label_y = min(a.get_tightbbox(renderer).transformed(to_fig).y0 for a in labels) - 0.015
    for a, text in labels.items():
        box = a.get_position()
        fig.text((box.x0 + box.x1) / 2, label_y, text, ha="center", va="top", fontsize=8)

    handles = [Line2D([], [], color=PAST_C, lw=1.1, marker="o", ms=1.6, label="real past"),
               Line2D([], [], color=CRAFTED_C, lw=1.1, marker="o", ms=1.6, label="crafted past"),
               Line2D([], [], color=TRUTH_C, lw=1.3, label="true future"),
               Line2D([], [], color=BEST_C, lw=1.2, ls="--", label="best of 6 modes"),
               Line2D([], [], color=MODE_C, lw=0.5, ls="--", label="other modes")]
    handles += [Line2D([], [], color=c, marker=mk, ls="none", ms=3.5, label=kind.replace("_", " "))
                for kind, (mk, c, _) in AGENT_STYLE.items() if any(a[0] == kind for a in agents)]
    fig.legend(handles=handles, loc="upper center", ncol=len(handles), frameon=False,
               bbox_to_anchor=(0.5, label_y - 0.085), columnspacing=1.2, handlelength=2.0, handletextpad=0.5)

    fig.savefig(OUT_DIR / "attack_example.pdf", facecolor=WHITE, bbox_inches="tight", pad_inches=0.01)
    print({n: {k: round(v, 3) for k, v in m.items()} for n, m in metrics.items()}, "perturbation cm", np.round(size_cm, 1))


if __name__ == "__main__":
    main()
