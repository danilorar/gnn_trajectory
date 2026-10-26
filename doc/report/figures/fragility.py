"""Why the final model is easy to fool: error under attack, input removal, candidate coverage.

Run from the project root:
    python doc/report/figures/fragility.py

(a) minADE6 / minFDE6 clean, under same-size random noise and under PGD, and (c) candidate
coverage and spread, both from doc/poster/figures/summary.json (500 val samples, 3 seeds).
(b) minADE6 added when one input is removed, on the full val set (issue #22, per seed).
"""

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import numpy as np

HERE = Path(__file__).resolve().parent
SUMMARY = HERE.parents[1] / "poster/figures/summary.json"
WIDTH = 3.5  # IEEE \columnwidth in inches
CLEAN_C, ATTACK_C, LANE_C = "#2563b0", "#eb6834", "#c4a35a"
INK, MUTED, BASE, WHITE = "#1a1a1a", "#6b6b6b", "#b9b9b5", "#ffffff"

# Final model: minADE6 change when one input is removed, seeds 42 / 123 / 2026 (issue #22)
ABLATION = [("Target history", [7.223, 6.430, 5.819], ATTACK_C),
            ("Lane map", [1.181, 1.702, 1.495], LANE_C),
            ("Neighbours", [0.0, 0.0, 0.0], BASE)]

plt.rcParams.update({"font.family": "serif", "font.serif": ["Times New Roman", "Times", "STIXGeneral"],
                     "mathtext.fontset": "stix", "font.size": 7, "pdf.fonttype": 42,
                     "axes.spines.top": False, "axes.spines.right": False, "axes.linewidth": 0.5,
                     "xtick.major.width": 0.5, "ytick.major.width": 0.5, "xtick.major.size": 2,
                     "ytick.major.size": 2, "xtick.color": MUTED, "ytick.color": MUTED})


def bar_styles():
    return [dict(facecolor=CLEAN_C), dict(facecolor=WHITE, edgecolor=CLEAN_C, lw=0.8), dict(facecolor=ATTACK_C)]


def main():
    final = json.loads(SUMMARY.read_text())
    gru, modes = final["models"]["multimodal_map_gru"], final["modes_final"]

    fig, (ax_err, ax_abl, ax_cov) = plt.subplots(1, 3, figsize=(WIDTH, 1.45),
                                                 gridspec_kw={"width_ratios": [1.15, 1.25, 0.95], "wspace": 0.35})

    # (a) error: clean, random noise of the same size, PGD
    width = 0.26
    for j, metric in enumerate(("ade", "fde")):
        for i, (cond, style) in enumerate(zip(("clean", "random", "attacked"), bar_styles())):
            m, s = gru[metric][cond]["mean"], gru[metric][cond]["std"]
            ax_err.bar(j + (i - 1) * width, m, width * 0.92, yerr=s, error_kw=dict(lw=0.5, capsize=0), **style)
            if cond != "random":
                ax_err.text(j + (i - 1) * width, m + s + 0.2, f"{m:.1f}", ha="center", va="bottom", fontsize=6,
                            color=ATTACK_C if cond == "attacked" else INK)
    ax_err.set_xticks([0, 1], ["minADE$_6$", "minFDE$_6$"])
    ax_err.tick_params(axis="x", length=0, colors=INK)
    ax_err.set_ylim(0, 7.2); ax_err.set_yticks([0, 2, 4, 6])

    # (b) input removal: added minADE6 per seed (dots) and mean (bar)
    for i, (name, values, color) in enumerate(ABLATION):
        y = len(ABLATION) - 1 - i
        ax_abl.barh(y, np.mean(values), 0.42, color=color)
        ax_abl.scatter(values, [y] * 3, s=3, color=INK, zorder=3, lw=0)
        label = f"+{np.mean(values):.1f}" if np.mean(values) > 0 else "≈0.001 m"
        ax_abl.text(max(values) + 0.25, y, label, va="center", fontsize=6, color=INK)
        ax_abl.text(0, y + 0.27, name, va="bottom", fontsize=6.5, color=INK)
    ax_abl.set_yticks([]); ax_abl.set_ylim(-0.35, len(ABLATION) - 0.2)
    ax_abl.set_xlim(0, 8.8); ax_abl.set_xticks([0, 4, 8])
    ax_abl.spines["left"].set_visible(False)

    # (c) candidate coverage (share of samples with a candidate within 2 m at every step) and spread
    conditions = [("clean", modes["clean"]), ("random", modes["random"]), ("attacked", modes["attacked"]["ade"])]
    for i, ((cond, m), style) in enumerate(zip(conditions, bar_styles())):
        coverage = 100 * (1 - m["miss"]["mean"])
        ax_cov.bar(i, coverage, 0.7, **style)
        ax_cov.text(i, coverage + 1.5, f"{coverage:.0f}%", ha="center", va="bottom", fontsize=6,
                    color=ATTACK_C if cond == "attacked" else INK)
        ax_cov.text(i, -6, f"{m['spread']['mean']:.1f}", ha="center", va="top", fontsize=5.5, color=MUTED)
    ax_cov.text(-0.75, -6, "spread\n(m)", ha="right", va="top", fontsize=5.5, color=MUTED, linespacing=0.9)
    ax_cov.set_xticks([]); ax_cov.set_ylim(0, 45); ax_cov.set_yticks([0, 20, 40])

    fig.legend(handles=[Patch(label="clean", **bar_styles()[0]), Patch(label="random, same size", **bar_styles()[1]),
                        Patch(label="PGD attack", **bar_styles()[2])],
               loc="lower center", bbox_to_anchor=(0.5, 0.93), ncol=3, frameon=False,
               handlelength=1.2, handleheight=0.7, columnspacing=1.2)

    # Subfigure labels below the panels, in the caption font
    fig.subplots_adjust(left=0.07, right=0.99, top=0.88, bottom=0.24)
    fig.canvas.draw()
    renderer, to_fig = fig.canvas.get_renderer(), fig.transFigure.inverted()
    labels = {ax_err: "(a) Error (m)", ax_abl: "(b) Input removal (m)",
              ax_cov: "(c) Within 2 m (%)"}
    label_y = min(a.get_tightbbox(renderer).transformed(to_fig).y0 for a in labels) - 0.02
    for a, text in labels.items():
        box = a.get_position()
        fig.text((box.x0 + box.x1) / 2, label_y, text, ha="center", va="top", fontsize=7)

    fig.savefig(HERE / "fragility.pdf", facecolor=WHITE, bbox_inches="tight", pad_inches=0.01)


if __name__ == "__main__":
    main()
