"""Results figures for the poster, one per question.

results_final:  is the final model vulnerable? Clean vs random noise vs PGD for minADE6, minFDE6
                and the share pushed beyond half a lane.
results_models: which component adds robustness? Clean -> PGD share beyond half a lane per model,
                baselines first, our final model last.

Reads summary.json from aggregate.py. Clean and attacked share the colours of the real and
crafted past in the hero figure.
"""

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

HERE = Path(__file__).resolve().parent
CLEAN_C, ATTACK_C = "#2563b0", "#eb6834"
INK, MUTED, GRID, WHITE, HIGHLIGHT = "#1a1a1a", "#6b6b6b", "#e4e4e1", "#ffffff", "#fdeee6"
ORDER = ["cv", "target_mlp", "attention", "multimodal", "multimodal_map", "multimodal_map_gru"]
LABELS = {"cv": "Constant velocity", "target_mlp": "Target-only MLP", "attention": "GATv2 + neighbours",
          "multimodal": "GATv2, 6 modes", "multimodal_map": "+ lane map, MLP decoder",
          "multimodal_map_gru": "+ lane map, GRU decoder"}
FINAL = ORDER[-1]
PANELS = [("lateral", "Lateral attacks (left / right)"), ("longitudinal", "Longitudinal attacks (front / rear)")]

plt.rcParams.update({"font.family": "Arial", "font.size": 12})


def style(ax):
    for side in ax.spines.values():
        side.set_visible(False)
    ax.tick_params(length=0, colors=MUTED, labelsize=11)


def final_model(models):
    """(a) Final model: clean, random noise and PGD on three metrics."""
    m = models[FINAL]
    metrics = [
        ("minADE$_6$\n(m)", [m["ade"][s] for s in ("clean", "random", "attacked")], 1, "{:.2f}"),
        ("minFDE$_6$\n(m)", [m["fde"][s] for s in ("clean", "random", "attacked")], 1, "{:.2f}"),
        ("> half a lane\n(%)", [m[f"share_{s}"]["all"] for s in ("clean", "random", "attacked")], 100, "{:.0f}%"),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(3.6, 2.6),
                             gridspec_kw=dict(wspace=0.12, left=0.01, right=0.99, top=0.67, bottom=0.03))
    faces, edges = [CLEAN_C, WHITE, ATTACK_C], [CLEAN_C, CLEAN_C, ATTACK_C]
    for ax, (title, values, scale, fmt) in zip(axes, metrics):
        means = [v["mean"] * scale for v in values]
        stds = [v["std"] * scale for v in values]
        ax.bar(range(3), means, width=0.72, color=faces, edgecolor=edges, linewidth=1.6,
               yerr=stds, error_kw=dict(ecolor=MUTED, lw=1, capsize=0))
        for x, (mean, std) in enumerate(zip(means, stds)):
            ax.text(x, mean + std + max(means) * 0.03, fmt.format(mean), ha="center", va="bottom", fontsize=11,
                    color=ATTACK_C if x == 2 else INK, weight="bold" if x == 2 else "normal")
        ax.set_ylim(0, max(means) * 1.28)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.axhline(0, color=MUTED, lw=0.8)
        style(ax)
        ax.set_title(title, fontsize=11.5, color=INK, weight="bold", pad=4, linespacing=1.1)
    handles = [Patch(facecolor=CLEAN_C, edgecolor=CLEAN_C, label="clean"),
               Patch(facecolor=WHITE, edgecolor=CLEAN_C, lw=1.6, label="random noise"),
               Patch(facecolor=ATTACK_C, edgecolor=ATTACK_C, label="PGD attack")]
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, 1.02), ncol=3, frameon=False,
               fontsize=11, handlelength=1.0, handletextpad=0.3, columnspacing=0.8)
    return fig


def per_model(models):
    """(b) Clean -> PGD share beyond half a lane for every model; baselines above, ours below."""
    fig, axes = plt.subplots(1, 2, figsize=(8.3, 2.85), sharey=True,
                             gridspec_kw=dict(wspace=0.09, left=0.275, right=0.975, top=0.82, bottom=0.17))
    # simplest model on top; ours at the bottom, set apart in its own band
    rows = [len(ORDER) + 0.6 - i for i in range(1, len(ORDER))] + [0]
    band = (-0.5, 0.95)
    for ax, (group, title) in zip(axes, PANELS):
        ax.axhspan(*band, color=HIGHLIGHT, zorder=0)
        for y, key in zip(rows, ORDER):
            m = models[key]
            clean = m["share_clean"][group]["mean"] * 100
            attacked = m["share_attacked"][group]["mean"] * 100
            std = m["share_attacked"][group]["std"] * 100
            ax.plot([clean, attacked], [y, y], color=GRID, lw=3, solid_capstyle="round", zorder=1)
            if std:
                ax.plot([attacked - std, attacked + std], [y, y], color=ATTACK_C, lw=1.2, alpha=0.55, zorder=2)
            ax.scatter(clean, y, s=70, color=CLEAN_C, edgecolors=WHITE, lw=1.5, zorder=3)
            ax.scatter(attacked, y, s=80, color=ATTACK_C, edgecolors=WHITE, lw=1.5, zorder=5)
            ax.annotate(f"{attacked:.0f}%", (attacked + std, y), xytext=(7, 0), textcoords="offset points",
                        va="center", fontsize=12, color=INK, weight="bold" if key == FINAL else "normal")
        ax.axhline(band[1], color=MUTED, lw=0.8)
        ax.set_xlim(-6, 102)
        ax.set_xticks([0, 25, 50, 75, 100])
        ax.xaxis.grid(True, color=GRID, lw=0.8)
        ax.set_axisbelow(True)
        style(ax)
        ax.set_title(title, fontsize=12.5, color=INK, weight="bold", loc="left", pad=4)
    axes[0].set_yticks(rows)
    axes[0].set_yticklabels([LABELS[k] for k in ORDER], fontsize=12, color=INK)
    axes[0].get_yticklabels()[-1].set_weight("bold")
    for y, text in ((rows[0] + 0.3, "Baselines"), (rows[-1] + 0.3, "Ours")):
        axes[0].annotate(text, (0, y), xycoords=("figure fraction", "data"), xytext=(4, 0), textcoords="offset points",
                         fontsize=11, color=MUTED, ha="left", va="bottom", style="italic", weight="bold")
    axes[0].set_ylim(band[0] - 0.1, rows[0] + 0.75)
    fig.text(0.625, 0.025, "% of 500 val samples pushed > 1.85 m (half a lane) in the attacked direction",
             fontsize=11, color=MUTED, ha="center")
    handles = [Line2D([], [], marker="o", ls="none", color=CLEAN_C, ms=8, label="clean"),
               Line2D([], [], marker="o", ls="none", color=ATTACK_C, ms=8, label="PGD attack (± seed std)")]
    fig.legend(handles=handles, loc="upper left", bbox_to_anchor=(0.265, 1.0), ncol=2, frameon=False,
               fontsize=11.5, handletextpad=0.3, columnspacing=1.4)
    return fig


def main():
    models = json.loads((HERE / "summary.json").read_text())["models"]
    for name, make in (("results_final", final_model), ("results_models", per_model)):
        fig = make(models)
        for ext in ("pdf", "png"):
            fig.savefig(HERE / f"{name}.{ext}", dpi=300, facecolor=WHITE)
        plt.close(fig)


if __name__ == "__main__":
    main()
