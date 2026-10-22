"""Why is the model easy to fool? One small chart per reason, placed next to its text on the poster.

why_amplification: minADE6 added per metre of mean perturbation under the ADE attack, from the
                   PGD runs. Even constant velocity, with no learned weights, amplifies the shift.
why_inputs:        minADE6 added when one input of the final model is removed (issue #22 ablation,
                   seeds 42 / 123 / 2026, numbers from drafts/issue22_reply_draft.md). "Target's
                   motion" sets all five past positions to the current one.
"""

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
ADV = HERE.parents[2] / "data/processed/adversarial"
SEEDS = (42, 123, 2026)
CLEAN_C, ATTACK_C, LANE_C = "#2563b0", "#eb6834", "#c4a35a"
INK, MUTED, BASE, WHITE = "#1a1a1a", "#6b6b6b", "#b9b9b5", "#ffffff"

AMPLIFICATION = [("cv", "Constant velocity", BASE), ("attention", "GATv2 + neighbours", BASE),
                 ("multimodal_map_gru", "Ours (best of 6)", CLEAN_C)]
# Final model: minADE6 change when one input is removed, per seed (issue #22)
ABLATION = [("Target's motion", [7.223, 6.430, 5.819], ATTACK_C),
            ("Lane map", [1.181, 1.702, 1.495], LANE_C),
            ("Neighbours", [0.0, 0.0, 0.0], BASE)]

plt.rcParams.update({"font.family": "Arial", "font.size": 12})


def amplification(spec):
    """Mean over seeds of (attacked - clean minADE) / mean perturbation per point, ADE attack."""
    combined = json.loads((ADV / "results_val_b3_combined.json").read_text())["models"]
    if spec == "cv":
        runs = [combined["cv"]]
    else:
        runs = [combined.get(f"{spec}:{s}") or
                json.loads((ADV / f"results_val_b3_{spec}_seed{s}.json").read_text())["models"][f"{spec}:{s}"]
                for s in SEEDS]
    gains = [(e["attacked"]["ade"] - e["clean"]["ade"]) / e["perturbation"]["ade"]["mean_point_m"] for e in runs]
    return sum(gains) / len(gains)


def bar_chart(labels, values, colors, texts, title, xmax):
    fig = plt.figure(figsize=(2.75, 1.45))
    ax = fig.add_axes([0.54, 0.03, 0.29, 0.68])
    rows = list(range(len(labels)))[::-1]
    ax.barh(rows, values, height=0.62, color=colors)
    for y, v, t in zip(rows, values, texts):
        ax.text(v + xmax * 0.03, y, t, va="center", ha="left", fontsize=12, color=INK)
    ax.set_yticks(rows)
    ax.set_yticklabels(labels, fontsize=12, color=INK)
    ax.set_xlim(0, xmax)
    ax.set_xticks([])
    for side in ax.spines.values():
        side.set_visible(False)
    ax.axvline(0, color=MUTED, lw=0.8)
    ax.tick_params(length=0)
    fig.text(0.0, 0.79, title, fontsize=12, color=INK, weight="bold", ha="left", va="bottom")
    return fig, ax


def main():
    gains = [amplification(spec) for spec, _, _ in AMPLIFICATION]
    fig, ax = bar_chart([label for _, label, _ in AMPLIFICATION], gains, [c for _, _, c in AMPLIFICATION],
                        [f"{g:.0f} m" for g in gains], "minADE$_6$ added per 1 m of shift", 28)
    ax.get_yticklabels()[-1].set_weight("bold")
    charts = {"why_amplification": fig}

    means = [sum(v) / len(v) for _, v, _ in ABLATION]
    texts = [f"+{m:.1f} m" if m else "≈0.001 m" for m in means]
    fig, ax = bar_chart([label for label, _, _ in ABLATION], means, [c for _, _, c in ABLATION], texts,
                        "minADE$_6$ added if removed", 9.5)
    ax.get_yticklabels()[0].set_color(ATTACK_C)
    ax.get_yticklabels()[0].set_weight("bold")
    charts["why_inputs"] = fig

    for name, fig in charts.items():
        for ext in ("pdf", "png"):
            fig.savefig(HERE / f"{name}.{ext}", dpi=300, facecolor=WHITE, bbox_inches="tight", pad_inches=0.03)
        plt.close(fig)


if __name__ == "__main__":
    main()
