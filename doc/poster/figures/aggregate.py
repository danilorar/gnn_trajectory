"""Aggregate the PGD attack results over seeds for the poster (prints a table, writes summary.json).

Reads the per-run JSONs written by scripts/adversarial_attack.py. All runs cover the same
500 validation samples; this is checked.
"""

import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
ADV = ROOT / "data/processed/adversarial"
MODES = ROOT / "data/processed/adversarial_modes"
OUT = Path(__file__).resolve().parent / "summary.json"
SEEDS = (42, 123, 2026)
DIRECTIONS = ("left", "right", "front", "rear")

# Model ladder, simplest first: (key, label, spec name, results file per seed)
LADDER = [
    ("cv", "Constant velocity", "cv", None),
    ("target_mlp", "Target-only MLP", "target_mlp", "results_val_b3_target_mlp_seed{}.json"),
    ("attention", "GATv2 (neighbours)", "attention", "results_val_b3_attention_seed{}.json"),
    ("multimodal", "GATv2, 6 modes", "multimodal", None),
    ("multimodal_map", "+ lane map, MLP dec.", "multimodal_map", None),
    ("multimodal_map_gru", "+ lane map, GRU dec. (final)", "multimodal_map_gru", None),
]


def load(path):
    return json.loads(path.read_text())


def entries(spec, pattern, combined):
    """One results entry per seed (or the single CV run)."""
    if spec == "cv":
        return [combined["models"]["cv"]]
    if pattern is None:
        return [combined["models"][f"{spec}:{s}"] for s in SEEDS]
    out = []
    for s in SEEDS:
        run = load(ADV / pattern.format(s))
        if run["sample_indices"] != combined["sample_indices"]:
            raise ValueError(f"{pattern.format(s)} covers different samples")
        out.append(run["models"][f"{spec}:{s}"])
    return out


def stats(values):
    values = np.asarray(values, dtype=float)
    std = float(values.std(ddof=1)) if len(values) > 1 else 0.0
    return {"mean": float(values.mean()), "std": std, "n": len(values)}


def shares(beyond):
    return {"lateral": np.mean([beyond["left"], beyond["right"]]),
            "longitudinal": np.mean([beyond["front"], beyond["rear"]]),
            "all": np.mean([beyond[d] for d in DIRECTIONS])}


def main():
    combined = load(ADV / "results_val_b3_combined.json")
    summary = {"num_samples": combined["num_samples"], "models": {}}
    for key, label, spec, pattern in LADDER:
        runs = entries(spec, pattern, combined)
        row = {"label": label, "seeds": len(runs)}
        for state, getter in (("clean", lambda e: e["clean_beyond_half_lane"]),
                              ("random", lambda e: e["random_control"]["beyond_half_lane"]),
                              ("attacked", lambda e: e["attacked_beyond_half_lane"])):
            per_run = [shares(getter(e)) for e in runs]
            row[f"share_{state}"] = {g: stats([p[g] for p in per_run]) for g in ("lateral", "longitudinal", "all")}
        for metric in ("ade", "fde"):
            row[metric] = {"clean": stats([e["clean"][metric] for e in runs]),
                           "random": stats([e["random_control"]["metrics"][metric] for e in runs]),
                           "attacked": stats([e["attacked"][metric] for e in runs]),
                           "ratio": stats([e["attacked"][metric] / e["clean"][metric] for e in runs])}
        row["perturbation_m"] = stats([np.mean([e["perturbation"][o]["mean_point_m"] for o in e["perturbation"]])
                                       for e in runs])
        summary["models"][key] = row

    # Do the six candidates stay diverse when attacked? (final model, mode re-runs)
    modes = []
    for s in SEEDS:
        run = load(MODES / f"results_val_b3_multimodal_map_gru_seed{s}.json")
        if run["sample_indices"] != combined["sample_indices"]:
            raise ValueError("mode re-run covers different samples")
        e = run["models"][f"multimodal_map_gru:{s}"]
        if abs(e["attacked"]["ade"] - combined["models"][f"multimodal_map_gru:{s}"]["attacked"]["ade"]) > 1e-6:
            raise ValueError("mode re-run does not reproduce the combined attack")
        modes.append(e)
    summary["modes_final"] = {
        "clean": {k: stats([m["modes"]["clean"][k] for m in modes]) for k in ("spread", "hits", "miss")},
        "random": {k: stats([m["random_control"]["modes"][k] for m in modes])
                   for k in ("spread", "spread_ratio", "hits", "miss", "coherence_recentred")},
        "attacked": {o: {k: stats([m["modes"]["attacked"][o][k] for m in modes])
                         for k in ("spread", "spread_ratio", "hits", "miss", "coherence_recentred")}
                     for o in modes[0]["modes"]["attacked"]},
    }
    OUT.write_text(json.dumps(summary, indent=2))

    print(f"{'model':32s} {'ADE clean→atk (×)':>24s} {'lat c/r/a':>22s} {'long c/r/a':>22s} {'all c/r/a':>22s} {'δ m':>5s}")
    for key, row in summary["models"].items():
        def fmt(g):
            return "/".join(f"{row[f'share_{s}'][g]['mean']:.1%}" for s in ("clean", "random", "attacked")) + \
                   f"±{row['share_attacked'][g]['std']:.1%}"
        ade = row["ade"]
        print(f"{row['label']:32s} {ade['clean']['mean']:.2f}→{ade['attacked']['mean']:.2f} (×{ade['ratio']['mean']:.2f}) "
              f"{fmt('lateral'):>26s} {fmt('longitudinal'):>26s} {fmt('all'):>26s} {row['perturbation_m']['mean']:.2f}")
    m = summary["modes_final"]
    print("modes clean", {k: round(v["mean"], 3) for k, v in m["clean"].items()})
    print("modes random", {k: round(v["mean"], 3) for k, v in m["random"].items()})
    for o, v in m["attacked"].items():
        print("modes", o, {k: round(x["mean"], 3) for k, x in v.items()})


if __name__ == "__main__":
    main()
