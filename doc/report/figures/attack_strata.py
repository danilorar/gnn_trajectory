"""When does the attack succeed? Final model's attack outcomes split by motion type and by speed.

Run from the project root:
    python doc/report/figures/attack_strata.py

Replays the PGD perturbations that scripts/adversarial_attack.py saved for the three
multimodal_map_gru seeds (data/processed/adversarial_modes), so no attack is rerun. Checks
that the replayed means match the saved results, then writes attack_strata.json.

Groups (500 validation samples, the same for every seed):
- motion, from the ground-truth future: slow (endpoint < 5 m from the current position),
  else straight (endpoint angle < 15 degrees) or turning (>= 15 degrees), as in
  scripts/run_turn_angle_analysis.py
- speed, from the clean history: |p_0 - p_-1| / 0.5 s
"""

import json
import sys
from pathlib import Path

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.adversarial_attack import (
    CHECKPOINT_DIRS, DIRECTIONS, HALF_LANE_WIDTH, OBJECTIVES, build_inputs, checkpoint_path,
    dataset_path, load_model, load_samples, motion_directions, perturbed_forward, sample_metrics,
)
from utils.experiment_utils import sha256

SEEDS = (42, 123, 2026)
ATTACK_DIR = PROJECT_ROOT / "data/processed/adversarial_modes"
OUT = Path(__file__).resolve().parent / "attack_strata.json"
SPEED_EDGES = (2.0, 8.0)  # m/s


def main():
    samples = load_samples("val", use_map=True)
    runs, indices = [], None
    for seed in SEEDS:
        spec = f"multimodal_map_gru:{seed}"
        results = json.loads((ATTACK_DIR / f"results_val_b3_multimodal_map_gru_seed{seed}.json").read_text())
        entry = results["models"][spec]
        deltas = np.load(ATTACK_DIR / f"deltas_val_b3_multimodal_map_gru_seed{seed}.npz")
        assert deltas["sample_indices"].tolist() == results["sample_indices"]
        assert indices is None or indices == results["sample_indices"], "seeds attacked different samples"
        indices = results["sample_indices"]
        assert sha256(checkpoint_path(spec, CHECKPOINT_DIRS)) == entry["checkpoint_sha256"]
        assert sha256(dataset_path("val", use_map=True)) == results["dataset_sha256"]

        batch, past, future = build_inputs([samples[i] for i in indices])
        model = load_model(spec, CHECKPOINT_DIRS)
        forward, left = motion_directions(future)
        with torch.no_grad():
            clean = sample_metrics(perturbed_forward(model, batch, torch.zeros_like(past)), future, forward, left)
            attacked = {o: sample_metrics(perturbed_forward(model, batch, torch.tensor(deltas[o])), future, forward, left)
                        for o in OBJECTIVES}
        # Replaying the saved perturbations must reproduce the saved results
        for o in OBJECTIVES:
            assert abs(float(attacked[o][o].mean()) - entry["attacked"][o]) < 1e-4, (seed, o)
        runs.append({"clean_ade": clean["ade"].numpy(), "attacked_ade": attacked["ade"]["ade"].numpy(),
                     **{f"hit_{d}": (attacked[d][d] > HALF_LANE_WIDTH).numpy() for d in DIRECTIONS},
                     **{f"clean_hit_{d}": (clean[d] > HALF_LANE_WIDTH).numpy() for d in DIRECTIONS}})

    # Groups depend only on the data, so they are the same for every seed
    end = future[:, -1].numpy()
    distance = np.linalg.norm(end, axis=1)
    angle = np.degrees(np.abs(np.arctan2(end[:, 1], end[:, 0])))
    speed = np.linalg.norm((past[:, -1] - past[:, -2]).numpy(), axis=1) / 0.5
    groups = {
        "motion": {"slow (< 5 m in 6 s)": distance < 5,
                   "straight (< 15 deg)": (distance >= 5) & (angle < 15),
                   "turning (>= 15 deg)": (distance >= 5) & (angle >= 15)},
        "speed": {f"< {SPEED_EDGES[0]:g} m/s": speed < SPEED_EDGES[0],
                  f"{SPEED_EDGES[0]:g}-{SPEED_EDGES[1]:g} m/s": (speed >= SPEED_EDGES[0]) & (speed < SPEED_EDGES[1]),
                  f">= {SPEED_EDGES[1]:g} m/s": speed >= SPEED_EDGES[1]},
    }

    def summary(mask):
        def over_seeds(f):
            values = [f(r) for r in runs]
            return {"mean": float(np.mean(values)), "std": float(np.std(values))}
        return {"n": int(mask.sum()),
                "clean_ade": over_seeds(lambda r: r["clean_ade"][mask].mean()),
                "attacked_ade": over_seeds(lambda r: r["attacked_ade"][mask].mean()),
                "lateral_clean": over_seeds(lambda r: np.mean([r[f"clean_hit_{d}"][mask].mean() for d in ("left", "right")])),
                "lateral": over_seeds(lambda r: np.mean([r[f"hit_{d}"][mask].mean() for d in ("left", "right")])),
                "longitudinal_clean": over_seeds(lambda r: np.mean([r[f"clean_hit_{d}"][mask].mean() for d in ("front", "rear")])),
                "longitudinal": over_seeds(lambda r: np.mean([r[f"hit_{d}"][mask].mean() for d in ("front", "rear")]))}

    out = {"model": "multimodal_map_gru", "seeds": SEEDS, "num_samples": len(indices),
           "all": summary(np.ones(len(indices), dtype=bool)),
           **{kind: {name: summary(mask) for name, mask in masks.items()} for kind, masks in groups.items()}}
    OUT.write_text(json.dumps(out, indent=2) + "\n")

    print(f"{'group':24s} {'n':>4s} {'clean':>6s} {'att.':>6s} {'lat.':>12s} {'long.':>12s}")
    for kind in ("all", "motion", "speed"):
        rows = {"all": out["all"]} if kind == "all" else out[kind]
        for name, s in rows.items():
            print(f"{name:24s} {s['n']:4d} {s['clean_ade']['mean']:6.2f} {s['attacked_ade']['mean']:6.2f} "
                  f"{100 * s['lateral_clean']['mean']:4.0f}->{100 * s['lateral']['mean']:3.0f}% "
                  f"{100 * s['longitudinal_clean']['mean']:4.0f}->{100 * s['longitudinal']['mean']:3.0f}%")


if __name__ == "__main__":
    main()
