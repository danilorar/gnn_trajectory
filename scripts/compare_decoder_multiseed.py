"""Verify and compare matched 60-epoch MLP/GRU decoder runs; no training.

Requires saved per-seed turn-angle analyses. Re-evaluates validation and
full test metrics for every checkpoint before writing the final comparison.
Across-seed and within-bin standard deviations are population values.
"""

import argparse
import json
import pickle
import sys
from pathlib import Path

import numpy as np
import torch
from torch_geometric.loader import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.run_turn_angle_analysis import sha256
from scripts.train_gru_decoder import BASELINE_REFERENCE, verify_baseline_classes
from utils.graph_utils import sample_to_graph
from utils.gru_future_decoder import GraphNetworkAttentionMultimodalWithMapGRU
from utils.models import GraphNetworkAttentionMultimodalWithMap
from utils.training_utils import evaluate_multimodal_model, set_seed

SEEDS = (42, 123, 2026)
METRICS = ("minADE6", "minFDE6")
LABELS = ("0-15 deg", "15-30 deg", "30-60 deg", "60+ deg")
SAMPLING = "Standard uniform shuffle without replacement; unmodified validation/test"


def read_json(path):
    return json.loads(Path(path).read_text())


def paths(decoder, seed):
    if decoder == "mlp":
        checkpoint = PROJECT_ROOT / f"checkpoints_60ep/multimodal_map_seed{seed}_best.pth"
        analysis = PROJECT_ROOT / f"doc/results/turn_angle_analysis_seed{seed}_60ep.json"
    else:
        folder = PROJECT_ROOT / "checkpoints_gru_decoder"
        if seed != 2026:
            folder /= f"seed{seed}"
        checkpoint = folder / f"multimodal_map_gru_seed{seed}_best.pth"
        analysis = PROJECT_ROOT / f"doc/results/turn_angle_analysis_seed{seed}_gru_decoder.json"
    return checkpoint, analysis


def reduction(mlp, gru):
    """Positive means GRU improves; percentage is reduction of the MLP mean."""
    return {"absolute_improvement_m": float(mlp - gru),
            "percentage_improvement": float(100 * (mlp - gru) / mlp)}


def compare(values):
    result = {}
    for decoder, numbers in values.items():
        result[decoder] = {"mean": float(np.mean(numbers)),
                           "std": float(np.std(numbers, ddof=0)),
                           "per_seed": dict(zip(SEEDS, map(float, numbers)))}
    result.update(reduction(result["mlp"]["mean"], result["gru"]["mean"]))
    result["paired_seed_improvements_m"] = dict(zip(
        SEEDS, map(float, np.asarray(values["mlp"]) - np.asarray(values["gru"]))
    ))
    return result


def load_loaders():
    loaders = {}
    for split in ("val", "test"):
        path = PROJECT_ROOT / f"data/processed/nuscenes_{split}_prediction_samples_map.pkl"
        with path.open("rb") as handle:
            samples = pickle.load(handle)
        graphs = [sample_to_graph(sample) for sample in samples]
        loaders[split] = DataLoader(graphs, batch_size=64, shuffle=False)
        print(f"Verification: {split}, {len(graphs)} samples", flush=True)
    return loaders


def verify_run(decoder, seed, loaders, hashes, device):
    checkpoint, analysis_path = paths(decoder, seed)
    manifest = read_json(checkpoint.with_suffix(".json"))
    analysis = read_json(analysis_path)
    expected_model = "multimodal_map" if decoder == "mlp" else "multimodal_map_gru"
    for key, expected in {"seed": seed, "num_epochs": 60, "num_modes": 6,
                          "batch_size": 64, "lr": .001, "radius": 35,
                          "model": expected_model, "device": "cuda"}.items():
        if manifest.get(key) != expected:
            raise ValueError(f"{checkpoint}: {key} differs from {expected}")
    if manifest.get("smoke_test", False):
        raise ValueError(f"Smoke-test checkpoint: {checkpoint}")
    if not 1 <= manifest.get("best_epoch", 0) <= 60:
        raise ValueError(f"Invalid best epoch: {checkpoint}")
    history_path = checkpoint.parent / f"training_history_seed{seed}.json"
    if decoder == "gru" and seed == 2026:
        history_path = checkpoint.parent / "training_history.json"
    if history_path.exists():
        history = read_json(history_path)
        if (len(history["train_losses"]) != 60 or len(history["val_losses"]) != 60
                or int(np.argmin(history["val_losses"])) + 1 != manifest["best_epoch"]
                or not np.isclose(min(history["val_losses"]), manifest["best_val_loss"])):
            raise ValueError(f"Training history differs from manifest: {history_path}")
    elif not (decoder == "mlp" and seed == 2026):
        raise ValueError(f"Missing training history: {history_path}")
    recorded_hashes = manifest.get("dataset_sha256")
    legacy = decoder == "mlp" and seed == 2026
    if recorded_hashes is None and not legacy:
        raise ValueError(f"Missing dataset provenance: {checkpoint}")
    if recorded_hashes is not None and recorded_hashes != hashes:
        raise ValueError(f"Different training/evaluation datasets: {checkpoint}")
    recorded_sampling = manifest.get("sampling")
    if recorded_sampling is None and not legacy:
        raise ValueError(f"Missing sampling provenance: {checkpoint}")
    if recorded_sampling is not None and recorded_sampling != SAMPLING:
        raise ValueError(f"Unexpected sampling: {checkpoint}")
    source_hashes = manifest.get("protected_source_sha256", {})
    source_exceptions = []
    for name, digest in source_hashes.items():
        if sha256(PROJECT_ROOT / name) != digest:
            if name == "utils/models.py":
                # The local TargetHistoryMLP addition was intentionally not
                # committed. Check the actual map/backbone classes against
                # the immutable baseline instead of requiring that extra code.
                verify_baseline_classes()
                source_exceptions.append("utils/models.py whole-file hash differs; relevant classes match pinned baseline")
            else:
                raise ValueError(f"Model/graph/training source changed: {name}")
    if (analysis["checkpoint_sha256"] != sha256(checkpoint)
            or analysis["dataset_sha256"] != hashes["test"]
            or analysis["seed"] != seed
            or analysis["minimum_endpoint_displacement_m"] != 5
            or analysis["std_ddof"] != 0
            or analysis["sanity_check"]["manifest_num_epochs"] != 60
            or not analysis["sanity_check"]["manifest_matches"]
            or not analysis["sanity_check"]["shared_evaluator_matches"]
            or analysis["invalid_sample_indices"]):
        raise ValueError(f"Analysis provenance/protocol mismatch: {analysis_path}")
    set_seed(seed)
    cls = (GraphNetworkAttentionMultimodalWithMap if decoder == "mlp"
           else GraphNetworkAttentionMultimodalWithMapGRU)
    model = cls(num_modes=6).to(device)
    model.load_state_dict(torch.load(checkpoint, map_location=device, weights_only=True), strict=True)
    if analysis["model"] != type(model).__name__:
        raise ValueError(f"Analysis model type differs: {analysis_path}")
    actual = {}
    for split, field in (("val", "validation_metrics"), ("test", "test_metrics")):
        actual[split] = dict(zip(METRICS, map(float, evaluate_multimodal_model(model, loaders[split], device))))
        for name in METRICS:
            value = actual[split][name]
            if not np.isclose(value, manifest[field][name], atol=.001, rtol=.0001):
                raise ValueError(f"{checkpoint}: {split}/{name} does not reproduce manifest")
            if split == "test" and not np.isclose(value, analysis["full_test"][name]["mean"], atol=1e-5, rtol=1e-5):
                raise ValueError(f"{checkpoint}: saved turn analysis does not reproduce")
    if analysis["full_test"]["n"] != len(loaders["test"].dataset):
        raise ValueError("Test sample count differs")
    print(f"Verified {decoder}, seed {seed}: validation={actual['val']}; test={actual['test']}", flush=True)
    return {"manifest": manifest, "analysis": str(analysis_path.relative_to(PROJECT_ROOT)),
            "checkpoint_sha256": sha256(checkpoint), "evaluation_reproduced": actual,
            "training_dataset_hashes_recorded": recorded_hashes is not None,
            "sampling_recorded": recorded_sampling is not None,
            "training_history_verified": history_path.exists(),
            "source_hash_exceptions": source_exceptions,
            "turn_analysis": analysis}


def write_report(result, path):
    lines = ["# Final matched decoder comparison", "",
             "Seeds 42, 123, 2026; 60 epochs; standard unweighted sampling; best validation minADE6 checkpoint. All errors in metres. All standard deviations use population ddof=0, including variability across the three seeds.", "",
             "## Individual runs", "",
             "| Decoder | Seed | Best epoch | Val minADE6 | Val minFDE6 | Test minADE6 | Test minFDE6 |",
             "|---|---:|---:|---:|---:|---:|---:|"]
    for decoder in ("mlp", "gru"):
        for seed in SEEDS:
            run = result["runs"][decoder][seed]
            m, ev = run["manifest"], run["evaluation_reproduced"]
            lines.append(f"| {decoder.upper()} | {seed} | {m['best_epoch']} | "
                         f"{ev['val']['minADE6']:.6f} | {ev['val']['minFDE6']:.6f} | "
                         f"{ev['test']['minADE6']:.6f} | {ev['test']['minFDE6']:.6f} |")
    lines += ["", "## Overall full test (7,242 samples)", "",
              "Improvement = MLP minus GRU; positive is better. Percentage = 100 × (MLP mean − GRU mean) / MLP mean.", "",
              "| Metric | MLP mean ± std | GRU mean ± std | Improvement (m) | Improvement (%) |",
              "|---|---:|---:|---:|---:|"]
    for name, stat in result["overall"].items():
        lines.append(f"| {name} | {stat['mlp']['mean']:.6f} ± {stat['mlp']['std']:.6f} | "
                     f"{stat['gru']['mean']:.6f} ± {stat['gru']['std']:.6f} | "
                     f"{stat['absolute_improvement_m']:+.6f} | {stat['percentage_improvement']:+.2f}% |")
    lines += ["", "## Turn bins", "",
              "Displacement ≥5 m; 7,024 retained samples, 218 excluded. Bins [0,15), [15,30), [30,60), [60,180] degrees. ADE and FDE independently minimize across six modes. Exact per-seed n/mean/median/std and manifests are included in the JSON.", "",
              "| Angle | n | Metric | MLP mean ± std across seeds | GRU mean ± std across seeds | Improvement (m) | Improvement (%) |",
              "|---|---:|---|---:|---:|---:|---:|"]
    for label, row in result["turn_bins"].items():
        for name in METRICS:
            stat = row[name]
            lines.append(f"| {label} | {row['n']} | {name} | "
                         f"{stat['mlp']['mean']:.6f} ± {stat['mlp']['std']:.6f} | "
                         f"{stat['gru']['mean']:.6f} ± {stat['gru']['std']:.6f} | "
                         f"{stat['absolute_improvement_m']:+.6f} | {stat['percentage_improvement']:+.2f}% |")
    lines += ["", "## Consistency by matched seed", "",
              "| Population | Metric | Seed 42 improvement (m) | Seed 123 improvement (m) | Seed 2026 improvement (m) |",
              "|---|---|---:|---:|---:|"]
    for label, row in [("Full test", result["overall"])] + list(result["turn_bins"].items()):
        for name in METRICS:
            diffs = row[name]["paired_seed_improvements_m"]
            lines.append(f"| {label} | {name} | " + " | ".join(f"{diffs[s]:+.6f}" for s in SEEDS) + " |")
    lines += ["", "## Interpretation", ""] + [f"- {note}" for note in result["interpretation"]]
    lines += ["", "## Integrity and interpretation limits", ""] + [f"- {note}" for note in result["integrity_notes"]]
    lines += ["", "Endpoint angle is net direction from the origin to the final future point, not path curvature or total heading change. The sharpest bin has 87 examples; seed variability is not a scene-level confidence interval. The decoder changes recurrence, relative displacement prediction, mode conditioning and parameter count together. These tests do not isolate their separate effects. Test-set results have informed experiment development, so this is a development comparison, not an untouched external final benchmark.", "",
              "## Reproduction", "", "```bash",
              "python scripts/train_gru_decoder.py --decoder mlp --seed 42 --checkpoint-dir checkpoints_60ep",
              "python scripts/train_gru_decoder.py --decoder mlp --seed 123 --checkpoint-dir checkpoints_60ep",
              "# For each new MLP checkpoint, run run_turn_angle_analysis.py with --seed, --checkpoint,",
              "# and --output doc/results/turn_angle_analysis_seed{seed}_60ep.json.",
              "python scripts/compare_decoder_multiseed.py --plot", "```", "",
              "Training commands refuse to overwrite existing checkpoints. Checkpoints and generated datasets remain local; lightweight manifests are embedded in the comparison JSON."]
    path.write_text("\n".join(lines) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--output", type=Path, default=PROJECT_ROOT / "doc/results/final_decoder_multiseed_comparison.json")
    parser.add_argument("--protected-snapshot", type=Path)
    parser.add_argument("--plot", action="store_true")
    args = parser.parse_args()
    verify_baseline_classes()
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    hashes = {split: sha256(PROJECT_ROOT / f"data/processed/nuscenes_{split}_prediction_samples_map.pkl")
              for split in ("train", "val", "test")}
    loaders = load_loaders()
    runs = {decoder: {seed: verify_run(decoder, seed, loaders, hashes, torch.device(args.device))
                      for seed in SEEDS} for decoder in ("mlp", "gru")}
    first = runs["mlp"][42]["turn_analysis"]
    for decoder in runs.values():
        for run in decoder.values():
            analysis = run["turn_analysis"]
            for key in ("angle_definition", "minimum_endpoint_displacement_m", "input_samples",
                        "excluded_displacement_below_5m", "invalid_sample_indices"):
                if analysis[key] != first[key]:
                    raise ValueError(f"Analysis population/protocol differs: {key}")
            for label in LABELS:
                if analysis["turn_bins"][label]["n"] != first["turn_bins"][label]["n"]:
                    raise ValueError("Turn-bin sample counts differ")
    overall = {name: compare({d: [runs[d][s]["evaluation_reproduced"]["test"][name] for s in SEEDS]
                              for d in runs}) for name in METRICS}
    bins = {label: {"n": first["turn_bins"][label]["n"], **{
        name: compare({d: [runs[d][s]["turn_analysis"]["turn_bins"][label][name]["mean"] for s in SEEDS]
                       for d in runs}) for name in METRICS}} for label in LABELS}
    interpretation = []
    for label, row in [("Overall full test", overall)] + list(bins.items()):
        details = []
        for name in METRICS:
            stat = row[name]
            wins = sum(delta > 0 for delta in stat["paired_seed_improvements_m"].values())
            details.append(f"{name}: {stat['percentage_improvement']:+.2f}% mean improvement; "
                           f"GRU improves in {wins}/3 matched seeds")
        interpretation.append(f"{label}: " + "; ".join(details) + ".")
    robust_overall = all(min(overall[name]["paired_seed_improvements_m"].values()) > 0 for name in METRICS)
    sharp_improves = all(bins[label][name]["absolute_improvement_m"] > 0
                         for label in LABELS[2:] for name in METRICS)
    straight_preserved = all(bins[LABELS[0]][name]["percentage_improvement"] >= -2 for name in METRICS)
    interpretation.append(
        "The matched results support selecting the GRU as the final model for this project: "
        "overall metrics improve in every seed, sharp-turn bin means improve, and mean straight-trajectory error stays close or improves. "
        "This is repeated descriptive evidence, not a statistical significance or causal claim."
        if robust_overall and sharp_improves and straight_preserved else
        "The results do not satisfy all of the planned consistency criteria; review the seed-level tradeoffs before choosing the final decoder."
    )
    interpretation.append("Sharp-turn degradation remains present; these improvements do not resolve the difficulty.")
    protected = read_json(args.protected_snapshot) if args.protected_snapshot else {}
    for path, digest in protected.items():
        if sha256(PROJECT_ROOT / path) != digest:
            raise ValueError(f"Pre-existing artifact changed: {path}")
    result = {"seeds": SEEDS, "num_epochs": 60, "std_ddof": 0,
              "baseline_code_reference": BASELINE_REFERENCE,
              "improvement_definition": "MLP minus GRU; positive means GRU improves",
              "dataset_sha256": hashes, "runs": runs, "overall": overall, "turn_bins": bins,
              "source_sha256": {name: sha256(PROJECT_ROOT / name) for name in (
                  "scripts/train_gru_decoder.py", "scripts/run_turn_angle_analysis.py",
                  "scripts/compare_decoder_multiseed.py", "utils/gru_future_decoder.py")},
              "interpretation": interpretation,
              "protected_existing_files_verified": len(protected),
              "integrity_notes": [
                  "All six checkpoints strictly load into the expected original MLP or GRU class; all manifests specify 60 epochs, six modes, radius 35 m, batch size 64 and Adam learning rate 0.001.",
                  "All six validation and full-test evaluations reproduce their manifests. Saved turn-bin evaluations match the same checkpoints and test dataset hashes and passed independent/shared evaluator checks.",
                  "New MLP runs and all GRU runs record identical train/validation/test hashes, unchanged model/graph/training utility hashes and uniform sampling without replacement. These settings were checked at runtime for the new MLP runs.",
                  "The relevant map-model and map-encoder classes are checked against pinned baseline commit " + BASELINE_REFERENCE + ". On a clean checkout, removing the unrelated uncommitted TargetHistoryMLP addition may change the whole utils/models.py hash; that exception is reported and accepted only when the relevant classes still match exactly. Graph/training utility hashes must still match.",
                  "Historical MLP seed-2026 manifest does not record training/validation dataset hashes or sampler details. Its full-test result and prior training-set evaluation match the current files; original source/default training procedure is consistent with unweighted sampling, but missing historical provenance cannot be retroactively verified.",
                  f"{len(protected)} pre-existing files verified unchanged, including existing baseline/GRU checkpoints, datasets, historical results and unrelated user edits." if protected else "No pre-existing-file snapshot supplied; use checkpoint/dataset hashes for provenance.",
                  "Model architecture, map encoder, graph construction, preprocessing, splits, loss and evaluator were not modified. No weighted sampling is used in this matched comparison.",
              ]}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    write_report(result, args.output.with_suffix(".md"))
    print(json.dumps({"overall": overall, "turn_bins": bins}, indent=2), flush=True)
    if args.plot:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(1, 2, figsize=(11, 4))
        x = np.arange(len(LABELS))
        for ax, name in zip(axes, METRICS):
            for offset, decoder, title in ((-.18, "mlp", "MLP"), (.18, "gru", "GRU")):
                ax.bar(x + offset, [bins[label][name][decoder]["mean"] for label in LABELS],
                       yerr=[bins[label][name][decoder]["std"] for label in LABELS],
                       width=.36, capsize=4, label=title)
            ax.set_xticks(x, LABELS)
            ax.set(ylabel=f"Mean {name} (m)", xlabel="Absolute endpoint angle (displacement ≥5 m)")
            ax.legend()
            ax.grid(axis="y", alpha=.2)
        fig.suptitle("Matched 60-epoch comparison; three seeds, population std")
        fig.tight_layout()
        figure = PROJECT_ROOT / "doc/figures/final_decoder_multiseed_comparison.png"
        figure.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(figure, dpi=160)
        plt.close(fig)
    print(f"Saved {args.output} and {args.output.with_suffix('.md')}")


if __name__ == "__main__":
    main()
