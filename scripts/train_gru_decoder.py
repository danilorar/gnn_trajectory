"""Controlled 60-epoch GRU/MLP decoder comparison using baseline utilities.

The default decoder is GRU. Use --decoder mlp for the original map-model
baseline; only the model class and separate output directory change.
"""

import argparse
import ast
import json
import pickle
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import RandomSampler, SequentialSampler
from torch_geometric.loader import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[1]
BASELINE_REFERENCE = "631c7d4e1ad2fcda37d57570989b630541af874a"
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.run_turn_angle_analysis import sha256
from utils.graph_utils import sample_to_graph
from utils.gru_future_decoder import GraphNetworkAttentionMultimodalWithMapGRU
from utils.models import GraphNetworkAttentionMultimodalWithMap
from utils.training_utils import (
    best_of_k_ade_loss, evaluate_multimodal_model, save_run_manifest,
    set_seed, train_and_validate,
)


def verify_baseline_classes():
    """Ignore unrelated target-MLP additions; require unchanged map/backbone class code."""
    committed = subprocess.check_output(
        ["git", "show", f"{BASELINE_REFERENCE}:utils/models.py"], cwd=PROJECT_ROOT, text=True
    )
    current = (PROJECT_ROOT / "utils/models.py").read_text()
    def classes(source):
        return {node.name: ast.dump(node) for node in ast.parse(source).body
                if isinstance(node, ast.ClassDef)}
    a, b = classes(committed), classes(current)
    for name in ("MapEncoder", "GraphNetworkAttentionMultimodalWithMap"):
        if a[name] != b[name]:
            raise ValueError(f"Baseline class was changed: {name}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--decoder", choices=("gru", "mlp"), default="gru")
    parser.add_argument("--checkpoint-dir", type=Path)
    parser.add_argument("--smoke-test", action="store_true",
                        help="One epoch, three samples/split; requires an explicit temporary directory")
    args = parser.parse_args()
    if not 0 <= args.seed < 2**32:
        parser.error("--seed must be between 0 and 2**32 - 1")
    if args.smoke_test and args.checkpoint_dir is None:
        parser.error("Smoke test must use a separate --checkpoint-dir")
    if args.checkpoint_dir is None:
        args.checkpoint_dir = PROJECT_ROOT / ("checkpoints_gru_decoder" if args.decoder == "gru" else "checkpoints_60ep")
    for folder in ("checkpoints_gru_decoder", "checkpoints_60ep"):
        if args.smoke_test and args.checkpoint_dir.resolve().is_relative_to(PROJECT_ROOT / folder):
            parser.error("Smoke test must use a temporary directory outside experiment outputs")
    verify_baseline_classes()
    baseline_path = PROJECT_ROOT / "checkpoints_60ep/multimodal_map_seed2026_best.pth"
    baseline = json.loads(baseline_path.with_suffix(".json").read_text())
    if tuple(baseline[k] for k in ("seed", "num_epochs", "batch_size", "lr")) != (2026, 60, 64, .001):
        raise ValueError("Baseline settings differ from the planned experiment")
    model_name = "multimodal_map_gru" if args.decoder == "gru" else "multimodal_map"
    checkpoint = args.checkpoint_dir / f"{model_name}_seed{args.seed}_best.pth"
    history_path = args.checkpoint_dir / f"training_history_seed{args.seed}.json"
    if history_path.exists():
        raise FileExistsError(f"Refusing to overwrite existing history: {history_path}")
    if checkpoint.exists() or checkpoint.with_suffix(".json").exists():
        raise FileExistsError(f"Refusing to overwrite existing checkpoint: {checkpoint}")
    other_dir = PROJECT_ROOT / ("checkpoints_60ep" if args.decoder == "gru" else "checkpoints_gru_decoder")
    if args.checkpoint_dir.resolve().is_relative_to(other_dir.resolve()):
        raise ValueError("MLP and GRU experiment directories must remain separate")
    args.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    protected = {str(p): sha256(p) for p in (baseline_path, baseline_path.with_suffix(".json"))}
    source_hashes = {name: sha256(PROJECT_ROOT / name) for name in
                     ("utils/models.py", "utils/graph_utils.py", "utils/training_utils.py")}
    set_seed(args.seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if not args.smoke_test and str(device) != baseline["device"]:
        raise RuntimeError("Full experiment must use the baseline CUDA device")
    graphs, dataset_hashes = {}, {}
    for split in ("train", "val", "test"):
        path = PROJECT_ROOT / f"data/processed/nuscenes_{split}_prediction_samples_map.pkl"
        dataset_hashes[split] = sha256(path)
        with path.open("rb") as handle:
            samples = pickle.load(handle)
        if args.smoke_test:
            samples = samples[:3]
        print(f"Building {split} graphs: {len(samples)}", flush=True)
        graphs[split] = [sample_to_graph(s) for s in samples]
    del samples
    batch_size = 2 if args.smoke_test else 64
    epochs = 1 if args.smoke_test else 60
    train = DataLoader(graphs["train"], batch_size=batch_size, shuffle=True,
                       generator=torch.Generator().manual_seed(args.seed))
    val = DataLoader(graphs["val"], batch_size=batch_size, shuffle=False)
    test = DataLoader(graphs["test"], batch_size=batch_size, shuffle=False)
    assert isinstance(train.sampler, RandomSampler) and not train.sampler.replacement
    assert isinstance(val.sampler, SequentialSampler) and isinstance(test.sampler, SequentialSampler)
    # Save/restore RNG around the baseline parameter-count comparison so model
    # initialization remains identical to a normal run with the requested seed.
    model_class = (GraphNetworkAttentionMultimodalWithMapGRU if args.decoder == "gru"
                   else GraphNetworkAttentionMultimodalWithMap)
    model = model_class(num_modes=6).to(device)
    if args.decoder == "mlp" and (type(model) is not GraphNetworkAttentionMultimodalWithMap
                                  or not isinstance(model.decoder, torch.nn.Sequential)):
        raise RuntimeError("MLP baseline must use the original one-shot decoder model")
    state = torch.random.get_rng_state()
    baseline_model = GraphNetworkAttentionMultimodalWithMap(num_modes=6)
    torch.random.set_rng_state(state)
    parameter_counts = {"baseline_decoder": sum(p.numel() for p in baseline_model.decoder.parameters()),
                        f"{args.decoder}_decoder": sum(p.numel() for p in model.decoder.parameters()),
                        f"{args.decoder}_total": sum(p.numel() for p in model.parameters())}
    del baseline_model
    optimizer = torch.optim.Adam(model.parameters(), lr=.001)
    print(f"Training {type(model).__name__} ({args.decoder} decoder): "
          f"seed={args.seed}, epochs={epochs}, device={device}, "
          f"standard unweighted shuffle, batches/epoch={len(train)}; {parameter_counts}", flush=True)
    train_losses, val_losses = train_and_validate(
        model, train, val, optimizer, best_of_k_ade_loss, epochs, checkpoint, device
    )
    model.load_state_dict(torch.load(checkpoint, map_location=device, weights_only=True))
    validation = evaluate_multimodal_model(model, val, device)
    testing = evaluate_multimodal_model(model, test, device)
    for path, digest in protected.items():
        if sha256(path) != digest:
            raise RuntimeError(f"Baseline artifact changed: {path}")
    for path, digest in source_hashes.items():
        if sha256(PROJECT_ROOT / path) != digest:
            raise RuntimeError(f"Protected source changed: {path}")
    for split, digest in dataset_hashes.items():
        if sha256(PROJECT_ROOT / f"data/processed/nuscenes_{split}_prediction_samples_map.pkl") != digest:
            raise RuntimeError(f"Dataset changed: {split}")
    names = ("minADE6", "minFDE6")
    save_run_manifest(
        checkpoint, model=model_name, model_class=type(model).__name__,
        seed=args.seed, radius=35, num_modes=6,
        num_epochs=epochs, best_epoch=int(np.argmin(val_losses)) + 1,
        best_val_loss=min(val_losses), batch_size=batch_size, lr=.001, device=str(device),
        validation_metrics=dict(zip(names, validation)), test_metrics=dict(zip(names, testing)),
        decoder={"mode_embedding_dim": 16, "hidden_dim": 64, "layers": 1, "steps": 12,
                 "input": "previous predicted displacement, initially zero",
                 "h0": "tanh(linear(fused context concatenated with learned mode embedding))",
                 "output": "cumulative sum of predicted xy displacements", "teacher_forcing": False}
                if args.decoder == "gru" else
                {"type": "Original one-shot MLP", "context_dim": 96, "hidden_dim": 128,
                 "output": "6 x 12 x 2 absolute positions", "teacher_forcing": False},
        sampling="Standard uniform shuffle without replacement; unmodified validation/test",
        baseline_checkpoint_sha256=protected[str(baseline_path)], baseline_reference_seed=2026,
        dataset_sha256=dataset_hashes,
        protected_source_sha256=source_hashes, parameter_counts=parameter_counts,
        smoke_test=args.smoke_test,
    )
    history_path.write_text(
        json.dumps({"train_losses": train_losses, "val_losses": val_losses}, indent=2) + "\n"
    )
    print(f"Best epoch: {int(np.argmin(val_losses)) + 1}; validation: {validation}; test: {testing}", flush=True)
    print(f"Saved {checkpoint}; baseline sources/checkpoints/datasets unchanged.", flush=True)


if __name__ == "__main__":
    main()
