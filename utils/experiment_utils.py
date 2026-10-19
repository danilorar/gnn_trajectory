"""File provenance and behavior checks for reproducing the frozen experiment."""

import ast
import hashlib
import json
import subprocess
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT_COMMIT = "ca94027dd17b5fb0d4a1bd0d2ceea3ca07983a6c"


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_manifest(checkpoint):
    return json.loads(Path(checkpoint).with_suffix(".json").read_text())


def write_json(path, data):
    """Keep evaluations separate from existing checkpoints and final results."""
    path = Path(path)
    if path.exists():
        raise FileExistsError(f"Refusing to overwrite existing output: {path}")
    payload = json.dumps(data, indent=2, allow_nan=False)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(payload + "\n")


def verify_frozen_implementation():
    """Compare executable ASTs with the tested commit, ignoring docstrings.

    Classes may move and comments/formatting may change during cleanup.
    Constructor/forward logic, graph construction and training/metric math
    must remain identical. Original manifests retain their original hashes.
    """
    checks = {
        "utils/models.py": ("utils/models.py", (
            "GraphNetworkAttention", "GraphNetworkAttentionMultimodal",
            "MapEncoder", "GraphNetworkAttentionMultimodalWithMap")),
        "utils/gru_future_decoder.py": ("utils/models.py", (
            "GRUFutureDecoder", "GraphNetworkAttentionMultimodalWithMapGRU")),
        "utils/graph_utils.py": ("utils/graph_utils.py", (
            "TrajectoryData", "general_category", "category_one_hot", "sample_to_graph")),
        "utils/training_utils.py": ("utils/training_utils.py", (
            "set_seed", "ade_loss", "best_of_k_ade_loss", "train_and_validate",
            "evaluate_model", "evaluate_multimodal_model")),
        "scripts/run_turn_angle_analysis.py": ("scripts/run_turn_angle_analysis.py", (
            "endpoint_geometry", "per_sample_metrics", "summarize")),
    }

    class WithoutDocstrings(ast.NodeTransformer):
        def clean(self, node):
            self.generic_visit(node)
            if (node.body and isinstance(node.body[0], ast.Expr)
                    and isinstance(node.body[0].value, ast.Constant)
                    and isinstance(node.body[0].value.value, str)):
                node.body.pop(0)
            return node
        visit_ClassDef = clean
        visit_FunctionDef = clean

    def definitions(source):
        tree = WithoutDocstrings().visit(ast.parse(source))
        return {node.name: ast.dump(node) for node in tree.body
                if isinstance(node, (ast.ClassDef, ast.FunctionDef))}

    for old_path, (new_path, names) in checks.items():
        original = subprocess.check_output(
            ["git", "show", f"{EXPERIMENT_COMMIT}:{old_path}"],
            cwd=PROJECT_ROOT, text=True,
        )
        old, new = definitions(original), definitions((PROJECT_ROOT / new_path).read_text())
        for name in names:
            if old[name] != new[name]:
                raise ValueError(f"Frozen executable logic changed: {name}")
    return {"reference_commit": EXPERIMENT_COMMIT, "executable_ast_matches": True}
