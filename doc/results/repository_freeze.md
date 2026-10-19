# Final repository freeze audit

Recovery point: **`ca94027dd17b5fb0d4a1bd0d2ceea3ca07983a6c`**. This commit was recorded before editing and was not rewritten. Cleanup is a separate commit on `experiment/gru-future-decoder`. Local uncommitted source/results/notebooks/figures were backed up to `/tmp/gnn-freeze-ca94027d` before retirement (temporary local backup, not a portable archive). Tracked historical experiments remain recoverable from the recovery commit.

## Scientific freeze

The canonical `final_decoder_multiseed_comparison.md`, `.json` and figure remain byte-identical. No training epochs, optimizer updates, dataset generation or data edits were performed. All checkpoint weights and existing checkpoint manifests remain unchanged. Graph conversion, all tested model constructor/forward bodies, training loop and loss/evaluation math were compared as executable ASTs with the recovery commit, ignoring docstrings/comments and allowing the GRU class relocation. Exact old/new GRU initialization and forward output parity also passed.

The selected model is `GraphNetworkAttentionMultimodalWithMapGRU`: unchanged 15-D node inputs, 64-D agent encoder, two GATv2 layers, existing map encoder, fused 96-D context, six 16-D mode embeddings, 64-D one-layer GRU with predicted-displacement feedback, twelve displacements and cumulative positions. It is now defined once in `utils/models.py`.

The previously local `TargetHistoryMLP` is retained unchanged in behavior: target history/category features only, 15→64 encoder and 64→128→24 decoder. Its existing seed-42 checkpoint strictly loads and reproduces validation ADE/FDE 3.851514/9.350993 and test ADE/FDE 3.818514/9.045736. The final interaction analysis uses this checkpoint and the deterministic GATv2 seed-42 checkpoint.

## Retained script classification

| Script | Class | Reason |
|---|---|---|
| `build_processed_dataset.py` | A: final pipeline | Unchanged preprocessing/splits/radii; rebuilds map samples from raw assets. |
| `train_model.py` | A: final pipeline | Single training/evaluation interface for target MLP, attention, multimodal, map MLP and map GRU. Uniform sampling, existing losses/evaluators and validation checkpoint selection. Evaluation-only mode verifies manifests and never trains/saves weights. |
| `run_interaction_analysis.py` | B: final analysis | Actual deterministic target-MLP vs GAT comparison by count/distance; saved result records checkpoint provenance. |
| `run_turn_angle_analysis.py` | B: final analysis | Saved-checkpoint analysis with independent mode minima and manifest sanity check before bins. |
| `compare_decoder_multiseed.py` | B: final analysis | Recomputes all six validation/test metrics and turn-bin mean/median/std against the embedded canonical records; no dependency on deleted seed JSON files. |
| `run_radius_sweep.py` | B: retained historical reproduction | Reproduces the reported radius-selection study supporting fixed 35 m; no new sweep is performed or required. |

Historical train-vs-test, map-vs-no-map and weighted-sampling executables are class C; their numerical outcomes are consolidated in `diagnostic_summary.md`. The experiment-specific training runner and separate GRU module are class D after integration. Their important safeguards moved to the normal CLI and `utils/experiment_utils.py`; original controlled settings are reproduced by `train_model.py --model multimodal_map[_gru] --epochs 60 --seed ... --batch-size 64 --lr 0.001` with a separate output directory.

Canonical documents retain their original historical commands without alteration. The README gives the replacement commands. `utils/experiment_utils.py` pins semantic verification to the recovery commit, allowing comment/docstring cleanup and class relocation without relying on obsolete whole-file source hashes. Original manifests are not retroactively edited.

## Results and figures

Kept: canonical comparison JSON/Markdown (including all six lightweight manifests and detailed per-seed statistics), the one canonical comparison figure, `diagnostic_summary.md`, `interaction_analysis_seed42.json`, and this audit. Kept the earlier radius-selection report and its data/processed summaries outside doc/results.

Intermediate turn-bin JSONs, one-seed/unmatched decoder summaries and diagnostic plots were consolidated or superseded. The invalid one-epoch seed-42 map turn analysis/figure and `qualitative_candidates.png` were removed. The qualitative inspection script selected that overwritten seed-42 map checkpoint, so its figure could not be treated as valid final-model evidence; the script was also retired. Existing camera/processed-sample inspection tools remain. No invalid qualitative figure is presented as final evidence.

## Notebook audit

All five notebooks are retained for unique exploration/teaching provenance, not as final pipeline entry points. Every generated output and execution count was cleared; no notebook was executed for training. Early graph choices, metric conventions and narrative performance claims are explicitly historical. No source formula/model rewrite was done; specific obsolete imports and misleading labels/comments were repaired.

| Notebook | Why kept | Technical/staleness finding | Size before → after |
|---|---|---|---:|
| `01_nuscenes_investigate.ipynb` | Raw scene/annotation hierarchy, target tracks and local-frame prototypes. | Requires raw nuScenes; early extraction prototypes are historical, not the frozen builder. | 87,569 → 22,369 bytes |
| `02_dataset_analysis.ipynb` | Unique density, category and target-motion exploration. | Earlier narrative counts may not describe current samples; target/ego wording corrected. | 168,442 → 17,168 bytes |
| `03_baselines.ipynb` | Unique constant-velocity, GINE baseline, graph/edge walkthrough and ablation history. | Earlier radius-pruned graphs and inline training/model code remain labelled historical. Obsolete helper imports repaired; history is 2 s, not 6 s. | 2,795,032 → 61,073 bytes |
| `04_attention.ipynb` | GATv2 teaching rationale and deterministic neighbor-removal ablations. | Training/evaluation scripts supersede notebook execution for final runs; target terminology corrected. | 104,386 → 16,808 bytes |
| `05_multimodal.ipynb` | Best-of-six loss derivation, no-map decoder explanation and archived multi-seed experiment. | Historical FDE-at-ADE-best evaluator relabelled legacy; final independent minFDE6 is in training_utils. Not a map/GRU benchmark. | 78,018 → 21,265 bytes |

The large `03_baselines.ipynb` was mostly embedded generated output; source cells for constant velocity and GINE remain. Notebook 05 historically reports FDE at the ADE-best mode; it is not relabelled as independent minFDE6. Its numerical convention remains unchanged, with corrected names/labels. All code cells pass Python syntax checks, but full notebook execution would require data/images and may train historical models, so it was intentionally not run.

## Final structure

```text
scripts/
  build_processed_dataset.py
  train_model.py
  run_interaction_analysis.py
  run_turn_angle_analysis.py
  compare_decoder_multiseed.py
  run_radius_sweep.py
utils/
  models.py
  graph_utils.py
  training_utils.py
  analysis_utils.py
  experiment_utils.py
notebooks/  01–05, exploratory source retained, outputs cleared
tools/      view_sample_image.py, project_sample_to_camera.py
doc/results/
  final_decoder_multiseed_comparison.md
  final_decoder_multiseed_comparison.json
  diagnostic_summary.md
  interaction_analysis_seed42.json
  repository_freeze.md
doc/figures/
  final_decoder_multiseed_comparison.png
```

Legacy deterministic density/ablation helpers remain because notebooks and the radius reproduction import them. The legacy radius-distance ablation retains its historical normal-minus-ablated sign; the final comparison script clearly uses MLP-minus-GAT, positive for GAT wins. The previously local count-ablation helper behavior was preserved and its stale comments corrected.

## Validation

- `compileall` for scripts/utils/tools, import checks, all notebook code-cell syntax checks.
- Graph batching with no neighbors, no lanes, and empty lane polylines; fully connected directed edge counts and lane-ID offsets.
- Exact old/new GRU state initialization and forward parity; strict checkpoint load; `[B,6,12,2]` output; finite best-of-six loss/backward gradients.
- Common training CLI routing verified with a stub that stops before training; no training epochs or optimizer steps. Correct model, criterion, seeded uniform sampler, optimizer settings and epoch budget were checked.
- Normal evaluation CLI reproduced final seed-2026 GRU validation/test metrics and the actual target-MLP seed-42 metrics.
- All six final MLP/GRU checkpoints reproduced validation/test manifests and every saved turn-bin mean, median and population std. Aggregate matched means/std were unchanged.
- Retained turn-angle CLI reproduced seed-2026 results separately; final interaction analysis reproduces each deterministic checkpoint’s full-test ADE.
- README command help/import checks where safe; raw-data rebuild was inspected/imported, not run because it writes datasets. Overwrite and seed/model-mismatch guards reject unsafe invocations.
- All 106 pre-existing checkpoint/data/canonical-file SHA256 checks passed; no weights, datasets or canonical results changed.

Verified runtime: Python 3.12.3; torch 2.14.1, torch_geometric 2.8.0.post1, numpy 1.26.4, nuscenes-devkit 1.2.0.

## Remaining limits before merge

- Repository Conda templates predate this verified runtime; exact numerical reproduction should use the recorded versions and saved weights, and seeded CUDA scatter may vary across devices. No dependency migration was performed during this freeze.
- No new scientific discrepancy was found. Historical MLP seed-2026 training-data/sampler metadata remains incomplete, as already disclosed.
- Local weights and generated map datasets are excluded from Git. Reproduction needs those weights/assets; machine-readable canonical metadata remains tracked.
- Only three seeds and 87 sharpest-turn samples; endpoint angle is not curvature, and the test set informed development.
- The canonical report intentionally retains pre-cleanup command/source-hash history. Use README commands and the pinned executable-behavior check for the frozen code.
- No merge to main is performed; only the updated experiment branch is pushed.

## Retired files

The list includes tracked deletions and local generated artifacts removed after backup:

- `doc/figures/gru_decoder_multiseed_comparison.png`
- `doc/figures/gru_decoder_vs_baseline_seed2026.png`
- `doc/figures/qualitative_candidates.png`
- `doc/figures/train_turn_angle_distribution.png`
- `doc/figures/train_vs_test_turn_angle_seed2026_60ep.png`
- `doc/figures/turn_angle_analysis_seed123.png`
- `doc/figures/turn_angle_analysis_seed2026.png`
- `doc/figures/turn_angle_analysis_seed2026_60ep.png`
- `doc/figures/turn_angle_analysis_seed42.png`
- `doc/figures/turn_angle_map_comparison_seed123.png`
- `doc/figures/turn_angle_map_comparison_seed2026.png`
- `doc/figures/turn_aware_vs_baseline_seed2026.png`
- `doc/results/decoder_experiment_commit_scope.md`
- `doc/results/gru_decoder_comparison_seed2026.json`
- `doc/results/gru_decoder_experiment_seed2026.md`
- `doc/results/gru_decoder_multiseed_summary.json`
- `doc/results/gru_decoder_multiseed_summary.md`
- `doc/results/train_turn_angle_analysis_seed2026_60ep.json`
- `doc/results/train_turn_angle_distribution.json`
- `doc/results/train_vs_test_turn_angle_seed2026_60ep.md`
- `doc/results/turn_angle_30_vs_60_seed2026.json`
- `doc/results/turn_angle_analysis_seed123.json`
- `doc/results/turn_angle_analysis_seed123_60ep.json`
- `doc/results/turn_angle_analysis_seed123_gru_decoder.json`
- `doc/results/turn_angle_analysis_seed2026.json`
- `doc/results/turn_angle_analysis_seed2026_60ep.json`
- `doc/results/turn_angle_analysis_seed2026_gru_decoder.json`
- `doc/results/turn_angle_analysis_seed2026_turn_aware.json`
- `doc/results/turn_angle_analysis_seed42.json`
- `doc/results/turn_angle_analysis_seed42_60ep.json`
- `doc/results/turn_angle_analysis_seed42_gru_decoder.json`
- `doc/results/turn_angle_map_comparison_seed123.json`
- `doc/results/turn_angle_map_comparison_seed2026.json`
- `doc/results/turn_angle_map_diagnostics.md`
- `doc/results/turn_aware_training_comparison_seed2026.json`
- `doc/results/turn_aware_training_seed2026.md`
- `scripts/run_train_turn_angle_analysis.py`
- `scripts/run_turn_angle_map_comparison.py`
- `scripts/run_turn_aware_training.py`
- `scripts/train_gru_decoder.py`
- `tools/plot_qualitative_prediction.py`
- `utils/gru_future_decoder.py`

## Subsequent qualitative visualization

After the freeze, a user-requested plot for test sample 6706 was added using the validated 60-epoch GRU seed-2026 checkpoint (best epoch 53). The new `tools/plot_qualitative_prediction.py` verifies checkpoint and dataset hashes against the canonical comparison and saves a PNG plus per-mode metrics/coordinates and provenance JSON. The saved figure is `doc/figures/qualitative_gru_seed2026_sample6706.png`; minADE6 is 4.376181 m and independent minFDE6 is 12.432691 m. It shows all six modes and highlights the ADE-best mode. The turn is partially represented but its final position is still substantially wrong.

This is a valid final-model visualization, distinct from the retired tool/figure based on the overwritten seed-42 checkpoint. The retirement list above refers to that earlier implementation. The sample index was specified by the user, not selected by performance. The canonical aggregate artifacts, model behavior, weights and datasets remain unchanged. The README documents the current qualitative command; `--show` displays without saving files.
