# Decoder experiment commit scope

Branch: `experiment/gru-future-decoder`. Only the reviewed files below are included. No push, cleanup, or deletion is performed. The final matched comparison covers six standard-unweighted 60-epoch runs; historical turn-aware and 30-epoch records are supporting analyses, not members of the final matched comparison.

## Included files

- `doc/figures/final_decoder_multiseed_comparison.png`
- `doc/results/decoder_experiment_commit_scope.md`
- `doc/results/final_decoder_multiseed_comparison.json`
- `doc/results/final_decoder_multiseed_comparison.md`
- `doc/results/gru_decoder_comparison_seed2026.json`
- `doc/results/gru_decoder_experiment_seed2026.md`
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
- `doc/results/turn_angle_analysis_seed42_60ep.json`
- `doc/results/turn_angle_analysis_seed42_gru_decoder.json`
- `doc/results/turn_angle_map_comparison_seed123.json`
- `doc/results/turn_angle_map_comparison_seed2026.json`
- `doc/results/turn_angle_map_diagnostics.md`
- `doc/results/turn_aware_training_comparison_seed2026.json`
- `doc/results/turn_aware_training_seed2026.md`
- `scripts/compare_decoder_multiseed.py`
- `scripts/run_train_turn_angle_analysis.py`
- `scripts/run_turn_angle_analysis.py`
- `scripts/run_turn_angle_map_comparison.py`
- `scripts/run_turn_aware_training.py`
- `scripts/train_gru_decoder.py`
- `utils/gru_future_decoder.py`

## Intentionally left uncommitted

| File | Reason |
|---|---|
| `checkpoints_60ep/multimodal_map_seed123_best.json` | Local checkpoint weight/log/history/manifest; final six-run lightweight manifests are embedded in final_decoder_multiseed_comparison.json. |
| `checkpoints_60ep/multimodal_map_seed123_best.pth` | Local checkpoint weight/log/history/manifest; final six-run lightweight manifests are embedded in final_decoder_multiseed_comparison.json. |
| `checkpoints_60ep/multimodal_map_seed2026_best.json` | Local checkpoint weight/log/history/manifest; final six-run lightweight manifests are embedded in final_decoder_multiseed_comparison.json. |
| `checkpoints_60ep/multimodal_map_seed2026_best.pth` | Local checkpoint weight/log/history/manifest; final six-run lightweight manifests are embedded in final_decoder_multiseed_comparison.json. |
| `checkpoints_60ep/multimodal_map_seed42_best.json` | Local checkpoint weight/log/history/manifest; final six-run lightweight manifests are embedded in final_decoder_multiseed_comparison.json. |
| `checkpoints_60ep/multimodal_map_seed42_best.pth` | Local checkpoint weight/log/history/manifest; final six-run lightweight manifests are embedded in final_decoder_multiseed_comparison.json. |
| `checkpoints_60ep/training_history_seed123.json` | Local checkpoint weight/log/history/manifest; final six-run lightweight manifests are embedded in final_decoder_multiseed_comparison.json. |
| `checkpoints_60ep/training_history_seed42.json` | Local checkpoint weight/log/history/manifest; final six-run lightweight manifests are embedded in final_decoder_multiseed_comparison.json. |
| `checkpoints_60ep/training_seed123.log` | Local checkpoint weight/log/history/manifest; final six-run lightweight manifests are embedded in final_decoder_multiseed_comparison.json. |
| `checkpoints_60ep/training_seed42.log` | Local checkpoint weight/log/history/manifest; final six-run lightweight manifests are embedded in final_decoder_multiseed_comparison.json. |
| `checkpoints_gru_decoder/multimodal_map_gru_seed2026_best.json` | Local checkpoint weight/log/history/manifest; final six-run lightweight manifests are embedded in final_decoder_multiseed_comparison.json. |
| `checkpoints_gru_decoder/multimodal_map_gru_seed2026_best.pth` | Local checkpoint weight/log/history/manifest; final six-run lightweight manifests are embedded in final_decoder_multiseed_comparison.json. |
| `checkpoints_gru_decoder/seed123/multimodal_map_gru_seed123_best.json` | Local checkpoint weight/log/history/manifest; final six-run lightweight manifests are embedded in final_decoder_multiseed_comparison.json. |
| `checkpoints_gru_decoder/seed123/multimodal_map_gru_seed123_best.pth` | Local checkpoint weight/log/history/manifest; final six-run lightweight manifests are embedded in final_decoder_multiseed_comparison.json. |
| `checkpoints_gru_decoder/seed123/training.log` | Local checkpoint weight/log/history/manifest; final six-run lightweight manifests are embedded in final_decoder_multiseed_comparison.json. |
| `checkpoints_gru_decoder/seed123/training_history_seed123.json` | Local checkpoint weight/log/history/manifest; final six-run lightweight manifests are embedded in final_decoder_multiseed_comparison.json. |
| `checkpoints_gru_decoder/seed42/multimodal_map_gru_seed42_best.json` | Local checkpoint weight/log/history/manifest; final six-run lightweight manifests are embedded in final_decoder_multiseed_comparison.json. |
| `checkpoints_gru_decoder/seed42/multimodal_map_gru_seed42_best.pth` | Local checkpoint weight/log/history/manifest; final six-run lightweight manifests are embedded in final_decoder_multiseed_comparison.json. |
| `checkpoints_gru_decoder/seed42/training.log` | Local checkpoint weight/log/history/manifest; final six-run lightweight manifests are embedded in final_decoder_multiseed_comparison.json. |
| `checkpoints_gru_decoder/seed42/training_history_seed42.json` | Local checkpoint weight/log/history/manifest; final six-run lightweight manifests are embedded in final_decoder_multiseed_comparison.json. |
| `checkpoints_gru_decoder/training.log` | Local checkpoint weight/log/history/manifest; final six-run lightweight manifests are embedded in final_decoder_multiseed_comparison.json. |
| `checkpoints_gru_decoder/training_history.json` | Local checkpoint weight/log/history/manifest; final six-run lightweight manifests are embedded in final_decoder_multiseed_comparison.json. |
| `checkpoints_turn_aware/multimodal_map_seed2026_best.json` | Local checkpoint weight/log/history/manifest; final six-run lightweight manifests are embedded in final_decoder_multiseed_comparison.json. |
| `checkpoints_turn_aware/multimodal_map_seed2026_best.pth` | Local checkpoint weight/log/history/manifest; final six-run lightweight manifests are embedded in final_decoder_multiseed_comparison.json. |
| `checkpoints_turn_aware/sampling_distribution.json` | Local checkpoint weight/log/history/manifest; final six-run lightweight manifests are embedded in final_decoder_multiseed_comparison.json. |
| `checkpoints_turn_aware/training.log` | Local checkpoint weight/log/history/manifest; final six-run lightweight manifests are embedded in final_decoder_multiseed_comparison.json. |
| `checkpoints_turn_aware/training_history.json` | Local checkpoint weight/log/history/manifest; final six-run lightweight manifests are embedded in final_decoder_multiseed_comparison.json. |
| `data/processed/nuscenes_test_prediction_samples_map.pkl` | Large generated dataset; preserved locally and recorded by SHA256. |
| `data/processed/nuscenes_train_prediction_samples_map.pkl` | Large generated dataset; preserved locally and recorded by SHA256. |
| `data/processed/nuscenes_val_prediction_samples_map.pkl` | Large generated dataset; preserved locally and recorded by SHA256. |
| `doc/figures/gru_decoder_multiseed_comparison.png` | Historical or unrelated figure; kept locally to avoid committing redundant plots. No cleanup/deletion. |
| `doc/figures/gru_decoder_vs_baseline_seed2026.png` | Historical or unrelated figure; kept locally to avoid committing redundant plots. No cleanup/deletion. |
| `doc/figures/qualitative_candidates.png` | Historical or unrelated figure; kept locally to avoid committing redundant plots. No cleanup/deletion. |
| `doc/figures/train_turn_angle_distribution.png` | Historical or unrelated figure; kept locally to avoid committing redundant plots. No cleanup/deletion. |
| `doc/figures/train_vs_test_turn_angle_seed2026_60ep.png` | Historical or unrelated figure; kept locally to avoid committing redundant plots. No cleanup/deletion. |
| `doc/figures/turn_angle_analysis_seed123.png` | Historical or unrelated figure; kept locally to avoid committing redundant plots. No cleanup/deletion. |
| `doc/figures/turn_angle_analysis_seed2026.png` | Historical or unrelated figure; kept locally to avoid committing redundant plots. No cleanup/deletion. |
| `doc/figures/turn_angle_analysis_seed2026_60ep.png` | Historical or unrelated figure; kept locally to avoid committing redundant plots. No cleanup/deletion. |
| `doc/figures/turn_angle_analysis_seed42.png` | Invalid historical seed-42 result from an overwritten one-epoch checkpoint; excluded from final conclusions. |
| `doc/figures/turn_angle_map_comparison_seed123.png` | Historical or unrelated figure; kept locally to avoid committing redundant plots. No cleanup/deletion. |
| `doc/figures/turn_angle_map_comparison_seed2026.png` | Historical or unrelated figure; kept locally to avoid committing redundant plots. No cleanup/deletion. |
| `doc/figures/turn_aware_vs_baseline_seed2026.png` | Historical or unrelated figure; kept locally to avoid committing redundant plots. No cleanup/deletion. |
| `doc/results/gru_decoder_multiseed_summary.json` | Historical unmatched comparison, superseded by the matched final comparison; preserved without rewriting. |
| `doc/results/gru_decoder_multiseed_summary.md` | Historical unmatched comparison, superseded by the matched final comparison; preserved without rewriting. |
| `doc/results/turn_angle_analysis_seed42.json` | Invalid historical seed-42 result from an overwritten one-epoch checkpoint; excluded from final conclusions. |
| `scripts/run_interaction_analysis.py` | Pre-existing unrelated target-only MLP, interaction analysis, or qualitative inspection work; preserved. |
| `scripts/train_model.py` | Pre-existing unrelated target-only MLP, interaction analysis, or qualitative inspection work; preserved. |
| `tools/plot_qualitative_prediction.py` | Pre-existing unrelated target-only MLP, interaction analysis, or qualitative inspection work; preserved. |
| `utils/analysis_utils.py` | Pre-existing unrelated target-only MLP, interaction analysis, or qualitative inspection work; preserved. |
| `utils/models.py` | Pre-existing unrelated target-only MLP, interaction analysis, or qualitative inspection work; preserved. |

The original seed-2026 MLP manifest lacks historical sampler/train-validation dataset hashes. This is documented in the final comparison rather than filled in retroactively. All six model types and validation/test metrics were verified. All 63 pre-existing snapshot files, including baseline/GRU weights and unrelated user edits, remained byte-identical. Syntax checks, independent-metric/population-std checks, temporary training/backward/reload smoke checks, and isolated imports with the committed baseline utilities passed.
