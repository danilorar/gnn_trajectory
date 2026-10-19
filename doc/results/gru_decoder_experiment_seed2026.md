# GRU future-decoder experiment, seed 2026

## Design and controls

New `GraphNetworkAttentionMultimodalWithMapGRU` subclasses the baseline map-aware model and replaces only `decoder`. The inherited node encoder, two GATv2 layers, pooled map encoder, fused 96-dimensional context and forward path remain unchanged. Baseline model-class AST checks and identical seed-2026 backbone/map initialization were verified.

For each mode: concatenate fused context with a learned 16-dimensional mode embedding, project through a linear layer and tanh to a 64-dimensional initial hidden state, then run a shared one-layer GRUCell for 12 steps. Input is the previous predicted displacement, initially (0,0). A linear hidden-to-xy head predicts each displacement; cumulative sum reconstructs absolute positions. No teacher forcing or ground-truth future inputs. Output remains [B,6,12,2].

Decoder parameter counts: baseline 30,992; GRU 20,514. Adam lr=0.001, batch size 64, 60 epochs, seed 2026, standard uniform shuffle without replacement, unchanged best-of-six ADE and unweighted validation selection. No turn-aware sampler. Fresh initialization, not resumed from a trained checkpoint.

Smoke test used an automatically removed temporary directory. Verified shape, loss, finite gradients, cumulative displacement reconstruction, checkpoint reload and evaluation before the full CUDA run.

Best epoch: 53; validation minADE6/minFDE6: 1.508728 / 3.045300 m.

## Full-test results

Change = GRU minus baseline, negative is improvement.

| Metric | Baseline | GRU | Change (m) | Change (%) |
|---|---:|---:|---:|---:|
| minADE6 | 1.527754 | 1.452923 | -0.074831 | -4.898117 |
| minFDE6 | 3.115541 | 2.872186 | -0.243355 | -7.811014 |

## minADE6 by angle

| Angle | n | Baseline | GRU | Change (m) | Change (%) |
|---|---:|---:|---:|---:|---:|
| 0-15 deg | 6022 | 1.315389 | 1.280135 | -0.035254 | -2.680110 |
| 15-30 deg | 381 | 2.395072 | 2.308196 | -0.086876 | -3.627283 |
| 30-60 deg | 534 | 3.130648 | 2.646420 | -0.484228 | -15.467337 |
| 60+ deg | 87 | 3.945208 | 3.748494 | -0.196713 | -4.986129 |

## minFDE6 by angle

| Angle | n | Baseline | GRU | Change (m) | Change (%) |
|---|---:|---:|---:|---:|---:|
| 0-15 deg | 6022 | 2.509393 | 2.424226 | -0.085167 | -3.393927 |
| 15-30 deg | 381 | 5.864170 | 5.447601 | -0.416569 | -7.103627 |
| 30-60 deg | 534 | 7.608039 | 5.666943 | -1.941096 | -25.513756 |
| 60+ deg | 87 | 9.365035 | 8.960498 | -0.404537 | -4.319656 |

## Interpretation

The decoder change improves 30–60 degree errors substantially (~15.5% ADE, ~25.5% FDE), improves mean 60+ errors modestly (~5.0% ADE, ~4.3% FDE), improves straight trajectories, and improves full-test ADE/FDE (~4.9%/~7.8%). This satisfies the planned descriptive success criterion and supports the original decoder formulation as one contributor to prediction difficulty.

It does not establish the decoder as the main or sole cause: sharp-turn error remains high, this is one seed, and the sharpest bin has only 87 examples. Temporal recurrence, relative displacement representation, mode conditioning, parameter count and optimization dynamics change together within the decoder; the experiment does not isolate their individual contributions. No confidence intervals or further architectural tuning were performed.

All 7,242 test samples were evaluated; 218 below 5 m displacement were excluded from bins, retaining 7,024. Endpoint angle, bin definitions, independent mode minima and metrics are unchanged. Evaluation matched the new manifest and shared evaluator. Baseline checkpoints, processed datasets and protected model/graph/training-utility sources were verified unchanged.

## Commands

```bash
python scripts/train_gru_decoder.py --checkpoint-dir checkpoints_gru_decoder
python scripts/run_turn_angle_analysis.py --seed 2026 --checkpoint checkpoints_gru_decoder/multimodal_map_gru_seed2026_best.pth --output doc/results/turn_angle_analysis_seed2026_gru_decoder.json
```

Training refuses to overwrite an existing checkpoint; use a new directory to repeat the run.
