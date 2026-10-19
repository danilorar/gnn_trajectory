# Turn-aware training experiment, seed 2026

Fresh 60-epoch map-aware model. Same architecture, map encoder, decoder, processed data, splits, Adam lr=0.001, batch size 64, best-of-six ADE loss and unweighted validation selection as baseline. Only training sampling changed.

Sampling: with replacement; non-stationary endpoint-angle weights 1/1/2/4 for 0–15/15–30/30–60/60+ degrees. Displacement below 5 m has weight 1. Exactly 25,711 draws and 402 batches per epoch. Separate generators preserve model initialization RNG and make the preview match the first training draw.

| Group | Original n | Original % | First epoch draws | First epoch % | Weight |
|---|---:|---:|---:|---:|---:|
| stationary (<5 m) | 800 | 3.111509 | 714 | 2.777022 | 1 |
| 0-15 deg | 20605 | 80.140796 | 18346 | 71.354673 | 1 |
| 15-30 deg | 1768 | 6.876434 | 1567 | 6.094668 | 1 |
| 30-60 deg | 2204 | 8.572206 | 3897 | 15.156937 | 2 |
| 60+ deg | 334 | 1.299055 | 1187 | 4.616701 | 4 |

The first epoch contains 15,848 unique examples; repeats are expected with replacement. Validation/test are sequential and unweighted.

Best epoch: 29 (baseline: 59). Validation minADE6/minFDE6: 1.628415 / 3.387648 m.

## Full-test comparison

Change = turn-aware minus baseline; negative is better.

| Metric | Baseline | Turn-aware | Change (m) | Change (%) |
|---|---:|---:|---:|---:|
| minADE6 | 1.527754 | 1.560115 | +0.032361 | +2.118184 |
| minFDE6 | 3.115541 | 3.161153 | +0.045612 | +1.464000 |

## minADE6 by angle

| Angle | n | Baseline | Turn-aware | Change (m) | Change (%) |
|---|---:|---:|---:|---:|---:|
| 0-15 deg | 6022 | 1.315389 | 1.384869 | +0.069480 | +5.282088 |
| 15-30 deg | 381 | 2.395072 | 2.374505 | -0.020566 | -0.858699 |
| 30-60 deg | 534 | 3.130648 | 2.767417 | -0.363231 | -11.602416 |
| 60+ deg | 87 | 3.945208 | 4.004106 | +0.058898 | +1.492908 |

## minFDE6 by angle

| Angle | n | Baseline | Turn-aware | Change (m) | Change (%) |
|---|---:|---:|---:|---:|---:|
| 0-15 deg | 6022 | 2.509393 | 2.714968 | +0.205574 | +8.192180 |
| 15-30 deg | 381 | 5.864170 | 5.323496 | -0.540674 | -9.219960 |
| 30-60 deg | 534 | 7.608039 | 6.103399 | -1.504640 | -19.776980 |
| 60+ deg | 87 | 9.365035 | 9.671204 | +0.306168 | +3.269268 |

## Interpretation

30–60 degree trajectories improve clearly in magnitude (~11.6% ADE, ~19.8% FDE). Mean 60+ degree errors do not improve; both worsen slightly, although the median ADE improves. Straight trajectories worsen and full-test means degrade. This supports a contribution from training emphasis for the 30–60 degree range, but does not establish imbalance as the main explanation for all sharp-turn failures.

Caveats: one seed, only 87 test examples at 60+ degrees, no scene-level confidence intervals or uniform-with-replacement control. Replacement also reduces unique examples seen in any individual epoch. The best unweighted-validation checkpoint is epoch 29 versus baseline epoch 59; final reporting follows the same selection rule and does not select checkpoints by turn-bin test results. Endpoint angle measures net deviation, not curvature.

No baseline checkpoint or datasets were overwritten. All 60 epochs completed on CUDA. Existing test evaluator matches independently computed per-sample metrics. Model/graph/training utility hashes and dataset hashes were verified unchanged.

## Reproduction

```bash
python scripts/run_turn_aware_training.py --checkpoint-dir checkpoints_turn_aware
python scripts/run_turn_angle_analysis.py --seed 2026 --checkpoint checkpoints_turn_aware/multimodal_map_seed2026_best.pth --output doc/results/turn_angle_analysis_seed2026_turn_aware.json
```

The training command refuses to overwrite an existing experiment checkpoint. Use a new directory for another run.
