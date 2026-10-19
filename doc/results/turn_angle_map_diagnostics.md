# Turn-angle map/no-map diagnostics

Evaluated existing fully trained checkpoints on aligned test samples. No retraining, architecture changes, or dataset changes. All errors are metres. Standard deviations are population values (`ddof=0`). Bins are [0,15), [15,30), [30,60), and [60,180] degrees, with final displacement >= 5 m.

## Checkpoint provenance and sanity checks

The canonical no-map checkpoint paths for seeds 123/2026 are absent. The archived 30-epoch checkpoints in `data/processed/multimodal_seed_check/seed{seed}_best_model.pth` were used instead. Their computed test minADE6 and historical FDE-at-best-ADE match `results.json`. The archived report does not contain independent minFDE6, so that metric was checked against the shared evaluator rather than falsely compared to the historical FDE field. Map minADE6/minFDE6 match each seed's saved manifest. All checks passed within 0.001 m absolute / 0.0001 relative tolerance.

| Seed | Model | Full-test minADE6 | Full-test minFDE6 | Retained minADE6 | Retained minFDE6 |
|---|---|---:|---:|---:|---:|
| 123 | no_map | 1.717576 | 3.692145 | 1.741996 | 3.746259 |
| 123 | map | 1.567762 | 3.200677 | 1.589028 | 3.254096 |
| 2026 | no_map | 1.721122 | 3.640284 | 1.749730 | 3.712601 |
| 2026 | map | 1.535276 | 3.116311 | 1.558612 | 3.172916 |

All 7,242 test identities and original ordering match; target past/future, category, neighbor identities/order, and neighbor histories are identical. 218 samples are excluded for displacement below 5 m, retaining 7,024.

## Seed 123: exact bin statistics

| Angle | n | Model | ADE mean | ADE median | ADE std | FDE mean | FDE median | FDE std |
|---|---:|---|---:|---:|---:|---:|---:|---:|
| 0-15 deg | 6022 | no_map | 1.458163 | 1.259481 | 0.965992 | 2.924462 | 2.472517 | 2.191478 |
| 0-15 deg | 6022 | map | 1.330595 | 1.171825 | 0.808990 | 2.563735 | 2.234556 | 1.795116 |
| 15-30 deg | 381 | no_map | 2.848251 | 2.526242 | 1.453856 | 7.422850 | 6.880992 | 3.888276 |
| 15-30 deg | 381 | map | 2.439918 | 2.281804 | 1.272344 | 5.718202 | 5.004078 | 3.561138 |
| 30-60 deg | 534 | no_map | 3.711577 | 2.954500 | 2.467189 | 9.220682 | 7.379447 | 6.497144 |
| 30-60 deg | 534 | map | 3.495997 | 2.777698 | 2.341660 | 8.203018 | 6.660022 | 5.903619 |
| 60+ deg | 87 | no_map | 4.454700 | 4.285463 | 2.843554 | 10.927145 | 10.269177 | 7.122231 |
| 60+ deg | 87 | map | 4.046171 | 3.379725 | 3.049274 | 9.872581 | 8.402331 | 8.092343 |

| Angle | Mean ADE improvement (m) | ADE improvement (%) | Mean FDE improvement (m) | FDE improvement (%) |
|---|---:|---:|---:|---:|
| 0-15 deg | 0.127568 | 8.748547 | 0.360727 | 12.334822 |
| 15-30 deg | 0.408334 | 14.336295 | 1.704648 | 22.964875 |
| 30-60 deg | 0.215580 | 5.808317 | 1.017664 | 11.036754 |
| 60+ deg | 0.408529 | 9.170747 | 1.054564 | 9.650863 |

Improvement = no-map minus map; positive means map helps. Percentage = 100 × (no-map bin mean − map bin mean) / no-map bin mean, not the mean of per-sample percentages.

## Seed 2026: exact bin statistics

| Angle | n | Model | ADE mean | ADE median | ADE std | FDE mean | FDE median | FDE std |
|---|---:|---|---:|---:|---:|---:|---:|---:|
| 0-15 deg | 6022 | no_map | 1.481703 | 1.216085 | 1.077038 | 2.938166 | 2.401509 | 2.411337 |
| 0-15 deg | 6022 | map | 1.303255 | 1.127473 | 0.858865 | 2.454517 | 2.069614 | 1.908413 |
| 15-30 deg | 381 | no_map | 2.835526 | 2.541453 | 1.554004 | 7.404675 | 6.668484 | 4.137472 |
| 15-30 deg | 381 | map | 2.371418 | 2.037972 | 1.285323 | 5.842646 | 4.785156 | 3.841666 |
| 30-60 deg | 534 | no_map | 3.612585 | 2.805989 | 2.414998 | 8.749297 | 6.747591 | 6.318059 |
| 30-60 deg | 534 | map | 3.445154 | 2.738044 | 2.275764 | 8.262274 | 6.232299 | 6.188274 |
| 60+ deg | 87 | no_map | 4.113033 | 3.682962 | 2.936911 | 10.234086 | 9.687441 | 7.414493 |
| 60+ deg | 87 | map | 4.095009 | 3.788018 | 2.839941 | 9.969656 | 8.763003 | 7.470127 |

| Angle | Mean ADE improvement (m) | ADE improvement (%) | Mean FDE improvement (m) | FDE improvement (%) |
|---|---:|---:|---:|---:|
| 0-15 deg | 0.178448 | 12.043455 | 0.483649 | 16.460927 |
| 15-30 deg | 0.464108 | 16.367620 | 1.562029 | 21.095171 |
| 30-60 deg | 0.167432 | 4.634673 | 0.487023 | 5.566427 |
| 60+ deg | 0.018024 | 0.438213 | 0.264430 | 2.583819 |

Improvement = no-map minus map; positive means map helps. Percentage = 100 × (no-map bin mean − map bin mean) / no-map bin mean, not the mean of per-sample percentages.

## Training distribution

Total: 25,711; invalid: 0; excluded displacement <5 m: 800; retained: 24,911.

| Angle | n | Percentage of retained train samples |
|---|---:|---:|
| 0-15 deg | 20605 | 82.714463% |
| 15-30 deg | 1768 | 7.097266% |
| 30-60 deg | 2204 | 8.847497% |
| 60+ deg | 334 | 1.340773% |

## Interpretation

1. Map context does not help equally: both seeds show their largest percentage ADE reduction at 15–30 degrees, with reduced gains at 30–60 degrees.
2. At 60+ degrees seed 123 retains a 9.17% mean ADE benefit, while seed 2026 has only 0.44%. The magnitude of map benefit on the sharpest turns is not consistent across seeds.
3. No angle bin has worse mean ADE or mean FDE with maps. However, seed 2026's 60+ median ADE is worse with maps (3.788018 vs 3.682962 m); bin means do not imply every sample improves.
4. Both map and no-map models degrade strongly on sharper turns. This favors explanation B (a broader trajectory-prediction difficulty). The training imbalance makes C plausible, but does not establish that rarity causes the errors.
5. Training samples >=30 degrees are 10.188270% (test: 8.841116%); 60+ degrees are 1.340773% of retained training samples. Sharp turns are rare relative to straight motion, not rarer in train than in this test set.
6. A (pooled map geometry is specifically responsible) is not isolated by these diagnostics: maps help on average and no-map also struggles. A combination of B and a possible contribution from C is better supported; A remains an untested mechanism.

Caveats: endpoint angle is net deviation, not curvature or accumulated heading change. Travel distance, speed, scene difficulty, mode diversity, optimization and capacity can confound comparisons. The sharpest test bin has only 87 samples, evaluated with two seeds on the same examples; no scene-level uncertainty estimates or controlled sampling/encoder experiments were performed.
