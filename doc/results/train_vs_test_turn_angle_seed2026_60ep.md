# Training versus test turn-angle evaluation

Seed 2026 map-aware model; 60 epochs, best epoch 59. Existing weights only; no fitting or retraining. Absolute endpoint-angle bins, final displacement >=5 m, independent six-mode minima. Standard deviations use ddof=0.

## Training statistics (metres)

| Angle | n | ADE mean | ADE median | ADE std | FDE mean | FDE median | FDE std |
|---|---:|---:|---:|---:|---:|---:|---:|
| 0-15 deg | 20605 | 1.256787 | 1.095041 | 0.798706 | 2.377112 | 2.022075 | 1.751191 |
| 15-30 deg | 1768 | 1.999079 | 1.760528 | 1.113246 | 4.835199 | 4.023803 | 3.311065 |
| 30-60 deg | 2204 | 2.347001 | 1.878805 | 1.691208 | 5.439500 | 4.112936 | 4.671914 |
| 60+ deg | 334 | 3.387345 | 2.459070 | 2.658201 | 8.028499 | 5.763192 | 6.831965 |

## Train/test means and gaps (metres)

Gap = test minus train.

| Angle | Train ADE | Test ADE | ADE gap | Train FDE | Test FDE | FDE gap |
|---|---:|---:|---:|---:|---:|---:|
| 0-15 deg | 1.256787 | 1.315389 | 0.058602 | 2.377112 | 2.509393 | 0.132281 |
| 15-30 deg | 1.999079 | 2.395072 | 0.395993 | 4.835199 | 5.864170 | 1.028971 |
| 30-60 deg | 2.347001 | 3.130648 | 0.783647 | 5.439500 | 7.608039 | 2.168540 |
| 60+ deg | 3.387345 | 3.945208 | 0.557862 | 8.028499 | 9.365035 | 1.336536 |

Retained train: 24,911 of 25,711, excluding 800 below 5 m; no invalid futures. Retained test: 7,024.

Retained train minADE6/minFDE6: 1.434492 / 2.898286 m.
Retained test minADE6/minFDE6: 1.544532 / 3.163905 m.

## Interpretation

Mixed behavior (C), with substantial residual training-set difficulty: both training metrics rise strongly with angle, and test error is higher. This is not a pure generalization failure. Generalization gaps are largest at 30–60 degrees, smaller at 60+ degrees, and not monotonic in angle.

The training error demonstrates that sharp-turn examples are not fitted as well as straight examples under this trained model/objective. It does not isolate insufficient parameter capacity from optimization, mode coverage, data imbalance, ambiguous futures, or differences in scene difficulty. Train/test bins contain different scenes and trajectory distributions; this is one seed with no scene-level uncertainty estimate. Endpoint angle measures net deviation rather than path curvature.

Sanity checks: checkpoint SHA256 matches the saved 60-epoch test evaluation; saved full-test metrics match the manifest; test dataset hash is unchanged; training metrics agree with the shared evaluator. No training-metric manifest is available, and train metrics were not compared to test-manifest values.
