# Neighbor radius experiment results

Final neighbor-selection radius: **35 m**. Retain this radius unless a bug requires correction.

Protocol: seeds 42, 123, 2026; radii 20, 35, 50 m; 30 epochs per model; batch size 64; Adam learning rate 0.001; ADE loss; checkpoint selected by validation ADE; four past steps plus current position; twelve future steps. Each model used 25,711 training, 6,685 validation, and 7,242 test samples. Errors are in metres; lower is better.

| Seed | Radius | Val ADE | Val FDE | Test ADE | Test FDE |
|---|---:|---:|---:|---:|---:|
| 42 | 20 m | 3.654 | 8.817 | 3.588 | 8.499 |
| 42 | 35 m | 3.610 | 8.702 | 3.545 | 8.348 |
| 42 | 50 m | 3.851 | 9.358 | 3.776 | 8.927 |
| 123 | 20 m | 3.684 | 8.896 | 3.579 | 8.504 |
| 123 | 35 m | 3.550 | 8.534 | 3.443 | 8.132 |
| 123 | 50 m | 3.873 | 9.384 | 3.787 | 8.974 |
| 2026 | 20 m | 3.878 | 9.379 | 3.820 | 9.033 |
| 2026 | 35 m | 3.862 | 9.346 | 3.829 | 9.073 |
| 2026 | 50 m | 3.861 | 9.328 | 3.833 | 9.068 |

## Mean across seeds

| Radius | Val ADE | Val FDE | Test ADE | Test FDE |
|---|---:|---:|---:|---:|
| 20 m | 3.739 | 9.030 | 3.662 | 8.679 |
| 35 m | 3.674 | 8.860 | 3.606 | 8.518 |
| 50 m | 3.862 | 9.357 | 3.799 | 8.990 |

35 m has the lowest mean validation ADE/FDE and test ADE/FDE. It wins on test ADE for seeds 42 and 123; seed 2026 slightly favors 20 m on test ADE. The seed variation means the average advantage is not a guarantee for every run.

All nine runs completed successfully. Reports and checkpoints were verified locally under `data/processed/radius_sweep/seed{42,123,2026}/`; large datasets and checkpoints are not included in Git.

Cache limitation: filenames do not encode dataset source/version. Use a new output directory or `--force` when changing the source/version.
