# White-box PGD attack: results

val split, 500 random samples; perturbation on all 5 observed target positions, ≤ 1.0 m per point, nuScenes physical bounds from the original code, 3 ground-truth future points included in the physical check; 10 restarts × 100 steps. Each column is attacked with its own objective. Deviations use the min-ADE candidate (issue #18). 83.8% of clean traces satisfy the physical bounds; for the others a perturbation is only accepted if it brings the trace within bounds, so most stay unperturbed. "random, same size" = smooth random perturbations (offset + linear drift) scaled to the attack's mean size per point, with the same feasibility scaling, averaged over 20 draws.

| Model | | minADE6 | minFDE6 | Left | Right | Front | Rear |
|---|---|---:|---:|---:|---:|---:|---:|
| multimodal:42 | clean | 1.719 | 3.851 | +0.118 | -0.118 | -0.301 | +0.301 |
|  | random, same size (0.18 m) | 1.742 | 3.858 | +0.147 | -0.147 | -0.293 | +0.293 |
|  | attacked | 3.526 | 7.680 | +1.887 | +1.356 | +1.962 | +2.436 |
| multimodal:123 | clean | 1.704 | 3.866 | +0.071 | -0.071 | -0.357 | +0.357 |
|  | random, same size (0.18 m) | 1.732 | 3.873 | +0.062 | -0.062 | -0.345 | +0.345 |
|  | attacked | 3.463 | 7.790 | +1.666 | +1.449 | +2.019 | +2.567 |
| multimodal:2026 | clean | 1.718 | 3.891 | +0.067 | -0.067 | -0.502 | +0.502 |
|  | random, same size (0.20 m) | 1.741 | 3.896 | +0.086 | -0.086 | -0.481 | +0.481 |
|  | attacked | 3.392 | 7.588 | +1.560 | +1.366 | +1.772 | +2.764 |
| cv | clean | 4.456 | 10.883 | -0.027 | +0.027 | -1.309 | +1.309 |
|  | random, same size (0.18 m) | 4.494 | 10.912 | -0.031 | +0.031 | -1.308 | +1.308 |
|  | attacked | 8.003 | 17.190 | +1.704 | +1.725 | +2.565 | +5.111 |

Share of samples whose directional deviation exceeds half a lane width (1.85 m):

| Model | | Left | Right | Front | Rear | Mean of 4 |
|---|---|---:|---:|---:|---:|---:|
| multimodal:42 | clean | 7.6% | 4.8% | 3.2% | 10.6% | 6.5% |
| | random, same size | 8.3% | 3.8% | 3.6% | 10.7% | 6.6% |
| | attacked | 49.0% | 35.8% | 70.4% | 71.6% | 56.7% |
| | attacked, attackable samples only | 53.0% | 40.8% | 81.1% | 77.3% | 63.1% |
| multimodal:123 | clean | 7.0% | 5.6% | 4.8% | 12.2% | 7.4% |
| | random, same size | 6.5% | 5.7% | 4.5% | 12.3% | 7.3% |
| | attacked | 39.4% | 34.6% | 66.6% | 71.2% | 53.0% |
| | attacked, attackable samples only | 42.5% | 38.9% | 77.3% | 77.8% | 59.1% |
| multimodal:2026 | clean | 8.0% | 5.8% | 2.4% | 13.0% | 7.3% |
| | random, same size | 7.7% | 5.5% | 2.7% | 12.9% | 7.2% |
| | attacked | 38.6% | 30.4% | 57.2% | 70.0% | 49.1% |
| | attacked, attackable samples only | 40.8% | 33.2% | 66.1% | 75.4% | 53.9% |
| cv | clean | 10.8% | 11.4% | 21.4% | 40.6% | 21.0% |
| | random, same size | 11.0% | 11.5% | 21.3% | 40.7% | 21.1% |
| | attacked | 40.6% | 33.0% | 63.0% | 81.2% | 54.4% |
| | attacked, attackable samples only | 42.5% | 32.7% | 70.6% | 82.8% | 57.2% |

Mean perturbation per observed point (m):

| Model | ade | fde | left | right | front | rear |
|---|---:|---:|---:|---:|---:|---:|
| multimodal:42 | 0.19 | 0.19 | 0.18 | 0.16 | 0.20 | 0.18 |
| multimodal:123 | 0.19 | 0.19 | 0.17 | 0.17 | 0.18 | 0.18 |
| multimodal:2026 | 0.20 | 0.20 | 0.16 | 0.17 | 0.19 | 0.19 |
| cv | 0.17 | 0.16 | 0.17 | 0.17 | 0.21 | 0.20 |
