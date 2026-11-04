# White-box PGD attack: results

val split (`nuscenes_val_prediction_samples.pkl`), 500 random samples; perturbation on all 5 observed target positions, ≤ 1.0 m per point, nuScenes physical bounds from the original code, 3 ground-truth future points included in the physical check; 10 restarts × 100 steps. Each column is attacked with its own objective. Deviations use the min-ADE candidate (issue #18). 83.8% of clean traces satisfy the physical bounds; for the others a perturbation is only accepted if it brings the trace within bounds, so most stay unperturbed. "random, same size" = smooth random perturbations (offset + linear drift) scaled to the attack's mean size per point, with the same feasibility scaling, averaged over 20 draws.

| Model | | minADE6 | minFDE6 | Left | Right | Front | Rear |
|---|---|---:|---:|---:|---:|---:|---:|
| attention:2026 | clean | 3.648 | 8.922 | +0.299 | -0.299 | -1.043 | +1.043 |
|  | random, same size (0.23 m) | 3.699 | 8.974 | +0.322 | -0.322 | -1.054 | +1.054 |
|  | attacked | 8.512 | 18.444 | +2.668 | +2.084 | +4.448 | +5.844 |

Share of samples whose directional deviation exceeds half a lane width (1.85 m):

| Model | | Left | Right | Front | Rear | Mean of 4 |
|---|---|---:|---:|---:|---:|---:|
| attention:2026 | clean | 11.8% | 9.0% | 19.6% | 38.4% | 19.7% |
| | random, same size | 12.2% | 9.4% | 19.9% | 38.8% | 20.1% |
| | attacked | 70.8% | 49.6% | 74.0% | 84.4% | 69.7% |
| | attacked, attackable samples only | 77.1% | 53.7% | 84.2% | 89.0% | 76.0% |

Mean perturbation per observed point (m):

| Model | ade | fde | left | right | front | rear |
|---|---:|---:|---:|---:|---:|---:|
| attention:2026 | 0.21 | 0.20 | 0.19 | 0.19 | 0.25 | 0.24 |
