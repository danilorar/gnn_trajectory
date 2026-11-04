# White-box PGD attack: results

val split (`nuscenes_val_prediction_samples.pkl`), 500 random samples; perturbation on all 5 observed target positions, ≤ 1.0 m per point, nuScenes physical bounds from the original code, 3 ground-truth future points included in the physical check; 10 restarts × 100 steps. Each column is attacked with its own objective. Deviations use the min-ADE candidate (issue #18). 83.8% of clean traces satisfy the physical bounds; for the others a perturbation is only accepted if it brings the trace within bounds, so most stay unperturbed. "random, same size" = smooth random perturbations (offset + linear drift) scaled to the attack's mean size per point, with the same feasibility scaling, averaged over 20 draws.

| Model | | minADE6 | minFDE6 | Left | Right | Front | Rear |
|---|---|---:|---:|---:|---:|---:|---:|
| target_mlp:123 | clean | 3.622 | 8.884 | +0.364 | -0.364 | -1.084 | +1.084 |
|  | random, same size (0.23 m) | 3.675 | 8.937 | +0.382 | -0.382 | -1.071 | +1.071 |
|  | attacked | 8.264 | 17.812 | +2.835 | +1.922 | +4.198 | +5.846 |

Share of samples whose directional deviation exceeds half a lane width (1.85 m):

| Model | | Left | Right | Front | Rear | Mean of 4 |
|---|---|---:|---:|---:|---:|---:|
| target_mlp:123 | clean | 12.0% | 8.4% | 19.4% | 38.4% | 19.6% |
| | random, same size | 12.6% | 8.7% | 20.1% | 38.4% | 19.9% |
| | attacked | 70.6% | 46.8% | 71.8% | 85.6% | 68.7% |
| | attacked, attackable samples only | 76.8% | 50.4% | 81.6% | 89.7% | 74.6% |

Mean perturbation per observed point (m):

| Model | ade | fde | left | right | front | rear |
|---|---:|---:|---:|---:|---:|---:|
| target_mlp:123 | 0.21 | 0.20 | 0.19 | 0.19 | 0.24 | 0.25 |
