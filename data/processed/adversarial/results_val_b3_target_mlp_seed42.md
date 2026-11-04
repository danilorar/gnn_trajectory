# White-box PGD attack: results

val split (`nuscenes_val_prediction_samples.pkl`), 500 random samples; perturbation on all 5 observed target positions, ≤ 1.0 m per point, nuScenes physical bounds from the original code, 3 ground-truth future points included in the physical check; 10 restarts × 100 steps. Each column is attacked with its own objective. Deviations use the min-ADE candidate (issue #18). 83.8% of clean traces satisfy the physical bounds; for the others a perturbation is only accepted if it brings the trace within bounds, so most stay unperturbed. "random, same size" = smooth random perturbations (offset + linear drift) scaled to the attack's mean size per point, with the same feasibility scaling, averaged over 20 draws.

| Model | | minADE6 | minFDE6 | Left | Right | Front | Rear |
|---|---|---:|---:|---:|---:|---:|---:|
| target_mlp:42 | clean | 3.626 | 8.908 | +0.050 | -0.050 | -0.906 | +0.906 |
|  | random, same size (0.23 m) | 3.673 | 8.947 | +0.067 | -0.067 | -0.899 | +0.899 |
|  | attacked | 8.689 | 18.847 | +2.530 | +2.115 | +4.799 | +5.923 |

Share of samples whose directional deviation exceeds half a lane width (1.85 m):

| Model | | Left | Right | Front | Rear | Mean of 4 |
|---|---|---:|---:|---:|---:|---:|
| target_mlp:42 | clean | 10.8% | 10.2% | 21.8% | 36.0% | 19.7% |
| | random, same size | 10.8% | 10.6% | 22.0% | 36.5% | 20.0% |
| | attacked | 63.2% | 52.4% | 76.4% | 82.8% | 68.7% |
| | attacked, attackable samples only | 68.5% | 56.3% | 86.6% | 86.6% | 74.5% |

Mean perturbation per observed point (m):

| Model | ade | fde | left | right | front | rear |
|---|---:|---:|---:|---:|---:|---:|
| target_mlp:42 | 0.20 | 0.20 | 0.19 | 0.18 | 0.24 | 0.23 |
