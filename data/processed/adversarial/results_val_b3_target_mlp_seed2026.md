# White-box PGD attack: results

val split (`nuscenes_val_prediction_samples.pkl`), 500 random samples; perturbation on all 5 observed target positions, ≤ 1.0 m per point, nuScenes physical bounds from the original code, 3 ground-truth future points included in the physical check; 10 restarts × 100 steps. Each column is attacked with its own objective. Deviations use the min-ADE candidate (issue #18). 83.8% of clean traces satisfy the physical bounds; for the others a perturbation is only accepted if it brings the trace within bounds, so most stay unperturbed. "random, same size" = smooth random perturbations (offset + linear drift) scaled to the attack's mean size per point, with the same feasibility scaling, averaged over 20 draws.

| Model | | minADE6 | minFDE6 | Left | Right | Front | Rear |
|---|---|---:|---:|---:|---:|---:|---:|
| target_mlp:2026 | clean | 3.640 | 8.911 | +0.071 | -0.071 | -1.142 | +1.142 |
|  | random, same size (0.23 m) | 3.688 | 8.956 | +0.089 | -0.089 | -1.144 | +1.144 |
|  | attacked | 8.604 | 18.599 | +2.476 | +2.129 | +4.422 | +6.086 |

Share of samples whose directional deviation exceeds half a lane width (1.85 m):

| Model | | Left | Right | Front | Rear | Mean of 4 |
|---|---|---:|---:|---:|---:|---:|
| target_mlp:2026 | clean | 10.4% | 9.6% | 19.8% | 40.6% | 20.1% |
| | random, same size | 10.5% | 10.4% | 19.9% | 40.7% | 20.4% |
| | attacked | 65.6% | 52.8% | 73.0% | 84.6% | 69.0% |
| | attacked, attackable samples only | 71.6% | 56.6% | 82.8% | 88.5% | 74.9% |

Mean perturbation per observed point (m):

| Model | ade | fde | left | right | front | rear |
|---|---:|---:|---:|---:|---:|---:|
| target_mlp:2026 | 0.21 | 0.20 | 0.19 | 0.19 | 0.25 | 0.23 |
