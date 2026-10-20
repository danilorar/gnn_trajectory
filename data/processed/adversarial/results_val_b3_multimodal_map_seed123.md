# White-box PGD attack: results

val split (`nuscenes_val_prediction_samples_map.pkl`), 500 random samples; perturbation on all 5 observed target positions, ≤ 1.0 m per point, nuScenes physical bounds from the original code, 3 ground-truth future points included in the physical check; 10 restarts × 100 steps. Each column is attacked with its own objective. Deviations use the min-ADE candidate (issue #18). 83.8% of clean traces satisfy the physical bounds; for the others a perturbation is only accepted if it brings the trace within bounds, so most stay unperturbed. "random, same size" = smooth random perturbations (offset + linear drift) scaled to the attack's mean size per point, with the same feasibility scaling, averaged over 20 draws.

| Model | | minADE6 | minFDE6 | Left | Right | Front | Rear |
|---|---|---:|---:|---:|---:|---:|---:|
| multimodal_map:123 | clean | 1.489 | 3.229 | +0.089 | -0.089 | -0.232 | +0.232 |
|  | random, same size (0.18 m) | 1.510 | 3.234 | +0.094 | -0.094 | -0.249 | +0.249 |
|  | attacked | 2.862 | 6.113 | +1.319 | +1.054 | +1.656 | +2.031 |

Share of samples whose directional deviation exceeds half a lane width (1.85 m):

| Model | | Left | Right | Front | Rear | Mean of 4 |
|---|---|---:|---:|---:|---:|---:|
| multimodal_map:123 | clean | 5.4% | 2.0% | 1.6% | 7.4% | 4.1% |
| | random, same size | 5.2% | 2.0% | 1.7% | 7.8% | 4.2% |
| | attacked | 27.2% | 19.8% | 48.8% | 48.8% | 36.2% |
| | attacked, attackable samples only | 29.8% | 22.0% | 56.8% | 52.5% | 40.3% |

Mean perturbation per observed point (m):

| Model | ade | fde | left | right | front | rear |
|---|---:|---:|---:|---:|---:|---:|
| multimodal_map:123 | 0.19 | 0.19 | 0.18 | 0.17 | 0.18 | 0.19 |
