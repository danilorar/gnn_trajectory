# White-box PGD attack: results

val split (`nuscenes_val_prediction_samples_map.pkl`), 500 random samples; perturbation on all 5 observed target positions, ≤ 1.0 m per point, nuScenes physical bounds from the original code, 3 ground-truth future points included in the physical check; 10 restarts × 100 steps. Each column is attacked with its own objective. Deviations use the min-ADE candidate (issue #18). 83.8% of clean traces satisfy the physical bounds; for the others a perturbation is only accepted if it brings the trace within bounds, so most stay unperturbed. "random, same size" = smooth random perturbations (offset + linear drift) scaled to the attack's mean size per point, with the same feasibility scaling, averaged over 20 draws.

| Model | | minADE6 | minFDE6 | Left | Right | Front | Rear |
|---|---|---:|---:|---:|---:|---:|---:|
| multimodal_map:2026 | clean | 1.486 | 3.159 | +0.090 | -0.090 | -0.337 | +0.337 |
|  | random, same size (0.20 m) | 1.514 | 3.187 | +0.081 | -0.081 | -0.313 | +0.313 |
|  | attacked | 2.777 | 5.957 | +1.125 | +0.861 | +1.586 | +2.088 |

Share of samples whose directional deviation exceeds half a lane width (1.85 m):

| Model | | Left | Right | Front | Rear | Mean of 4 |
|---|---|---:|---:|---:|---:|---:|
| multimodal_map:2026 | clean | 4.6% | 1.4% | 2.0% | 9.4% | 4.3% |
| | random, same size | 4.5% | 1.9% | 2.2% | 9.2% | 4.4% |
| | attacked | 21.2% | 15.2% | 44.4% | 51.6% | 33.1% |
| | attacked, attackable samples only | 22.9% | 16.7% | 51.6% | 54.4% | 36.4% |

Mean perturbation per observed point (m):

| Model | ade | fde | left | right | front | rear |
|---|---:|---:|---:|---:|---:|---:|
| multimodal_map:2026 | 0.19 | 0.19 | 0.18 | 0.18 | 0.18 | 0.18 |
