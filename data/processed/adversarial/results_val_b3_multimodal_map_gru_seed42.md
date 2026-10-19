# White-box PGD attack: results

val split (`nuscenes_val_prediction_samples_map.pkl`), 500 random samples; perturbation on all 5 observed target positions, ≤ 1.0 m per point, nuScenes physical bounds from the original code, 3 ground-truth future points included in the physical check; 10 restarts × 100 steps. Each column is attacked with its own objective. Deviations use the min-ADE candidate (issue #18). 83.8% of clean traces satisfy the physical bounds; for the others a perturbation is only accepted if it brings the trace within bounds, so most stay unperturbed. "random, same size" = smooth random perturbations (offset + linear drift) scaled to the attack's mean size per point, with the same feasibility scaling, averaged over 20 draws.

| Model | | minADE6 | minFDE6 | Left | Right | Front | Rear |
|---|---|---:|---:|---:|---:|---:|---:|
| multimodal_map_gru:42 | clean | 1.455 | 3.013 | -0.044 | +0.044 | -0.287 | +0.287 |
|  | random, same size (0.18 m) | 1.470 | 3.018 | -0.045 | +0.045 | -0.310 | +0.310 |
|  | attacked | 2.849 | 6.081 | +0.895 | +1.129 | +1.688 | +2.278 |

Share of samples whose directional deviation exceeds half a lane width (1.85 m):

| Model | | Left | Right | Front | Rear | Mean of 4 |
|---|---|---:|---:|---:|---:|---:|
| multimodal_map_gru:42 | clean | 3.4% | 3.8% | 3.0% | 8.2% | 4.6% |
| | random, same size | 3.3% | 3.5% | 2.9% | 7.9% | 4.4% |
| | attacked | 16.2% | 21.6% | 45.2% | 54.2% | 34.3% |
| | attacked, attackable samples only | 16.7% | 24.3% | 51.8% | 59.9% | 38.2% |

Mean perturbation per observed point (m):

| Model | ade | fde | left | right | front | rear |
|---|---:|---:|---:|---:|---:|---:|
| multimodal_map_gru:42 | 0.18 | 0.19 | 0.18 | 0.18 | 0.18 | 0.18 |
