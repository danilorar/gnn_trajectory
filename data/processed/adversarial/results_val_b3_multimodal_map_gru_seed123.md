# White-box PGD attack: results

val split (`nuscenes_val_prediction_samples_map.pkl`), 500 random samples; perturbation on all 5 observed target positions, ≤ 1.0 m per point, nuScenes physical bounds from the original code, 3 ground-truth future points included in the physical check; 10 restarts × 100 steps. Each column is attacked with its own objective. Deviations use the min-ADE candidate (issue #18). 83.8% of clean traces satisfy the physical bounds; for the others a perturbation is only accepted if it brings the trace within bounds, so most stay unperturbed. "random, same size" = smooth random perturbations (offset + linear drift) scaled to the attack's mean size per point, with the same feasibility scaling, averaged over 20 draws.

| Model | | minADE6 | minFDE6 | Left | Right | Front | Rear |
|---|---|---:|---:|---:|---:|---:|---:|
| multimodal_map_gru:123 | clean | 1.472 | 3.080 | +0.023 | -0.023 | -0.339 | +0.339 |
|  | random, same size (0.20 m) | 1.500 | 3.102 | +0.016 | -0.016 | -0.323 | +0.323 |
|  | attacked | 2.793 | 5.905 | +0.872 | +0.895 | +1.592 | +2.153 |

Share of samples whose directional deviation exceeds half a lane width (1.85 m):

| Model | | Left | Right | Front | Rear | Mean of 4 |
|---|---|---:|---:|---:|---:|---:|
| multimodal_map_gru:123 | clean | 4.2% | 2.4% | 2.2% | 10.0% | 4.7% |
| | random, same size | 4.0% | 2.5% | 2.4% | 9.7% | 4.7% |
| | attacked | 16.0% | 14.8% | 46.4% | 51.8% | 32.2% |
| | attacked, attackable samples only | 16.5% | 16.2% | 53.7% | 55.1% | 35.4% |

Mean perturbation per observed point (m):

| Model | ade | fde | left | right | front | rear |
|---|---:|---:|---:|---:|---:|---:|
| multimodal_map_gru:123 | 0.19 | 0.19 | 0.18 | 0.18 | 0.19 | 0.18 |
