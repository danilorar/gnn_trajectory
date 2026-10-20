# White-box PGD attack: results

val split (`nuscenes_val_prediction_samples_map.pkl`), 500 random samples; perturbation on all 5 observed target positions, ≤ 1.0 m per point, nuScenes physical bounds from the original code, 3 ground-truth future points included in the physical check; 10 restarts × 100 steps. Each column is attacked with its own objective. Deviations use the min-ADE candidate (issue #18). 83.8% of clean traces satisfy the physical bounds; for the others a perturbation is only accepted if it brings the trace within bounds, so most stay unperturbed. "random, same size" = smooth random perturbations (offset + linear drift) scaled to the attack's mean size per point, with the same feasibility scaling, averaged over 20 draws.

| Model | | minADE6 | minFDE6 | Left | Right | Front | Rear |
|---|---|---:|---:|---:|---:|---:|---:|
| multimodal_map_gru:2026 | clean | 1.460 | 2.986 | +0.061 | -0.061 | -0.247 | +0.247 |
|  | random, same size (0.20 m) | 1.479 | 2.994 | +0.060 | -0.060 | -0.241 | +0.241 |
|  | attacked | 2.816 | 6.052 | +1.062 | +0.911 | +1.690 | +2.088 |

Share of samples whose directional deviation exceeds half a lane width (1.85 m):

| Model | | Left | Right | Front | Rear | Mean of 4 |
|---|---|---:|---:|---:|---:|---:|
| multimodal_map_gru:2026 | clean | 3.8% | 1.4% | 3.2% | 7.6% | 4.0% |
| | random, same size | 3.9% | 1.8% | 2.7% | 7.9% | 4.1% |
| | attacked | 20.0% | 15.2% | 46.6% | 51.0% | 33.2% |
| | attacked, attackable samples only | 22.0% | 16.9% | 53.5% | 56.3% | 37.2% |

Mean perturbation per observed point (m):

| Model | ade | fde | left | right | front | rear |
|---|---:|---:|---:|---:|---:|---:|
| multimodal_map_gru:2026 | 0.19 | 0.19 | 0.18 | 0.17 | 0.18 | 0.19 |
