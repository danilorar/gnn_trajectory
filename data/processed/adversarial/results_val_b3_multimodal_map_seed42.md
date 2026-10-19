# White-box PGD attack: results

val split (`nuscenes_val_prediction_samples_map.pkl`), 500 random samples; perturbation on all 5 observed target positions, ≤ 1.0 m per point, nuScenes physical bounds from the original code, 3 ground-truth future points included in the physical check; 10 restarts × 100 steps. Each column is attacked with its own objective. Deviations use the min-ADE candidate (issue #18). 83.8% of clean traces satisfy the physical bounds; for the others a perturbation is only accepted if it brings the trace within bounds, so most stay unperturbed. "random, same size" = smooth random perturbations (offset + linear drift) scaled to the attack's mean size per point, with the same feasibility scaling, averaged over 20 draws.

| Model | | minADE6 | minFDE6 | Left | Right | Front | Rear |
|---|---|---:|---:|---:|---:|---:|---:|
| multimodal_map:42 | clean | 1.504 | 3.213 | +0.028 | -0.028 | -0.280 | +0.280 |
|  | random, same size (0.20 m) | 1.529 | 3.224 | +0.036 | -0.036 | -0.255 | +0.255 |
|  | attacked | 2.832 | 6.081 | +1.156 | +1.019 | +1.673 | +2.051 |

Share of samples whose directional deviation exceeds half a lane width (1.85 m):

| Model | | Left | Right | Front | Rear | Mean of 4 |
|---|---|---:|---:|---:|---:|---:|
| multimodal_map:42 | clean | 4.4% | 2.2% | 2.0% | 8.6% | 4.3% |
| | random, same size | 4.7% | 2.7% | 2.2% | 8.5% | 4.5% |
| | attacked | 22.4% | 19.8% | 49.0% | 49.0% | 35.1% |
| | attacked, attackable samples only | 24.1% | 21.5% | 57.3% | 52.0% | 38.7% |

Mean perturbation per observed point (m):

| Model | ade | fde | left | right | front | rear |
|---|---:|---:|---:|---:|---:|---:|
| multimodal_map:42 | 0.19 | 0.20 | 0.18 | 0.19 | 0.19 | 0.19 |
