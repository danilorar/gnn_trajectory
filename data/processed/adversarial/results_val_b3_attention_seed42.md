# White-box PGD attack: results

val split (`nuscenes_val_prediction_samples.pkl`), 500 random samples; perturbation on all 5 observed target positions, ≤ 1.0 m per point, nuScenes physical bounds from the original code, 3 ground-truth future points included in the physical check; 10 restarts × 100 steps. Each column is attacked with its own objective. Deviations use the min-ADE candidate (issue #18). 83.8% of clean traces satisfy the physical bounds; for the others a perturbation is only accepted if it brings the trace within bounds, so most stay unperturbed. "random, same size" = smooth random perturbations (offset + linear drift) scaled to the attack's mean size per point, with the same feasibility scaling, averaged over 20 draws.

| Model | | minADE6 | minFDE6 | Left | Right | Front | Rear |
|---|---|---:|---:|---:|---:|---:|---:|
| attention:42 | clean | 3.366 | 8.169 | +0.065 | -0.065 | -0.853 | +0.853 |
|  | random, same size (0.23 m) | 3.406 | 8.206 | +0.069 | -0.069 | -0.857 | +0.857 |
|  | attacked | 7.258 | 15.976 | +2.061 | +1.823 | +3.439 | +4.871 |

Share of samples whose directional deviation exceeds half a lane width (1.85 m):

| Model | | Left | Right | Front | Rear | Mean of 4 |
|---|---|---:|---:|---:|---:|---:|
| attention:42 | clean | 9.8% | 7.2% | 20.0% | 35.2% | 18.0% |
| | random, same size | 9.4% | 7.2% | 19.6% | 35.3% | 17.9% |
| | attacked | 46.2% | 39.6% | 66.0% | 76.8% | 57.2% |
| | attacked, attackable samples only | 49.4% | 44.4% | 74.2% | 80.9% | 62.2% |

Mean perturbation per observed point (m):

| Model | ade | fde | left | right | front | rear |
|---|---:|---:|---:|---:|---:|---:|
| attention:42 | 0.22 | 0.21 | 0.19 | 0.20 | 0.25 | 0.24 |
