# White-box PGD attack: results

val split (`nuscenes_val_prediction_samples.pkl`), 500 random samples; perturbation on all 5 observed target positions, ≤ 1.0 m per point, nuScenes physical bounds from the original code, 3 ground-truth future points included in the physical check; 10 restarts × 100 steps. Each column is attacked with its own objective. Deviations use the min-ADE candidate (issue #18). 83.8% of clean traces satisfy the physical bounds; for the others a perturbation is only accepted if it brings the trace within bounds, so most stay unperturbed. "random, same size" = smooth random perturbations (offset + linear drift) scaled to the attack's mean size per point, with the same feasibility scaling, averaged over 20 draws.

| Model | | minADE6 | minFDE6 | Left | Right | Front | Rear |
|---|---|---:|---:|---:|---:|---:|---:|
| attention:123 | clean | 3.609 | 8.859 | +0.286 | -0.286 | -1.050 | +1.050 |
|  | random, same size (0.23 m) | 3.658 | 8.911 | +0.300 | -0.300 | -1.056 | +1.056 |
|  | attacked | 8.528 | 18.596 | +2.823 | +1.912 | +4.377 | +5.927 |

Share of samples whose directional deviation exceeds half a lane width (1.85 m):

| Model | | Left | Right | Front | Rear | Mean of 4 |
|---|---|---:|---:|---:|---:|---:|
| attention:123 | clean | 12.0% | 8.4% | 19.4% | 38.4% | 19.6% |
| | random, same size | 12.3% | 8.6% | 19.7% | 38.9% | 19.9% |
| | attacked | 67.6% | 42.8% | 73.2% | 85.2% | 67.2% |
| | attacked, attackable samples only | 73.5% | 46.1% | 83.1% | 89.3% | 73.0% |

Mean perturbation per observed point (m):

| Model | ade | fde | left | right | front | rear |
|---|---:|---:|---:|---:|---:|---:|
| attention:123 | 0.21 | 0.20 | 0.19 | 0.19 | 0.24 | 0.24 |
