# White-box PGD attack: results

val split, 500 random samples; perturbation on all 5 observed target positions, ≤ 1.0 m per point, nuScenes physical bounds from the original code, 3 ground-truth future points included in the physical check; 10 restarts × 100 steps. Each column is attacked with its own objective. Deviations use the min-ADE candidate (issue #18). 83.8% of clean traces satisfy the physical bounds; the rest cannot be perturbed (θ = 0).

| Model | | minADE6 | minFDE6 | Left | Right | Front | Rear |
|---|---|---:|---:|---:|---:|---:|---:|
| multimodal:123 | clean | 1.704 | 3.866 | +0.071 | -0.071 | -0.357 | +0.357 |
| | attacked | 3.463 | 7.790 | +1.666 | +1.449 | +2.019 | +2.567 |

Share of samples whose directional deviation exceeds half a lane width (1.85 m):

| Model | | Left | Right | Front | Rear | Mean of 4 |
|---|---|---:|---:|---:|---:|---:|
| multimodal:123 | clean | 7.0% | 5.6% | 4.8% | 12.2% | 7.4% |
| | attacked | 39.4% | 34.6% | 66.6% | 71.2% | 53.0% |
| | attacked, attackable samples only | 42.5% | 38.9% | 77.3% | 77.8% | 59.1% |

Mean perturbation per observed point (m):

| Model | ade | fde | left | right | front | rear |
|---|---:|---:|---:|---:|---:|---:|
| multimodal:123 | 0.19 | 0.19 | 0.17 | 0.17 | 0.18 | 0.18 |
