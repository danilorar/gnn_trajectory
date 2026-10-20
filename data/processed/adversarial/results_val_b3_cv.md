# White-box PGD attack: results

val split, 500 random samples; perturbation on all 5 observed target positions, ≤ 1.0 m per point, nuScenes physical bounds from the original code, 3 ground-truth future points included in the physical check; 10 restarts × 100 steps. Each column is attacked with its own objective. Deviations use the min-ADE candidate (issue #18). 83.8% of clean traces satisfy the physical bounds; the rest cannot be perturbed (θ = 0).

| Model | | minADE6 | minFDE6 | Left | Right | Front | Rear |
|---|---|---:|---:|---:|---:|---:|---:|
| cv | clean | 4.456 | 10.883 | -0.027 | +0.027 | -1.309 | +1.309 |
| | attacked | 8.003 | 17.190 | +1.704 | +1.725 | +2.565 | +5.111 |

Share of samples whose directional deviation exceeds half a lane width (1.85 m):

| Model | | Left | Right | Front | Rear | Mean of 4 |
|---|---|---:|---:|---:|---:|---:|
| cv | clean | 10.8% | 11.4% | 21.4% | 40.6% | 21.0% |
| | attacked | 40.6% | 33.0% | 63.0% | 81.2% | 54.4% |
| | attacked, attackable samples only | 42.5% | 32.7% | 70.6% | 82.8% | 57.2% |

Mean perturbation per observed point (m):

| Model | ade | fde | left | right | front | rear |
|---|---:|---:|---:|---:|---:|---:|
| cv | 0.17 | 0.16 | 0.17 | 0.17 | 0.21 | 0.20 |
