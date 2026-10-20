# White-box PGD attack: results

val split, 500 random samples; perturbation on all 5 observed target positions, ≤ 1.0 m per point, nuScenes physical bounds from the original code, 3 ground-truth future points included in the physical check; 10 restarts × 100 steps. Each column is attacked with its own objective. Deviations use the min-ADE candidate (issue #18). 83.8% of clean traces satisfy the physical bounds; the rest cannot be perturbed (θ = 0).

| Model | | minADE6 | minFDE6 | Left | Right | Front | Rear |
|---|---|---:|---:|---:|---:|---:|---:|
| multimodal:42 | clean | 1.719 | 3.851 | +0.118 | -0.118 | -0.301 | +0.301 |
| | attacked | 3.526 | 7.680 | +1.887 | +1.356 | +1.962 | +2.436 |

Share of samples whose directional deviation exceeds half a lane width (1.85 m):

| Model | | Left | Right | Front | Rear | Mean of 4 |
|---|---|---:|---:|---:|---:|---:|
| multimodal:42 | clean | 7.6% | 4.8% | 3.2% | 10.6% | 6.5% |
| | attacked | 49.0% | 35.8% | 70.4% | 71.6% | 56.7% |
| | attacked, attackable samples only | 53.0% | 40.8% | 81.1% | 77.3% | 63.1% |

Mean perturbation per observed point (m):

| Model | ade | fde | left | right | front | rear |
|---|---:|---:|---:|---:|---:|---:|
| multimodal:42 | 0.19 | 0.19 | 0.18 | 0.16 | 0.20 | 0.18 |
