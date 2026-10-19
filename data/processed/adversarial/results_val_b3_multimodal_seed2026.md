# White-box PGD attack: results

val split, 500 random samples; perturbation on all 5 observed target positions, ≤ 1.0 m per point, nuScenes physical bounds from the original code, 3 ground-truth future points included in the physical check; 10 restarts × 100 steps. Each column is attacked with its own objective. Deviations use the min-ADE candidate (issue #18). 83.8% of clean traces satisfy the physical bounds; the rest cannot be perturbed (θ = 0).

| Model | | minADE6 | minFDE6 | Left | Right | Front | Rear |
|---|---|---:|---:|---:|---:|---:|---:|
| multimodal:2026 | clean | 1.718 | 3.891 | +0.067 | -0.067 | -0.502 | +0.502 |
| | attacked | 3.392 | 7.588 | +1.560 | +1.366 | +1.772 | +2.764 |

Share of samples whose directional deviation exceeds half a lane width (1.85 m):

| Model | | Left | Right | Front | Rear | Mean of 4 |
|---|---|---:|---:|---:|---:|---:|
| multimodal:2026 | clean | 8.0% | 5.8% | 2.4% | 13.0% | 7.3% |
| | attacked | 38.6% | 30.4% | 57.2% | 70.0% | 49.1% |
| | attacked, attackable samples only | 40.8% | 33.2% | 66.1% | 75.4% | 53.9% |

Mean perturbation per observed point (m):

| Model | ade | fde | left | right | front | rear |
|---|---:|---:|---:|---:|---:|---:|
| multimodal:2026 | 0.20 | 0.20 | 0.16 | 0.17 | 0.19 | 0.19 |
