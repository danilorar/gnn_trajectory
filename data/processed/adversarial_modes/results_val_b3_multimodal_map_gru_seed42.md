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

Candidates: spread = mean pairwise distance between candidate endpoints; hits = candidates within 2.0 m of the ground truth at every step; miss = share of samples without a hit. Spread ratio and coherence compare with the clean prediction, over samples whose prediction moved; coherence = |mean endpoint shift| / mean |endpoint shift| (1 = all candidates shift together, near 0 = their shifts cancel out); re-centred removes the shift of the current position, which moves every candidate with the frame.

| Model | | Spread (m) | Spread ratio | Hits | Miss rate | Coherence | Coherence, re-centred |
|---|---|---:|---:|---:|---:|---:|---:|
| multimodal_map_gru:42 | clean | 15.19 | – | 0.40 | 61.8% | – | – |
|  | random, same size | 15.26 | 1.01 | 0.38 | 63.7% | 0.87 | 0.85 |
|  | attacked: ade | 15.63 | 1.04 | 0.03 | 96.8% | 0.90 | 0.89 |
|  | attacked: fde | 15.79 | 1.05 | 0.04 | 97.2% | 0.88 | 0.88 |
|  | attacked: left | 16.06 | 1.08 | 0.21 | 79.8% | 0.88 | 0.87 |
|  | attacked: right | 15.36 | 1.02 | 0.19 | 82.4% | 0.88 | 0.87 |
|  | attacked: front | 16.18 | 1.09 | 0.05 | 94.6% | 0.92 | 0.92 |
|  | attacked: rear | 15.03 | 1.00 | 0.07 | 93.2% | 0.93 | 0.92 |
