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

Candidates: spread = mean pairwise distance between candidate endpoints; hits = candidates within 2.0 m of the ground truth at every step; miss = share of samples without a hit. Spread ratio and coherence compare with the clean prediction, over samples whose prediction moved; coherence = |mean endpoint shift| / mean |endpoint shift| (1 = all candidates shift together, near 0 = their shifts cancel out); re-centred removes the shift of the current position, which moves every candidate with the frame.

| Model | | Spread (m) | Spread ratio | Hits | Miss rate | Coherence | Coherence, re-centred |
|---|---|---:|---:|---:|---:|---:|---:|
| multimodal_map_gru:2026 | clean | 14.49 | – | 0.37 | 63.8% | – | – |
|  | random, same size | 14.53 | 1.00 | 0.37 | 63.6% | 0.88 | 0.86 |
|  | attacked: ade | 15.07 | 1.05 | 0.04 | 95.8% | 0.92 | 0.91 |
|  | attacked: fde | 14.94 | 1.04 | 0.03 | 97.2% | 0.93 | 0.93 |
|  | attacked: left | 14.98 | 1.05 | 0.23 | 77.4% | 0.92 | 0.91 |
|  | attacked: right | 14.74 | 1.03 | 0.26 | 74.8% | 0.92 | 0.91 |
|  | attacked: front | 14.97 | 1.04 | 0.07 | 93.2% | 0.94 | 0.94 |
|  | attacked: rear | 14.45 | 1.00 | 0.12 | 88.8% | 0.96 | 0.95 |
