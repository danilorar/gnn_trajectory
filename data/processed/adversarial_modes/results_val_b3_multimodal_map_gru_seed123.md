# White-box PGD attack: results

val split (`nuscenes_val_prediction_samples_map.pkl`), 500 random samples; perturbation on all 5 observed target positions, ≤ 1.0 m per point, nuScenes physical bounds from the original code, 3 ground-truth future points included in the physical check; 10 restarts × 100 steps. Each column is attacked with its own objective. Deviations use the min-ADE candidate (issue #18). 83.8% of clean traces satisfy the physical bounds; for the others a perturbation is only accepted if it brings the trace within bounds, so most stay unperturbed. "random, same size" = smooth random perturbations (offset + linear drift) scaled to the attack's mean size per point, with the same feasibility scaling, averaged over 20 draws.

| Model | | minADE6 | minFDE6 | Left | Right | Front | Rear |
|---|---|---:|---:|---:|---:|---:|---:|
| multimodal_map_gru:123 | clean | 1.472 | 3.080 | +0.023 | -0.023 | -0.339 | +0.339 |
|  | random, same size (0.20 m) | 1.500 | 3.102 | +0.016 | -0.016 | -0.323 | +0.323 |
|  | attacked | 2.793 | 5.905 | +0.872 | +0.895 | +1.592 | +2.153 |

Share of samples whose directional deviation exceeds half a lane width (1.85 m):

| Model | | Left | Right | Front | Rear | Mean of 4 |
|---|---|---:|---:|---:|---:|---:|
| multimodal_map_gru:123 | clean | 4.2% | 2.4% | 2.2% | 10.0% | 4.7% |
| | random, same size | 4.0% | 2.5% | 2.4% | 9.7% | 4.7% |
| | attacked | 16.0% | 14.8% | 46.4% | 51.8% | 32.2% |
| | attacked, attackable samples only | 16.5% | 16.2% | 53.7% | 55.1% | 35.4% |

Mean perturbation per observed point (m):

| Model | ade | fde | left | right | front | rear |
|---|---:|---:|---:|---:|---:|---:|
| multimodal_map_gru:123 | 0.19 | 0.19 | 0.18 | 0.18 | 0.19 | 0.18 |

Candidates: spread = mean pairwise distance between candidate endpoints; hits = candidates within 2.0 m of the ground truth at every step; miss = share of samples without a hit. Spread ratio and coherence compare with the clean prediction, over samples whose prediction moved; coherence = |mean endpoint shift| / mean |endpoint shift| (1 = all candidates shift together, near 0 = their shifts cancel out); re-centred removes the shift of the current position, which moves every candidate with the frame.

| Model | | Spread (m) | Spread ratio | Hits | Miss rate | Coherence | Coherence, re-centred |
|---|---|---:|---:|---:|---:|---:|---:|
| multimodal_map_gru:123 | clean | 14.54 | – | 0.39 | 63.2% | – | – |
|  | random, same size | 14.57 | 1.00 | 0.39 | 63.5% | 0.88 | 0.85 |
|  | attacked: ade | 15.11 | 1.05 | 0.06 | 95.4% | 0.94 | 0.93 |
|  | attacked: fde | 15.12 | 1.05 | 0.05 | 95.8% | 0.94 | 0.93 |
|  | attacked: left | 15.15 | 1.05 | 0.26 | 77.0% | 0.92 | 0.91 |
|  | attacked: right | 14.95 | 1.03 | 0.25 | 77.8% | 0.92 | 0.90 |
|  | attacked: front | 15.31 | 1.07 | 0.07 | 94.0% | 0.96 | 0.95 |
|  | attacked: rear | 14.41 | 0.99 | 0.11 | 90.8% | 0.96 | 0.96 |
