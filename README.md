# Multimodal Trajectory Prediction with a GRU Decoder

A map-aware GATv2–GRU model that predicts six possible vehicle trajectories, using 2 seconds of motion history to forecast 6 seconds ahead on the [nuScenes dataset](https://www.nuscenes.org/nuscenes). It combines agent motion histories, one-hot category encodings, and nearby lane geometry. GATv2 models agent interactions, while a GRU predicts future trajectories step by step.

<p align="center">
  <img src="assets/right_turn.gif" width="960" alt="A nuScenes right-turn example showing nearby agent types, observed motion, ground truth and six GRU trajectory predictions">
</p>

*Right-turn prediction on nuScenes. Red highlights the mode closest to ground truth by ADE.*

[Read the report](report/Trajectory_prediction_Report.pdf)

## What we found

Lane context gives the clearest gain over the model without maps. Replacing the one-shot MLP decoder with a GRU improves predictions further, especially on turns.

The matched decoder comparison uses seeds 42, 123 and 2026, 60 epochs, and standard unweighted sampling. Values are full-test means ± population standard deviations across seeds; lower is better.

| Map-aware decoder | minADE₆ [m] | minFDE₆ [m] |
|---|---:|---:|
| One-shot MLP | 1.533 ± 0.003 | 3.121 ± 0.012 |
| GRU displacement | 1.474 ± 0.016 | 2.931 ± 0.044 |

The GRU reduces overall minADE₆ by 3.8% and minFDE₆ by 6.1%. For 30–60° endpoint-angle trajectories, minFDE₆ improves by 18.1%. Sharp turns remain difficult, and removing neighbors has little effect on the final model. The report covers these diagnostics and limitations.

This project was developed as a final project for the SSY340 Deep Machine Learning course at Chalmers.
