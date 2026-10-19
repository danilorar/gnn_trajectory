# Diagnostics leading to the final displacement decoder

The scientific snapshot is commit `ca94027d` (before repository cleanup). The canonical final comparison is [the matched report](final_decoder_multiseed_comparison.md) and its JSON/figure, preserved byte-for-byte. Historical executable diagnostics and intermediate plots are recoverable from that commit; uncommitted development artifacts were backed up locally before cleanup. Current reproduction commands are in the README; the frozen report retains its original command history.

All trajectory errors below are metres. Endpoint angle is `abs(degrees(atan2(y_final, x_final)))` in the target-local frame. Angle bins exclude final displacement below 5 m. Test bins contain 6,022 / 381 / 534 / 87 examples; 7,024 retained, 218 excluded. Independent minADE6 and minFDE6 are used except where the historical FDE convention is explicitly noted.

## 1. Sharp-turn error rises strongly

Original map-aware MLP, seed 2026, 30 epochs: mean minADE6 is 1.303255 / 2.371418 / 3.445154 / 4.095009 across 0–15 / 15–30 / 30–60 / 60+ degrees. Corresponding minFDE6 is 2.454517 / 5.842646 / 8.262274 / 9.969656. Thus sharp-turn difficulty was already visible before decoder selection.

## 2. Lane-centerline map context helps, but does not solve turns

Existing 30-epoch map/no-map checkpoints were evaluated on aligned sample identities and identical agent inputs. Full-test results:

| Seed | No-map ADE / FDE | Map ADE / FDE |
|---|---:|---:|
| 123 | 1.717576 / 3.692145 | 1.567762 / 3.200677 |
| 2026 | 1.721122 / 3.640284 | 1.535276 / 3.116311 |

At 30–60°, ADE improves 3.711577 → 3.495997 (seed 123) and 3.612585 → 3.445154 (seed 2026). At 60°+, it improves 4.454700 → 4.046171 and 4.113033 → 4.095009. Map benefit is variable and small on the sharpest turns for seed 2026. Both models struggle there, so the pooled map encoder alone is not isolated as the explanation.

Archived no-map checkpoints were used because canonical checkpoint paths were absent. Their saved report verifies minADE6 and FDE at the ADE-best mode; independent minFDE6 above was recomputed and checked against the shared evaluator, not falsely equated with that historical FDE field.

## 3. Extending training to 60 epochs gives marginal overall gains

Standard map-aware MLP seed 2026:

| Epoch budget | Full-test minADE6 | Full-test minFDE6 |
|---|---:|---:|
| 30 | 1.535276 | 3.116311 |
| 60 (best epoch 59) | 1.527754 | 3.115541 |

Overall improvement is 0.49% ADE and 0.025% FDE. The 30–60° ADE decreases 3.445154 → 3.130648; 60°+ decreases 4.095009 → 3.945208. Longer training helps some turn bins, but substantial errors remain. This does not support undertraining as the main explanation by itself.

## 4. Training-set sharp-turn error is also high

The same standard 60-epoch seed-2026 checkpoint was evaluated without optimization:

| Angle | Train n | Train ADE / FDE | Test ADE / FDE |
|---|---:|---:|---:|
| 0–15° | 20,605 | 1.256787 / 2.377112 | 1.315389 / 2.509393 |
| 15–30° | 1,768 | 1.999079 / 4.835199 | 2.395072 / 5.864170 |
| 30–60° | 2,204 | 2.347001 / 5.439500 | 3.130648 / 7.608039 |
| 60°+ | 334 | 3.387345 / 8.028499 | 3.945208 / 9.365035 |

Retained-train ADE/FDE: 1.434492 / 2.898286 (24,911 samples, 800 excluded). This is mixed fitting/generalization difficulty, not a pure test-only failure. The gap peaks at 30–60° and is not monotonic in angle. It does not isolate capacity from optimization, ambiguity, mode coverage or data imbalance.

Non-stationary train proportions are 82.714463% / 7.097266% / 8.847497% / 1.340773%. Sharp turns are uncommon, but rarity alone does not establish causality.

## 5. Turn-aware sampling helps 30–60°, but not 60°+

One seed-2026, 60-epoch experiment changed only training sampling: replacement draws with weights 1/1/2/4 for the angle bins; displacement <5 m kept weight 1. Each epoch still had 25,711 draws / 402 batches, with unchanged unweighted validation/test evaluation. Best epoch was 29. Validation ADE/FDE was 1.628415 / 3.387648.

| Population | Standard ADE / FDE | Turn-aware ADE / FDE |
|---|---:|---:|
| Full test | 1.527754 / 3.115541 | 1.560115 / 3.161153 |
| 0–15° | 1.315389 / 2.509393 | 1.384869 / 2.714968 |
| 15–30° | 2.395072 / 5.864170 | 2.374505 / 5.323496 |
| 30–60° | 3.130648 / 7.608039 | 2.767417 / 6.103399 |
| 60°+ | 3.945208 / 9.365035 | 4.004106 / 9.671204 |

30–60° improves 11.60% ADE / 19.78% FDE, while overall metrics and 60°+ worsen. Replacement reduces unique examples per epoch (15,848 in the first draw), and no uniform-with-replacement control was run. This supports a contribution from training emphasis in one bin, not imbalance as a universal main cause. Weighted sampling is not used in the final model.

## 6. The GRU displacement decoder gives the strongest controlled evidence

Matched seeds 42/123/2026, standard unweighted sampling, 60 epochs, best validation checkpoint, unchanged map/interaction encoders:

| Decoder | Test minADE6 mean ± population std | Test minFDE6 mean ± population std |
|---|---:|---:|
| One-shot MLP | 1.532545 ± 0.003414 | 3.121037 ± 0.012009 |
| GRU displacement | 1.474151 ± 0.016075 | 2.930938 ± 0.044113 |

Overall gains are 3.81% ADE / 6.09% FDE. At 30–60° they are 11.62% / 18.08%; at 60°+ 6.38% / 7.59%. Overall, straight and both sharp-turn bins improve in every matched seed. The 15–30° mean gain is small and not consistent across seeds. See the canonical comparison for exact per-seed means/medians/std, manifests and all differences.

These results support selecting the GRU for this project; they do not isolate recurrence from relative displacement prediction, mode conditioning or changed decoder parameter count. There are only three seeds, only 87 sharpest-turn test samples, and no scene-level confidence intervals. Endpoint direction is not path curvature. Test results informed development, so this is not an untouched external benchmark. Historical MLP seed-2026 training/sampler provenance remains limited as recorded in the canonical report.

The seed-42 map checkpoint in `checkpoints/` was overwritten by a one-epoch smoke test. Its original turn-angle JSON/figure and qualitative candidates were excluded and removed during freeze; they are not evidence for the finalized model. This does not affect the valid 60-epoch seed-42 MLP/GRU checkpoints.
