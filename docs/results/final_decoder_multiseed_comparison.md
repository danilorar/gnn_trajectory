# Final matched decoder comparison

Seeds 42, 123, 2026; 60 epochs; standard unweighted sampling; best validation minADE6 checkpoint. All errors in metres. All standard deviations use population ddof=0, including variability across the three seeds.

## Individual runs

| Decoder | Seed | Best epoch | Val minADE6 | Val minFDE6 | Test minADE6 | Test minFDE6 |
|---|---:|---:|---:|---:|---:|---:|
| MLP | 42 | 56 | 1.585936 | 3.331344 | 1.535461 | 3.109869 |
| MLP | 123 | 51 | 1.573583 | 3.311277 | 1.534421 | 3.137702 |
| MLP | 2026 | 59 | 1.585267 | 3.375772 | 1.527754 | 3.115541 |
| GRU | 42 | 59 | 1.529004 | 3.110682 | 1.491811 | 2.978483 |
| GRU | 123 | 51 | 1.534708 | 3.130822 | 1.477720 | 2.942144 |
| GRU | 2026 | 53 | 1.508728 | 3.045300 | 1.452923 | 2.872186 |

## Overall full test (7,242 samples)

Improvement = MLP minus GRU; positive is better. Percentage = 100 × (MLP mean − GRU mean) / MLP mean.

| Metric | MLP mean ± std | GRU mean ± std | Improvement (m) | Improvement (%) |
|---|---:|---:|---:|---:|
| minADE6 | 1.532545 ± 0.003414 | 1.474151 ± 0.016075 | +0.058394 | +3.81% |
| minFDE6 | 3.121037 ± 0.012009 | 2.930938 ± 0.044113 | +0.190100 | +6.09% |

## Turn bins

Displacement ≥5 m; 7,024 retained samples, 218 excluded. Bins [0,15), [15,30), [30,60), [60,180] degrees. ADE and FDE independently minimize across six modes. Exact per-seed n/mean/median/std and manifests are included in the JSON.

| Angle | n | Metric | MLP mean ± std across seeds | GRU mean ± std across seeds | Improvement (m) | Improvement (%) |
|---|---:|---|---:|---:|---:|---:|
| 0-15 deg | 6022 | minADE6 | 1.331061 ± 0.014252 | 1.302545 ± 0.015865 | +0.028515 | +2.14% |
| 0-15 deg | 6022 | minFDE6 | 2.563682 ± 0.051420 | 2.472263 ± 0.033987 | +0.091418 | +3.57% |
| 15-30 deg | 381 | minADE6 | 2.341333 ± 0.059622 | 2.317672 ± 0.012937 | +0.023660 | +1.01% |
| 15-30 deg | 381 | minFDE6 | 5.602939 ± 0.222361 | 5.584955 ± 0.130598 | +0.017984 | +0.32% |
| 30-60 deg | 534 | minADE6 | 3.087940 ± 0.096900 | 2.729204 ± 0.088790 | +0.358736 | +11.62% |
| 30-60 deg | 534 | minFDE6 | 7.299229 ± 0.299105 | 5.979866 ± 0.276821 | +1.319363 | +18.08% |
| 60+ deg | 87 | minADE6 | 3.820888 ± 0.166212 | 3.577019 ± 0.122230 | +0.243868 | +6.38% |
| 60+ deg | 87 | minFDE6 | 9.037049 ± 0.397774 | 8.350898 ± 0.435039 | +0.686151 | +7.59% |

## Consistency by matched seed

| Population | Metric | Seed 42 improvement (m) | Seed 123 improvement (m) | Seed 2026 improvement (m) |
|---|---|---:|---:|---:|
| Full test | minADE6 | +0.043650 | +0.056701 | +0.074831 |
| Full test | minFDE6 | +0.131386 | +0.195558 | +0.243355 |
| 0-15 deg | minADE6 | +0.013226 | +0.037066 | +0.035254 |
| 0-15 deg | minFDE6 | +0.054051 | +0.135037 | +0.085167 |
| 15-30 deg | minADE6 | +0.034769 | -0.050664 | +0.086876 |
| 15-30 deg | minFDE6 | -0.136629 | -0.225988 | +0.416569 |
| 30-60 deg | minADE6 | +0.326992 | +0.264988 | +0.484228 |
| 30-60 deg | minFDE6 | +1.055115 | +0.961876 | +1.941096 |
| 60+ deg | minADE6 | +0.113587 | +0.421304 | +0.196713 |
| 60+ deg | minFDE6 | +0.503132 | +1.150783 | +0.404537 |

## Interpretation

- Overall full test: minADE6: +3.81% mean improvement; GRU improves in 3/3 matched seeds; minFDE6: +6.09% mean improvement; GRU improves in 3/3 matched seeds.
- 0-15 deg: minADE6: +2.14% mean improvement; GRU improves in 3/3 matched seeds; minFDE6: +3.57% mean improvement; GRU improves in 3/3 matched seeds.
- 15-30 deg: minADE6: +1.01% mean improvement; GRU improves in 2/3 matched seeds; minFDE6: +0.32% mean improvement; GRU improves in 1/3 matched seeds.
- 30-60 deg: minADE6: +11.62% mean improvement; GRU improves in 3/3 matched seeds; minFDE6: +18.08% mean improvement; GRU improves in 3/3 matched seeds.
- 60+ deg: minADE6: +6.38% mean improvement; GRU improves in 3/3 matched seeds; minFDE6: +7.59% mean improvement; GRU improves in 3/3 matched seeds.
- The matched results support selecting the GRU as the final model for this project: overall metrics improve in every seed, sharp-turn bin means improve, and mean straight-trajectory error stays close or improves. This is repeated descriptive evidence, not a statistical significance or causal claim.
- Sharp-turn degradation remains present; these improvements do not resolve the difficulty.

## Integrity and interpretation limits

- All six checkpoints strictly load into the expected original MLP or GRU class; all manifests specify 60 epochs, six modes, radius 35 m, batch size 64 and Adam learning rate 0.001.
- All six validation and full-test evaluations reproduce their manifests. Saved turn-bin evaluations match the same checkpoints and test dataset hashes and passed independent/shared evaluator checks.
- New MLP runs and all GRU runs record identical train/validation/test hashes, unchanged model/graph/training utility hashes and uniform sampling without replacement. These settings were checked at runtime for the new MLP runs.
- The relevant map-model and map-encoder classes are checked against pinned baseline commit 631c7d4e1ad2fcda37d57570989b630541af874a. On a clean checkout, removing the unrelated uncommitted TargetHistoryMLP addition may change the whole utils/models.py hash; that exception is reported and accepted only when the relevant classes still match exactly. Graph/training utility hashes must still match.
- Historical MLP seed-2026 manifest does not record training/validation dataset hashes or sampler details. Its full-test result and prior training-set evaluation match the current files; original source/default training procedure is consistent with unweighted sampling, but missing historical provenance cannot be retroactively verified.
- 63 pre-existing files verified unchanged, including existing baseline/GRU checkpoints, datasets, historical results and unrelated user edits.
- Model architecture, map encoder, graph construction, preprocessing, splits, loss and evaluator were not modified. No weighted sampling is used in this matched comparison.

Endpoint angle is net direction from the origin to the final future point, not path curvature or total heading change. The sharpest bin has 87 examples; seed variability is not a scene-level confidence interval. The decoder changes recurrence, relative displacement prediction, mode conditioning and parameter count together. These tests do not isolate their separate effects. Test-set results have informed experiment development, so this is a development comparison, not an untouched external final benchmark.

## Reproduction

```bash
python scripts/train_gru_decoder.py --decoder mlp --seed 42 --checkpoint-dir checkpoints_60ep
python scripts/train_gru_decoder.py --decoder mlp --seed 123 --checkpoint-dir checkpoints_60ep
# For each new MLP checkpoint, run run_turn_angle_analysis.py with --seed, --checkpoint,
# and --output doc/results/turn_angle_analysis_seed{seed}_60ep.json.
python scripts/compare_decoder_multiseed.py --plot
```

Training commands refuse to overwrite existing checkpoints. Checkpoints and generated datasets remain local; lightweight manifests are embedded in the comparison JSON.
