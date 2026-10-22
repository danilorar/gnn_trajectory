# Repaired 30-epoch map/no-map comparison

Only the missing map-aware MLP seed42 artifact was retrained (30 epochs, Adam 0.001, batch 64, six modes, unweighted sampling). All other weights are existing runs. This does not change the final 60-epoch GRU comparison.

All train/validation/test sample identities and non-map inputs align between the existing map and no-map datasets (tolerance 1e-6). All six test ADE values reproduce their saved results; all map FDE values reproduce their manifests. FDE minimizes independently across modes.

| Model | Seed | Test minADE6 (m) | Test minFDE6 (m) |
|---|---:|---:|---:|
| No-map MLP | 42 | 1.703838 | 3.625744 |
| No-map MLP | 123 | 1.717576 | 3.692145 |
| No-map MLP | 2026 | 1.721122 | 3.640284 |
| Map MLP | 42 (replacement) | 1.562118 | 3.170494 |
| Map MLP | 123 | 1.567762 | 3.200677 |
| Map MLP | 2026 | 1.535276 | 3.116310 |

Population standard deviation across seeds (ddof=0) is used for both groups.

| Model | Test minADE6 mean +/- std (m) | Test minFDE6 mean +/- std (m) |
|---|---:|---:|
| no_map | 1.714179 +/- 0.007454 | 3.652724 +/- 0.028499 |
| map | 1.555052 +/- 0.014172 | 3.162494 +/- 0.034904 |

Mean test ADE reduction with map context: 9.28%; mean test FDE reduction: 13.42%.

## Replacement artifact

Checkpoint: `checkpoints_map30_repair/seed42/multimodal_map_seed42_best.pth`. Best epoch: 28; validation minADE6/minFDE6: 1.656556 / 3.473069 m.

Training log, history and input/code hashes are kept under `checkpoints_map30_repair/seed42/` (git-ignored). Numerical output, checkpoint hashes and the map manifests are embedded in the companion JSON.

## Interpretation limits

- Historical map seed42 weights were overwritten. This replacement is newly trained and is not recovery of the old 1.542 result.
- Only map seed42 was retrained; map123/map2026 and all no-map weights are historical checkpoints.
- Historical no-map runs used CPU; map runs used CUDA. This is not a new experiment rerunning both groups under one runtime/device.
- The no-map historical FDE-at-ADE-best field is not minFDE6; the FDE figures here were independently re-evaluated.

The historical 1.542 and its three-seed mean are superseded for this repaired comparison; they are not silently treated as reproducible current results.

## Reproduce

```bash
python scripts/train_model.py --model multimodal_map --seed 42 --epochs 30 --batch-size 64 --lr 0.001 --device cuda --checkpoint-dir /path/to/new_checkpoint_dir
```

Use a new directory: existing weights are protected against overwrite.
