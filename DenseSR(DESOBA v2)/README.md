# DenseSR (DESOBA v2)

This folder contains the DESOBA v2 evaluation package for DenseSR.

## Files

- `best_path.pth`: best DenseSR checkpoint, slimmed to model weights plus anonymous metric metadata.
- `DenseSR_best.json`: anonymous best-checkpoint record.
- `region_metrics.py`: DESOBA Shadow / non-Shadow / all metric implementation.
- `region_metrics.csv`: validation metrics recorded every 3 epochs.
- `densesr_options.py`: DenseSR option definitions used by the run.
- `densesr_train_DDP.py`: DenseSR training / validation entrypoint with DESOBA region-metric logging.

## Best checkpoint record

- Validation epoch: 12
- Checkpoint file: `best_path.pth`
- Metric protocol: DESOBA true region pixels, 256x256 metric size.

## Recorded DESOBA v2 metrics

| Region | PSNR | SSIM | LPIPS | MAE | RMSE |
|---|---:|---:|---:|---:|---:|
| Shadow | 24.7404 | 0.7202 | 0.0093 | 6.4383 | 7.3573 |
| non-Shadow | 31.5768 | 0.9687 | 0.0360 | 1.7871 | 4.5768 |
| all | 29.4112 | 0.9573 | 0.0604 | 1.9594 | 4.9136 |

The release files intentionally avoid machine names, usernames, ports, absolute local paths, and absolute remote paths.
