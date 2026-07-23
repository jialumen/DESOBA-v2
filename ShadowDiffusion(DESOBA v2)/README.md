# ShadowDiffusion (DESOBA v2)

This folder contains the DESOBA v2 evaluation package for ShadowDiffusion.

## Files

- `best_path.pth`: best ShadowDiffusion generator checkpoint.
- `sr.py`: ShadowDiffusion training / validation entrypoint with DESOBA region metrics.
- `metrics.py`: Shadow / non-Shadow / all metric implementation.
- `ShadowDiffusion_best.json`: anonymous best-checkpoint metric record.

## Best checkpoint record

- Validation epoch: 57
- Validation iteration: 40000
- Checkpoint file: `best_path.pth`
- Evaluation uses DESOBA flat shadow masks, 256x256 metric size, and input-output composition for non-shadow regions.

## Recorded DESOBA v2 metrics

| Region | PSNR | SSIM | LPIPS | MAE | RMSE |
|---|---:|---:|---:|---:|---:|
| Shadow | 20.0894 | 0.5386 | 0.1660 | 15.0142 | 12.5383 |
| non-Shadow | 45.5052 | 0.9898 | 0.0146 | 0.5256 | 0.9046 |
| all | 37.8577 | 0.9777 | 0.0205 | 0.8350 | 1.8734 |

The release files intentionally avoid machine names, usernames, ports, absolute local paths, and absolute remote paths.
