# ShadowFormer (DESOBA v2)

This folder contains the DESOBA v2 release package for ShadowFormer.

## Files

- `best_path.pth`: best ShadowFormer checkpoint, pinned at epoch 30 and slimmed to model weights plus anonymous metric metadata.
- `test_desoba_v2_latest_metrics.py`: DESOBA v2 evaluation entrypoint.
- `model.py`, `dataset.py`, `options.py`, `utils/`: minimal ShadowFormer code needed by the evaluation script.
- `metrics_summary.csv`: full 750-image DESOBA v2 test metrics for the best checkpoint.
- `per_sample_metrics_best_epoch30.csv`: local two-sample verification metrics for samples `535_0` and `87_0`.

## Best checkpoint record

- Checkpoint file: `best_path.pth`
- Selected checkpoint: epoch 30
- Evaluation setting: `win_size=8`
- Metric protocol: 256x256 RGB PSNR/SSIM, LAB MAE/RMSE, LPIPS.

## Recorded DESOBA v2 metrics

| Region | PSNR | SSIM | LPIPS | MAE | RMSE |
|---|---:|---:|---:|---:|---:|
| Shadow | 25.65 | 0.735 | 0.0306 | 7.36 | 7.24 |
| non-Shadow | 48.20 | 0.994 | 0.0014 | 0.70 | 0.81 |
| all | 40.35 | 0.983 | 0.0174 | 0.94 | 1.54 |

## Evaluation

Run from this folder after preparing the DESOBA v2 test split in the expected ShadowFormer layout:

```bash
python test_desoba_v2_latest_metrics.py \
  --weights best_path.pth \
  --input_dir ./data/DESOBAv2_test/ \
  --win_size 8 \
  --cal_metrics
```

The release files intentionally avoid machine names, usernames, ports, absolute local paths, and absolute remote paths.
