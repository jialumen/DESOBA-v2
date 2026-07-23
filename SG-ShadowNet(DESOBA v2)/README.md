# SG-ShadowNet (DESOBA v2)

This folder contains the DESOBA v2 strict-evaluation package for SG-ShadowNet.

## Files

- `best_path.pth`: best strict checkpoint, saved with model weights and evaluation metadata only.
- `test_desoba_v2_latest_metrics.py`: DESOBA v2 strict evaluation entrypoint.
- `data/desoba_v2.py`: DESOBA v2 dataset loader used by the evaluation script.
- `models/model.py`: SG-ShadowNet model definition used by the checkpoint.
- `utils/eval_metrics.py`: strict Shadow / non-Shadow / all metric implementation.
- `metrics_every_3_epochs.csv`: training-time validation metrics recorded every 3 epochs.

## Strict DESOBA v2 metrics

| Region | PSNR | SSIM | LPIPS | MAE | RMSE |
|---|---:|---:|---:|---:|---:|
| Shadow | 25.3470 | 0.7056 | 0.0329 | 7.3444 | 7.2195 |
| non-Shadow | 41.2594 | 0.9877 | 0.0041 | 0.9309 | 1.1470 |
| all | 37.0664 | 0.9745 | 0.0222 | 1.2031 | 1.8534 |

## Evaluation

Run from this folder after preparing DESOBA v2 annotations and images:

```bash
python test_desoba_v2_latest_metrics.py \
  --weights best_path.pth \
  --annotation_root ./data/DESOBAv2_extended_annotations \
  --image_root ./data/DESOBAv2_work \
  --split test
```

The checkpoint is named `best_path.pth` intentionally for anonymous release consistency.
