# OmniSR (DESOBA v2)

This folder contains the OmniSR checkpoint and evaluation script for DESOBA v2 strict evaluation.

## Files

- `best_path.pth`: OmniSR checkpoint used for the reported strict DESOBA v2 metrics.
- `tools/test_desoba_v2_latest_metrics_omnisr.py`: evaluation entrypoint adapted for OmniSR.
- `utils/eval_metrics.py`: metric implementation used by the evaluation script.
- `samples/87_0_pred.jpg`: single predicted sample image for `87_0`.

## Strict DESOBA v2 metrics

The reported checkpoint corresponds to the strict/original-mask evaluation protocol.

| Region | PSNR | SSIM | LPIPS | MAE | RMSE |
|---|---:|---:|---:|---:|---:|
| Shadow | 15.9311 | 0.4311 | 0.0929 | 25.6128 | 22.7936 |
| non-Shadow | 42.5400 | 0.9860 | 0.0083 | 0.8716 | 1.6679 |
| all | 32.8990 | 0.9734 | 0.0349 | 1.3236 | 3.4576 |

## Notes

The metrics above use the original DESOBA v2 shadow masks without mask dilation or visual-region adjustment.

The checkpoint file contains only `epoch` and `state_dict`.
