# PhaSR (DESOBA v2)

This folder contains the PhaSR checkpoint and evaluation files for DESOBA v2 strict evaluation.

## Files

- `best_path.pth`: PhaSR checkpoint used for strict DESOBA v2 evaluation.
- `test_desoba_paper_metrics_phasr.py`: strict evaluation entrypoint adapted for PhaSR.
- `region_metrics.py`: region-metric helper used by the evaluation workflow.
- `samples/87_0_pred.png`: single predicted sample image for `87_0`.

## Strict DESOBA v2 metrics

The checkpoint corresponds to the original-mask strict evaluation protocol.

| Region | PSNR | SSIM | MAE | RMSE |
|---|---:|---:|---:|---:|
| Shadow | 15.854 | 0.427 | 8.63029 | 13.23580 |
| non-Shadow | 42.358 | 0.986 | 0.27345 | 0.99769 |
| all | 32.669 | 0.973 | 0.42879 | 2.04257 |

## Notes

The metrics above use the original DESOBA v2 shadow masks without mask dilation or visual-region adjustment.

The checkpoint file contains only `epoch` and `state_dict`.
