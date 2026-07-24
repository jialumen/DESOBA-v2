# PhaSR (DESOBA v2)

Contents:
- `best_path.pth`: selected PhaSR checkpoint, slimmed to model weights and anonymous metric metadata.
- `test_desoba_v2_metrics_phasr_no_lpips.py`: PhaSR DESOBA v2 evaluator for Shadow, non-Shadow, and All regions.
- `phasr_true_region_metrics_no_lpips.csv`: selected checkpoint metrics without LPIPS.
- `region_metrics.py`: metric and geometry helpers used by the evaluator.

Selected metrics without LPIPS:

| Region | PSNR | SSIM | MAE | RMSE |
|---|---:|---:|---:|---:|
| Shadow | 15.731 | 0.4219 | 8.6639 | 13.3159 |
| non-Shadow | 42.779 | 0.9862 | 0.2632 | 0.9141 |
| All | 32.849 | 0.9732 | 0.4208 | 1.9820 |

The release files intentionally avoid machine names, usernames, ports, absolute local paths, and absolute remote paths.
