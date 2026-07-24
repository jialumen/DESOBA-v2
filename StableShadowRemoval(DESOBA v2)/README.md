# StableShadowRemoval (DESOBA v2)

This folder contains the DESOBA v2 release package for StableShadowRemoval.

## Files

- `checkpoint-41736/unet/config.json`: anonymized Diffusers UNet config.
- `checkpoint-41736/unet/diffusion_pytorch_model.safetensors.part01` and `.part02`: split best UNet checkpoint weights.
- `test_files/test_desoba_v2_latest_metrics_stableshadowremoval_dir_eval.py`: directory-based DESOBA v2 Shadow / non-Shadow / all evaluator for StableShadowRemoval outputs.
- `test_files/test_desoba_v2_latest_metrics_original_homoformer.py`: reference HomoFormer/SRD metric protocol file included for protocol comparison.
- `metrics/desoba_v15_checkpoint41736_strict_testfile_full_metrics.csv`: full strict DESOBA v2 metrics for checkpoint 41736.
- `StableShadowRemoval_best.json`: anonymous best-checkpoint metric record.

## Best checkpoint record

- Checkpoint: `checkpoint-41736`
- Weight files: `checkpoint-41736/unet/diffusion_pytorch_model.safetensors.part01` and `.part02`
- Reconstructed weight SHA256: `8c4f4b12b0247176533a2db54fd0a3738b01e9917e9500a63497f2fb719f0a08`
- Metric protocol: strict DESOBA v2 Shadow / non-Shadow / all evaluation, 256x256 metrics.

## Reconstruct weights

Reconstruct the Diffusers UNet weight file before loading the checkpoint:

```bash
cat checkpoint-41736/unet/diffusion_pytorch_model.safetensors.part01 \
    checkpoint-41736/unet/diffusion_pytorch_model.safetensors.part02 \
    > checkpoint-41736/unet/diffusion_pytorch_model.safetensors
```

On Windows PowerShell:

```powershell
Get-Content checkpoint-41736/unet/diffusion_pytorch_model.safetensors.part01,
            checkpoint-41736/unet/diffusion_pytorch_model.safetensors.part02 `
  -Encoding Byte -ReadCount 0 |
  Set-Content checkpoint-41736/unet/diffusion_pytorch_model.safetensors -Encoding Byte
```

## Recorded DESOBA v2 metrics

| Region | PSNR | SSIM | LPIPS | MAE | RMSE |
|---|---:|---:|---:|---:|---:|
| Shadow | 17.0699 | 0.4438 | 0.2441 | 18.9147 | 18.8452 |
| non-Shadow | 24.2221 | 0.6875 | 0.0999 | 7.5199 | 7.7683 |
| all | 23.3908 | 0.6794 | 0.1063 | 7.8886 | 8.4954 |

The release files intentionally avoid machine names, usernames, ports, absolute local paths, and absolute remote paths.
