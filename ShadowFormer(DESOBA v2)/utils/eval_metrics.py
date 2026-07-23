import os

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import cv2
import numpy as np
import torch
from skimage.metrics import peak_signal_noise_ratio as psnr_loss
from skimage.metrics import structural_similarity as ssim_loss


METRIC_KEYS = (
    "psnr_all", "ssim_all", "mae_all", "rmse_all", "lpips_all",
    "psnr_shadow", "ssim_shadow", "mae_shadow", "rmse_shadow", "lpips_shadow",
    "psnr_nonshadow", "ssim_nonshadow", "mae_nonshadow", "rmse_nonshadow", "lpips_nonshadow",
)

_LPIPS_MODEL = None
_LPIPS_MODULE = None
_LPIPS_WARNING_PRINTED = False


def _resize_rgb(image, size=256):
    image = cv2.resize(image * 255.0, [size, size], interpolation=cv2.INTER_AREA) / 255.0
    return np.clip(image, 0, 1).astype(np.float32)


def _resize_mask(mask, size=256):
    if mask.ndim == 3:
        mask = mask.squeeze()
    mask = cv2.resize(mask * 255.0, [size, size], interpolation=cv2.INTER_AREA) / 255.0
    mask = np.where(mask < 0.001, np.zeros_like(mask), np.ones_like(mask))
    return mask.astype(np.float32)


def _masked_psnr(pred, gt, mask):
    if mask.sum() <= 0:
        return np.nan
    region = mask > 0.5
    mse = np.mean(np.square(pred[region] - gt[region]))
    if mse <= 0:
        return float("inf")
    return 20.0 * np.log10(1.0 / np.sqrt(mse))


def _masked_ssim(pred, gt, mask):
    if mask.sum() <= 0:
        return np.nan
    region = mask > 0.5
    _, ssim_map = ssim_loss(pred, gt, channel_axis=-1, data_range=1.0, full=True)
    if ssim_map.ndim == 3:
        return float(np.mean(ssim_map[region, :]))
    return float(np.mean(ssim_map[region]))


def _lab_mae(pred, gt, mask=None):
    lab_pred = cv2.cvtColor(pred, cv2.COLOR_RGB2LAB)
    lab_gt = cv2.cvtColor(gt, cv2.COLOR_RGB2LAB)
    if mask is None:
        return np.abs(lab_pred - lab_gt).mean() * 3

    if mask.sum() <= 0:
        return np.nan
    region = mask > 0.5
    return np.abs(lab_pred[region] - lab_gt[region]).mean() * 3


def _lab_rmse(pred, gt, mask=None):
    lab_pred = cv2.cvtColor(pred, cv2.COLOR_RGB2LAB)
    lab_gt = cv2.cvtColor(gt, cv2.COLOR_RGB2LAB)
    if mask is None:
        return np.sqrt(np.square(lab_pred - lab_gt).sum() / (pred.shape[0] * pred.shape[1]))

    if mask.sum() <= 0:
        return np.nan
    region = mask > 0.5
    return np.sqrt(np.square(lab_pred[region] - lab_gt[region]).sum() / region.sum())


def _get_lpips_model():
    global _LPIPS_MODEL, _LPIPS_MODULE, _LPIPS_WARNING_PRINTED
    if _LPIPS_MODULE is None:
        try:
            import lpips as lpips_module
            _LPIPS_MODULE = lpips_module
        except Exception as exc:
            if not _LPIPS_WARNING_PRINTED:
                print("Warning: lpips unavailable ({}); LPIPS will be NaN.".format(exc))
                _LPIPS_WARNING_PRINTED = True
            return None
    if _LPIPS_MODEL is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        _LPIPS_MODEL = _LPIPS_MODULE.LPIPS(net="alex").to(device).eval()
    return _LPIPS_MODEL


def _lpips_region(pred, gt, mask=None):
    if mask is not None and mask.sum() <= 0:
        return np.nan
    model = _get_lpips_model()
    if model is None:
        return np.nan
    if mask is not None:
        region = mask > 0.5
        rows, cols = np.where(region)
        r0, r1 = int(rows.min()), int(rows.max()) + 1
        c0, c1 = int(cols.min()), int(cols.max()) + 1
        min_size = 64
        height, width = region.shape
        if r1 - r0 < min_size:
            pad = min_size - (r1 - r0)
            r0 = max(0, r0 - pad // 2)
            r1 = min(height, r1 + pad - pad // 2)
        if c1 - c0 < min_size:
            pad = min_size - (c1 - c0)
            c0 = max(0, c0 - pad // 2)
            c1 = min(width, c1 + pad - pad // 2)
        pred = pred[r0:r1, c0:c1]
        gt = gt[r0:r1, c0:c1]
        mask3 = region[r0:r1, c0:c1, None].astype(np.float32)
        pred = pred * mask3
        gt = gt * mask3
    if min(pred.shape[:2]) < 64:
        new_h = max(64, pred.shape[0])
        new_w = max(64, pred.shape[1])
        pred = cv2.resize(pred, (new_w, new_h), interpolation=cv2.INTER_AREA)
        gt = cv2.resize(gt, (new_w, new_h), interpolation=cv2.INTER_AREA)
    pred_t = torch.from_numpy(pred.transpose(2, 0, 1)).unsqueeze(0).float() * 2.0 - 1.0
    gt_t = torch.from_numpy(gt.transpose(2, 0, 1)).unsqueeze(0).float() * 2.0 - 1.0
    device = next(model.parameters()).device
    with torch.no_grad():
        value = model(pred_t.to(device), gt_t.to(device))
    return float(value.detach().cpu().reshape(-1)[0].item())


def compute_region_metrics(pred_rgb, gt_rgb, mask, size=256):
    pred = _resize_rgb(pred_rgb.astype(np.float32), size=size)
    gt = _resize_rgb(gt_rgb.astype(np.float32), size=size)
    shadow = _resize_mask(mask.astype(np.float32), size=size)
    nonshadow = 1.0 - shadow

    return {
        "psnr_all": psnr_loss(pred, gt, data_range=1.0),
        "ssim_all": ssim_loss(pred, gt, channel_axis=-1, data_range=1.0),
        "mae_all": _lab_mae(pred, gt),
        "rmse_all": _lab_rmse(pred, gt),
        "lpips_all": _lpips_region(pred, gt),
        "psnr_shadow": _masked_psnr(pred, gt, shadow),
        "ssim_shadow": _masked_ssim(pred, gt, shadow),
        "mae_shadow": _lab_mae(pred, gt, shadow),
        "rmse_shadow": _lab_rmse(pred, gt, shadow),
        "lpips_shadow": _lpips_region(pred, gt, shadow),
        "psnr_nonshadow": _masked_psnr(pred, gt, nonshadow),
        "ssim_nonshadow": _masked_ssim(pred, gt, nonshadow),
        "mae_nonshadow": _lab_mae(pred, gt, nonshadow),
        "rmse_nonshadow": _lab_rmse(pred, gt, nonshadow),
        "lpips_nonshadow": _lpips_region(pred, gt, nonshadow),
    }


def merge_metric_lists(dst, src):
    for key in METRIC_KEYS:
        dst.setdefault(key, []).append(src[key])


def mean_metric_lists(metrics):
    out = {}
    for key in METRIC_KEYS:
        values = np.asarray(metrics.get(key, []), dtype=np.float64)
        values = values[np.isfinite(values)]
        out[key] = float(values.mean()) if values.size else float("nan")
    return out


def format_metric_table(method_name, metrics):
    lines = [
        "Metric table (256x256; RGB PSNR/SSIM; LAB MAE/RMSE; LPIPS):",
        "{:<24} | {:^45} | {:^49} | {:^45}".format(
            "Method", "Shadow Region", "Non-Shadow Region", "All Region"),
        "{:<24} | {:>8} {:>8} {:>8} {:>8} {:>8} | {:>8} {:>8} {:>8} {:>8} {:>8} | {:>8} {:>8} {:>8} {:>8} {:>8}".format(
            "", "PSNR", "SSIM", "MAE", "RMSE", "LPIPS",
            "PSNR", "SSIM", "MAE", "RMSE", "LPIPS",
            "PSNR", "SSIM", "MAE", "RMSE", "LPIPS"),
        "-" * 177,
        "{:<24} | {:8.2f} {:8.3f} {:8.2f} {:8.2f} {:8.4f} | {:8.2f} {:8.3f} {:8.2f} {:8.2f} {:8.4f} | {:8.2f} {:8.3f} {:8.2f} {:8.2f} {:8.4f}".format(
            method_name,
            metrics["psnr_shadow"], metrics["ssim_shadow"], metrics["mae_shadow"],
            metrics["rmse_shadow"], metrics["lpips_shadow"],
            metrics["psnr_nonshadow"], metrics["ssim_nonshadow"], metrics["mae_nonshadow"],
            metrics["rmse_nonshadow"], metrics["lpips_nonshadow"],
            metrics["psnr_all"], metrics["ssim_all"], metrics["mae_all"],
            metrics["rmse_all"], metrics["lpips_all"]),
    ]
    return "\n".join(lines)
