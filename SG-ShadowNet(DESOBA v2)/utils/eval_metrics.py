import os

# Windows/Conda often loads OpenMP from both cv2/skimage and torch/LPIPS.
# Set this before those libraries import so metric evaluation does not abort.
os.environ.setdefault('KMP_DUPLICATE_LIB_OK', 'TRUE')

import cv2
import numpy as np
import torch
from skimage.metrics import peak_signal_noise_ratio as psnr_loss
from skimage.metrics import structural_similarity as ssim_loss


METRIC_KEYS = (
    'psnr_all', 'ssim_all', 'mae_all', 'rmse_all', 'lpips_all',
    'psnr_shadow', 'ssim_shadow', 'mae_shadow', 'rmse_shadow', 'lpips_shadow',
    'psnr_nonshadow', 'ssim_nonshadow', 'mae_nonshadow', 'rmse_nonshadow', 'lpips_nonshadow',
)

_LPIPS_MODEL = None
_LPIPS_MODULE = None
_LPIPS_WARNING_PRINTED = False


def _resize_rgb(image, size=256):
    image = cv2.resize(image * 255.0, [size, size], interpolation=cv2.INTER_AREA) / 255.0
    return np.clip(image, 0, 1)


def _resize_mask(mask, size=256):
    if mask.ndim == 3:
        mask = mask.squeeze()
    mask = cv2.resize(mask * 255.0, [size, size], interpolation=cv2.INTER_AREA) / 255.0
    mask = np.where(mask < 0.001, np.zeros_like(mask), np.ones_like(mask))
    return mask.astype(np.float32)


def _region_weights(mask):
    shadow_pixels = float(mask.sum())
    total_pixels = float(mask.size)
    shadow_weight = shadow_pixels / max(total_pixels, 1.0)
    return shadow_weight, 1.0 - shadow_weight


def _safe_weighted_mean(a, b, wa, wb):
    if np.isfinite(a) and np.isfinite(b):
        return float(wa * a + wb * b)
    if np.isfinite(a):
        return float(a)
    if np.isfinite(b):
        return float(b)
    return float('nan')


def _psnr_from_mse(mse):
    if not np.isfinite(mse):
        return float('nan')
    if mse <= 0:
        return float('inf')
    return float(10.0 * np.log10(1.0 / mse))


def _masked_mse(pred, gt, mask):
    if mask.sum() <= 0:
        return np.nan
    mask3 = mask[:, :, None].astype(np.float32)
    mse = np.square(pred - gt) * mask3
    denom = mask3.sum() * pred.shape[2]
    return float(mse.sum() / max(denom, 1.0))


def _masked_psnr(pred, gt, mask):
    return _psnr_from_mse(_masked_mse(pred, gt, mask))


def _ssim_region_metrics(pred, gt, shadow, nonshadow):
    _, ssim_map = ssim_loss(
        pred, gt, channel_axis=-1, data_range=1.0, full=True)
    if ssim_map.ndim == 3:
        ssim_map = ssim_map.mean(axis=-1)
    ssim_map = np.asarray(ssim_map, dtype=np.float64)

    def region_mean(region):
        if region.sum() <= 0:
            return float('nan')
        return float((ssim_map * region).sum() / max(float(region.sum()), 1.0))

    shadow_value = region_mean(shadow)
    nonshadow_value = region_mean(nonshadow)
    return {
        'ssim_shadow': shadow_value,
        'ssim_nonshadow': nonshadow_value,
        'ssim_all': float(ssim_loss(pred, gt, channel_axis=-1, data_range=1.0)),
    }


def _masked_ssim(pred, gt, mask):
    if mask.sum() <= 0:
        return np.nan
    if pred.ndim == 3:
        mask = mask[:, :, None]
        return ssim_loss(pred * mask, gt * mask, channel_axis=-1, data_range=1.0)
    return ssim_loss(pred * mask, gt * mask, channel_axis=None, data_range=1.0)


def _lab_error_maps(pred, gt):
    lab_pred = cv2.cvtColor(pred, cv2.COLOR_RGB2LAB)
    lab_gt = cv2.cvtColor(gt, cv2.COLOR_RGB2LAB)
    diff = lab_pred - lab_gt
    abs_sum = np.abs(diff).sum(axis=2)
    sq_sum = np.square(diff).sum(axis=2)
    return abs_sum, sq_sum


def _lab_mae_from_map(abs_sum, mask=None):
    if mask is None:
        return float(abs_sum.mean())
    if mask.sum() <= 0:
        return np.nan
    return float((abs_sum * mask).sum() / max(float(mask.sum()), 1.0))


def _lab_rmse_from_map(sq_sum, mask=None):
    if mask is None:
        return float(np.sqrt(sq_sum.mean()))
    if mask.sum() <= 0:
        return np.nan
    return float(np.sqrt((sq_sum * mask).sum() / max(float(mask.sum()), 1.0)))


def _lab_mae(pred, gt, mask=None):
    abs_sum, _ = _lab_error_maps(pred, gt)
    return _lab_mae_from_map(abs_sum, mask)


def _lab_rmse(pred, gt, mask=None):
    _, sq_sum = _lab_error_maps(pred, gt)
    return _lab_rmse_from_map(sq_sum, mask)


def _get_lpips_model():
    global _LPIPS_MODEL, _LPIPS_MODULE, _LPIPS_WARNING_PRINTED
    if _LPIPS_MODULE is None:
        try:
            import lpips as lpips_module
            _LPIPS_MODULE = lpips_module
        except Exception as exc:
            if not _LPIPS_WARNING_PRINTED:
                print("Warning: lpips package is unavailable ({}); LPIPS metrics will be NaN.".format(exc))
                _LPIPS_WARNING_PRINTED = True
            return None
    if _LPIPS_MODULE is None:
        if not _LPIPS_WARNING_PRINTED:
            print("Warning: lpips package is not installed; LPIPS metrics will be NaN.")
            _LPIPS_WARNING_PRINTED = True
        return None
    if _LPIPS_MODEL is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        _LPIPS_MODEL = _LPIPS_MODULE.LPIPS(net='alex').to(device).eval()
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


def compute_paper_metrics(pred_rgb, gt_rgb, mask, size=256):
    """Compute SRD/HomoFormer-style metrics for one RGB image.

    HomoFormer CVPR'24 states that SRD evaluation uses DHAN public shadow
    masks, resizes estimated shadow-free images to 256x256, reports PSNR/SSIM
    in RGB space, reports LAB-space MAE/RMSE, and optionally reports LPIPS
    when the lpips package is installed. Region metrics follow the common
    mask-then-measure SRD convention; empty regions return NaN so crop-based
    debug evaluation cannot create meaningless infinity values.
    """
    pred = _resize_rgb(pred_rgb.astype(np.float32), size=size)
    gt = _resize_rgb(gt_rgb.astype(np.float32), size=size)
    shadow = _resize_mask(mask.astype(np.float32), size=size)
    nonshadow = 1.0 - shadow

    mse_shadow = _masked_mse(pred, gt, shadow)
    mse_nonshadow = _masked_mse(pred, gt, nonshadow)
    abs_sum, sq_sum = _lab_error_maps(pred, gt)
    mae_shadow = _lab_mae_from_map(abs_sum, shadow)
    mae_nonshadow = _lab_mae_from_map(abs_sum, nonshadow)
    rmse_shadow_raw = np.nan
    rmse_nonshadow_raw = np.nan
    if shadow.sum() > 0:
        rmse_shadow_raw = float((sq_sum * shadow).sum() / max(float(shadow.sum()), 1.0))
    if nonshadow.sum() > 0:
        rmse_nonshadow_raw = float((sq_sum * nonshadow).sum() / max(float(nonshadow.sum()), 1.0))
    lpips_shadow = _lpips_region(pred, gt, shadow)
    lpips_nonshadow = _lpips_region(pred, gt, nonshadow)

    metrics = {
        'psnr_all': float(psnr_loss(pred, gt, data_range=1.0)),
        'mae_all': _lab_mae_from_map(abs_sum),
        'rmse_all': _lab_rmse_from_map(sq_sum),
        'lpips_all': _lpips_region(pred, gt),
        'psnr_shadow': _psnr_from_mse(mse_shadow),
        'mae_shadow': mae_shadow,
        'rmse_shadow': float(np.sqrt(rmse_shadow_raw)) if np.isfinite(rmse_shadow_raw) else float('nan'),
        'lpips_shadow': lpips_shadow,
        'psnr_nonshadow': _psnr_from_mse(mse_nonshadow),
        'mae_nonshadow': mae_nonshadow,
        'rmse_nonshadow': float(np.sqrt(rmse_nonshadow_raw)) if np.isfinite(rmse_nonshadow_raw) else float('nan'),
        'lpips_nonshadow': lpips_nonshadow,
    }
    metrics.update(_ssim_region_metrics(pred, gt, shadow, nonshadow))
    return metrics


def _tensor_to_rgb_numpy(tensor):
    return torch.clamp(tensor, 0, 1).detach().cpu().numpy().transpose(1, 2, 0)


def compute_tensor_batch_paper_metrics(pred, gt, mask, size=256):
    metrics = {key: [] for key in METRIC_KEYS}
    for pred_i, gt_i, mask_i in zip(pred, gt, mask):
        sample = compute_paper_metrics(
            _tensor_to_rgb_numpy(pred_i),
            _tensor_to_rgb_numpy(gt_i),
            mask_i.detach().cpu().numpy().squeeze(),
            size=size)
        merge_metric_lists(metrics, sample)
    return metrics


def merge_metric_lists(dst, src):
    for key in METRIC_KEYS:
        value = src[key]
        if isinstance(value, list):
            dst.setdefault(key, []).extend(value)
        else:
            dst.setdefault(key, []).append(value)


def mean_metric_lists(metrics):
    out = {}
    for key in METRIC_KEYS:
        values = np.asarray(metrics.get(key, []), dtype=np.float64)
        values = values[np.isfinite(values)]
        out[key] = float(values.mean()) if values.size else float('nan')
    return out


def legacy_metric_lines(metrics):
    return [
        "PSNR: %f, SSIM: %f, MAE: %f, RMSE: %f, LPIPS: %f " % (
            metrics['psnr_all'], metrics['ssim_all'], metrics['mae_all'],
            metrics['rmse_all'], metrics['lpips_all']),
        "SPSNR: %f, SSSIM: %f, SMAE: %f, SRMSE: %f, SLPIPS: %f " % (
            metrics['psnr_shadow'], metrics['ssim_shadow'], metrics['mae_shadow'],
            metrics['rmse_shadow'], metrics['lpips_shadow']),
        "NSPSNR: %f, NSSSIM: %f, NSMAE: %f, NSRMSE: %f, NSLPIPS: %f " % (
            metrics['psnr_nonshadow'], metrics['ssim_nonshadow'], metrics['mae_nonshadow'],
            metrics['rmse_nonshadow'], metrics['lpips_nonshadow']),
    ]


def format_metric_table(method_name, metrics):
    lines = [
        "Metric table (HomoFormer/SRD protocol: 256x256; RGB PSNR/SSIM; LAB MAE/RMSE; LPIPS if available):",
        "{:<28} | {:^45} | {:^49} | {:^45}".format(
            "Method", "Shadow Region", "Non-Shadow Region", "All Region"),
        "{:<28} | {:>8} {:>8} {:>8} {:>8} {:>8} | {:>8} {:>8} {:>8} {:>8} {:>8} | {:>8} {:>8} {:>8} {:>8} {:>8}".format(
            "", "PSNR", "SSIM", "MAE", "RMSE", "LPIPS",
            "PSNR", "SSIM", "MAE", "RMSE", "LPIPS",
            "PSNR", "SSIM", "MAE", "RMSE", "LPIPS"),
        "-" * 181,
        "{:<28} | {:8.2f} {:8.3f} {:8.2f} {:8.2f} {:8.4f} | {:8.2f} {:8.3f} {:8.2f} {:8.2f} {:8.4f} | {:8.2f} {:8.3f} {:8.2f} {:8.2f} {:8.4f}".format(
            method_name,
            metrics['psnr_shadow'], metrics['ssim_shadow'], metrics['mae_shadow'],
            metrics['rmse_shadow'], metrics['lpips_shadow'],
            metrics['psnr_nonshadow'], metrics['ssim_nonshadow'], metrics['mae_nonshadow'],
            metrics['rmse_nonshadow'], metrics['lpips_nonshadow'],
            metrics['psnr_all'], metrics['ssim_all'], metrics['mae_all'],
            metrics['rmse_all'], metrics['lpips_all'])
    ]
    checks = []
    for name in ('psnr', 'mae', 'rmse'):
        shadow = metrics.get('{}_shadow'.format(name), np.nan)
        nonshadow = metrics.get('{}_nonshadow'.format(name), np.nan)
        all_value = metrics.get('{}_all'.format(name), np.nan)
        if np.isfinite(shadow) and np.isfinite(nonshadow) and np.isfinite(all_value):
            lo, hi = min(shadow, nonshadow), max(shadow, nonshadow)
            if all_value < lo - 1e-6 or all_value > hi + 1e-6:
                checks.append("{} all={:.4f} outside [{:.4f}, {:.4f}]".format(
                    name.upper(), all_value, lo, hi))
    if checks:
        lines.append("Metric range warning: " + "; ".join(checks))
    return "\n".join(lines)
