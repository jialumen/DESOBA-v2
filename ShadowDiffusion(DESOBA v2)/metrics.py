import os
os.environ.setdefault('KMP_DUPLICATE_LIB_OK', 'TRUE')
import math
import numpy as np
import cv2
import torch
from torchvision.utils import make_grid

try:
    from skimage.metrics import peak_signal_noise_ratio as skimage_psnr
    from skimage.metrics import structural_similarity as skimage_ssim
except Exception:
    skimage_psnr = None
    skimage_ssim = None


REGION_METRIC_KEYS = (
    'psnr_all', 'ssim_all', 'lpips_all', 'mae_all', 'rmse_all',
    'psnr_shadow', 'ssim_shadow', 'lpips_shadow', 'mae_shadow', 'rmse_shadow',
    'psnr_nonshadow', 'ssim_nonshadow', 'lpips_nonshadow', 'mae_nonshadow', 'rmse_nonshadow',
)

_LPIPS_MODEL = None
_LPIPS_MODEL_SPATIAL = None
_LPIPS_MODULE = None
_LPIPS_WARNING_PRINTED = False


def tensor2img(tensor, out_type=np.uint8, min_max=(-1, 1)):
    '''
    Converts a torch Tensor into an image Numpy array
    Input: 4D(B,(3/1),H,W), 3D(C,H,W), or 2D(H,W), any range, RGB channel order
    Output: 3D(H,W,C) or 2D(H,W), [0,255], np.uint8 (default)
    '''
    tensor = tensor.squeeze().float().cpu().clamp_(*min_max)  # clamp
    tensor = (tensor - min_max[0]) / \
        (min_max[1] - min_max[0])  # to range [0,1]
    n_dim = tensor.dim()
    if n_dim == 4:
        n_img = len(tensor)
        img_np = make_grid(tensor, nrow=int(
            math.sqrt(n_img)), normalize=False).numpy()
        img_np = np.transpose(img_np, (1, 2, 0))  # HWC, RGB
    elif n_dim == 3:
        img_np = tensor.numpy()
        img_np = np.transpose(img_np, (1, 2, 0))  # HWC, RGB
    elif n_dim == 2:
        img_np = tensor.numpy()
    else:
        raise TypeError(
            'Only support 4D, 3D and 2D tensor. But received with dimension: {:d}'.format(n_dim))
    if out_type == np.uint8:
        img_np = (img_np * 255.0).round()
        # Important. Unlike matlab, numpy.unit8() WILL NOT round by default.
    return img_np.astype(out_type)


def save_img(img, img_path, mode='RGB'):
    cv2.imwrite(img_path, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
    # cv2.imwrite(img_path, img)


def calculate_psnr(img1, img2):
    # img1 and img2 have range [0, 255]
    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    mse = np.mean((img1 - img2)**2)
    if mse == 0:
        return float('inf')
    return 20 * math.log10(255.0 / math.sqrt(mse))


def ssim(img1, img2):
    C1 = (0.01 * 255)**2
    C2 = (0.03 * 255)**2

    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    kernel = cv2.getGaussianKernel(11, 1.5)
    window = np.outer(kernel, kernel.transpose())

    mu1 = cv2.filter2D(img1, -1, window)[5:-5, 5:-5]  # valid
    mu2 = cv2.filter2D(img2, -1, window)[5:-5, 5:-5]
    mu1_sq = mu1**2
    mu2_sq = mu2**2
    mu1_mu2 = mu1 * mu2
    sigma1_sq = cv2.filter2D(img1**2, -1, window)[5:-5, 5:-5] - mu1_sq
    sigma2_sq = cv2.filter2D(img2**2, -1, window)[5:-5, 5:-5] - mu2_sq
    sigma12 = cv2.filter2D(img1 * img2, -1, window)[5:-5, 5:-5] - mu1_mu2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) *
                                                            (sigma1_sq + sigma2_sq + C2))
    return ssim_map.mean()


def calculate_ssim(img1, img2):
    '''calculate SSIM
    the same outputs as MATLAB's
    img1, img2: [0, 255]
    '''
    if not img1.shape == img2.shape:
        raise ValueError('Input images must have the same dimensions.')
    if img1.ndim == 2:
        return ssim(img1, img2)
    elif img1.ndim == 3:
        if img1.shape[2] == 3:
            ssims = []
            for i in range(3):
                ssims.append(ssim(img1[:, :, i], img2[:, :, i]))
            return np.array(ssims).mean()
        elif img1.shape[2] == 1:
            return ssim(np.squeeze(img1), np.squeeze(img2))
    else:
        raise ValueError('Wrong input image dimensions.')


def _resize_rgb01(image, size=256):
    image = cv2.resize(image, (size, size), interpolation=cv2.INTER_AREA)
    return np.clip(image.astype(np.float32), 0.0, 1.0)


def _resize_mask01(mask, size=256):
    if mask.ndim == 3:
        mask = np.squeeze(mask)
    mask = cv2.resize(mask.astype(np.float32), (size, size), interpolation=cv2.INTER_NEAREST)
    return (mask >= 0.5).astype(np.float32)


def _psnr01(pred, gt):
    if skimage_psnr is not None:
        return float(skimage_psnr(pred, gt, data_range=1.0))
    mse = float(np.mean((pred.astype(np.float64) - gt.astype(np.float64)) ** 2))
    if mse == 0:
        return float('inf')
    return 20 * math.log10(1.0 / math.sqrt(mse))


def _ssim01(pred, gt):
    if skimage_ssim is not None:
        return float(skimage_ssim(pred, gt, channel_axis=-1, data_range=1.0))
    return float(calculate_ssim((pred * 255.0).round().astype(np.uint8),
                                (gt * 255.0).round().astype(np.uint8)))


def _masked_psnr01(pred, gt, mask):
    if mask.sum() <= 0:
        return float('nan')
    region = mask > 0.5
    mse = float(np.mean(np.square(pred[region] - gt[region])))
    if mse == 0:
        return float('inf')
    return 20 * math.log10(1.0 / math.sqrt(mse))


def _masked_ssim01(pred, gt, mask):
    if mask.sum() <= 0:
        return float('nan')
    if skimage_ssim is not None:
        _, ssim_map = skimage_ssim(
            pred, gt, channel_axis=-1, data_range=1.0, full=True)
        if ssim_map.ndim == 3:
            ssim_map = ssim_map.mean(axis=2)
        return float((ssim_map * mask).sum() / mask.sum())

    # Fallback for environments without skimage. This is less exact than
    # averaging an SSIM map, but keeps masked SSIM available.
    mask3 = mask[:, :, None]
    return _ssim01(pred * mask3, gt * mask3)


def _lab_mae(pred, gt, mask=None):
    lab_pred = cv2.cvtColor(pred, cv2.COLOR_RGB2LAB)
    lab_gt = cv2.cvtColor(gt, cv2.COLOR_RGB2LAB)
    if mask is None:
        return float(np.abs(lab_pred - lab_gt).mean() * 3.0)

    if mask.sum() <= 0:
        return float('nan')
    region = mask > 0.5
    return float(np.abs(lab_pred[region] - lab_gt[region]).mean() * 3.0)


def _lab_rmse(pred, gt, mask=None):
    lab_pred = cv2.cvtColor(pred, cv2.COLOR_RGB2LAB)
    lab_gt = cv2.cvtColor(gt, cv2.COLOR_RGB2LAB)
    if mask is None:
        return float(np.sqrt(np.square(lab_pred - lab_gt).sum() / (pred.shape[0] * pred.shape[1])))

    if mask.sum() <= 0:
        return float('nan')
    region = mask > 0.5
    return float(np.sqrt(np.square(lab_pred[region] - lab_gt[region]).sum() / region.sum()))


def _get_lpips_model(spatial=False):
    global _LPIPS_MODEL, _LPIPS_MODEL_SPATIAL, _LPIPS_MODULE, _LPIPS_WARNING_PRINTED
    if _LPIPS_MODULE is None:
        try:
            import lpips as lpips_module
            _LPIPS_MODULE = lpips_module
        except Exception as exc:
            if not _LPIPS_WARNING_PRINTED:
                print("Warning: lpips is unavailable ({}); LPIPS metrics will be NaN.".format(exc))
                _LPIPS_WARNING_PRINTED = True
            return None
    if spatial:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        if _LPIPS_MODEL_SPATIAL is None:
            _LPIPS_MODEL_SPATIAL = _LPIPS_MODULE.LPIPS(net='alex', spatial=True).to(device).eval()
        return _LPIPS_MODEL_SPATIAL
    if _LPIPS_MODEL is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        _LPIPS_MODEL = _LPIPS_MODULE.LPIPS(net='alex', spatial=False).to(device).eval()
    return _LPIPS_MODEL


def _lpips_region(pred, gt, mask=None):
    if mask is not None and mask.sum() <= 0:
        return float('nan')
    model = _get_lpips_model(spatial=mask is not None)
    if model is None:
        return float('nan')
    region_mask = None
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
        region_mask = region[r0:r1, c0:c1].astype(np.float32)
    if min(pred.shape[:2]) < 64:
        new_h = max(64, pred.shape[0])
        new_w = max(64, pred.shape[1])
        pred = cv2.resize(pred, (new_w, new_h), interpolation=cv2.INTER_AREA)
        gt = cv2.resize(gt, (new_w, new_h), interpolation=cv2.INTER_AREA)
        if region_mask is not None:
            region_mask = cv2.resize(
                region_mask, (new_w, new_h), interpolation=cv2.INTER_NEAREST)
    pred_t = torch.from_numpy(pred.transpose(2, 0, 1)).unsqueeze(0).float() * 2.0 - 1.0
    gt_t = torch.from_numpy(gt.transpose(2, 0, 1)).unsqueeze(0).float() * 2.0 - 1.0
    device = next(model.parameters()).device
    with torch.no_grad():
        value = model(pred_t.to(device), gt_t.to(device))
    if region_mask is not None:
        value = value.detach().float().cpu()
        if value.dim() == 4:
            value_map = value[0, 0].numpy()
        else:
            value_map = value.reshape(value.shape[-2], value.shape[-1]).numpy()
        mask_map = cv2.resize(
            region_mask.astype(np.float32),
            (value_map.shape[1], value_map.shape[0]),
            interpolation=cv2.INTER_NEAREST)
        if mask_map.sum() <= 0:
            return float('nan')
        return float((value_map * mask_map).sum() / mask_map.sum())
    return float(value.detach().cpu().reshape(-1)[0].item())


def calculate_mask_stats_from_tensors(mask, size=256):
    if mask.dim() == 2:
        mask = mask.unsqueeze(0).unsqueeze(0)
    elif mask.dim() == 3:
        mask = mask.unsqueeze(1)

    stats = []
    for mask_i in mask:
        raw = mask_i.detach().float().cpu().numpy().squeeze()
        resized = _resize_mask01(raw, size=size)
        stats.append({
            'raw_min': float(np.min(raw)),
            'raw_max': float(np.max(raw)),
            'raw_mean': float(np.mean(raw)),
            'eval_shadow_ratio': float(np.mean(resized)),
        })
    return stats


def mean_mask_stats(stats):
    if not stats:
        return {
            'raw_min': float('nan'),
            'raw_max': float('nan'),
            'raw_mean': float('nan'),
            'eval_shadow_ratio': float('nan'),
        }
    return {
        key: float(np.mean([item[key] for item in stats]))
        for key in ('raw_min', 'raw_max', 'raw_mean', 'eval_shadow_ratio')
    }


def calculate_region_metrics(pred_rgb, gt_rgb, mask, size=256):
    pred = _resize_rgb01(pred_rgb, size=size)
    gt = _resize_rgb01(gt_rgb, size=size)
    shadow = _resize_mask01(mask, size=size)
    nonshadow = 1.0 - shadow

    return {
        'psnr_all': _psnr01(pred, gt),
        'ssim_all': _ssim01(pred, gt),
        'lpips_all': _lpips_region(pred, gt),
        'mae_all': _lab_mae(pred, gt),
        'rmse_all': _lab_rmse(pred, gt),
        'psnr_shadow': _masked_psnr01(pred, gt, shadow),
        'ssim_shadow': _masked_ssim01(pred, gt, shadow),
        'lpips_shadow': _lpips_region(pred, gt, shadow),
        'mae_shadow': _lab_mae(pred, gt, shadow),
        'rmse_shadow': _lab_rmse(pred, gt, shadow),
        'psnr_nonshadow': _masked_psnr01(pred, gt, nonshadow),
        'ssim_nonshadow': _masked_ssim01(pred, gt, nonshadow),
        'lpips_nonshadow': _lpips_region(pred, gt, nonshadow),
        'mae_nonshadow': _lab_mae(pred, gt, nonshadow),
        'rmse_nonshadow': _lab_rmse(pred, gt, nonshadow),
    }


def _tensor_to_rgb01(tensor, min_max=(-1, 1)):
    tensor = tensor.detach().float().cpu().clamp_(*min_max)
    tensor = (tensor - min_max[0]) / (min_max[1] - min_max[0])
    return tensor.numpy().transpose(1, 2, 0)


def calculate_region_metrics_from_tensors(pred, gt, mask, size=256, min_max=(-1, 1)):
    if pred.dim() == 3:
        pred = pred.unsqueeze(0)
    if gt.dim() == 3:
        gt = gt.unsqueeze(0)
    if mask.dim() == 2:
        mask = mask.unsqueeze(0).unsqueeze(0)
    elif mask.dim() == 3:
        mask = mask.unsqueeze(1)

    metrics = {key: [] for key in REGION_METRIC_KEYS}
    for pred_i, gt_i, mask_i in zip(pred, gt, mask):
        sample = calculate_region_metrics(
            _tensor_to_rgb01(pred_i, min_max=min_max),
            _tensor_to_rgb01(gt_i, min_max=min_max),
            mask_i.detach().float().cpu().numpy().squeeze(),
            size=size)
        merge_metric_lists(metrics, sample)
    return metrics


def merge_metric_lists(dst, src):
    for key in REGION_METRIC_KEYS:
        value = src[key]
        if isinstance(value, list):
            dst.setdefault(key, []).extend(value)
        else:
            dst.setdefault(key, []).append(value)


def mean_metric_lists(metrics):
    out = {}
    for key in REGION_METRIC_KEYS:
        values = np.asarray(metrics.get(key, []), dtype=np.float64)
        values = values[np.isfinite(values)]
        out[key] = float(values.mean()) if values.size else float('nan')
    return out


def format_region_metrics(metrics):
    lines = []
    for region, label in [('shadow', 'Shadow'), ('nonshadow', 'non-Shadow'), ('all', 'all')]:
        lines.append(
            '  {}: PSNR {:.4f} | SSIM {:.4f} | LPIPS {:.4f} | MAE {:.6f} | RMSE {:.6f}'.format(
                label,
                metrics['psnr_{}'.format(region)],
                metrics['ssim_{}'.format(region)],
                metrics['lpips_{}'.format(region)],
                metrics['mae_{}'.format(region)],
                metrics['rmse_{}'.format(region)]))
    return '\n'.join(lines)
