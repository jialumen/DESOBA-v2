import csv
import math
import os
import time

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from skimage.metrics import peak_signal_noise_ratio as psnr_loss
from skimage.metrics import structural_similarity as ssim_loss
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity

from utils import depthToPoint, load_depth, load_img, load_normal, process_normal


REGION_ORDER = ("shadow", "non_shadow", "all")


def make_lpips(device):
    metric = LearnedPerceptualImagePatchSimilarity(
        net_type="vgg",
        normalize=True,
    )
    return metric.to(device).eval()


def _find_by_stem(directory, filename):
    stem = os.path.splitext(filename)[0]
    direct = os.path.join(directory, filename)
    if os.path.exists(direct):
        return direct

    for ext in (".npy", ".png", ".jpg", ".jpeg", ".bmp"):
        candidate = os.path.join(directory, stem + ext)
        if os.path.exists(candidate):
            return candidate
    raise FileNotFoundError(f"Cannot find file for {filename} under {directory}")


def _load_mask(mask_dir, filename, size):
    path = _find_by_stem(mask_dir, filename)
    mask = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise FileNotFoundError(path)
    mask = cv2.resize(mask, size, interpolation=cv2.INTER_AREA)
    mask = mask.astype(np.float32) / 255.0
    mask = np.where(mask < 0.001, np.zeros_like(mask), np.ones_like(mask)).astype(np.float32)
    return mask


def _image_tensor(path, size):
    image = load_img(path)
    image = cv2.resize(image, size, interpolation=cv2.INTER_AREA)
    tensor = torch.from_numpy(image).permute(2, 0, 1).unsqueeze(0).float()
    return tensor


def _point_tensor(path, size):
    depth = load_depth(path)
    depth = cv2.resize(depth, size, interpolation=cv2.INTER_AREA)
    point = depthToPoint(60, depth)
    denom = 2 * point[:, :, 2].mean()
    point = point / max(float(denom), 1.0e-8)
    return torch.from_numpy(point).permute(2, 0, 1).unsqueeze(0).float()


def _normal_tensor(path, size):
    normal = load_normal(path)
    normal = cv2.resize(normal, size, interpolation=cv2.INTER_AREA)
    normal = process_normal(normal)
    return torch.from_numpy(normal).permute(2, 0, 1).unsqueeze(0).float()


def _as_numpy_image(tensor):
    image = tensor.detach().float().clamp(0, 1).squeeze(0).cpu().numpy()
    return np.transpose(image, (1, 2, 0))


def _torch_image(np_image, device):
    tensor = torch.from_numpy(np_image).permute(2, 0, 1).unsqueeze(0).float()
    return tensor.to(device)


def _masked_rgb(pred, target, mask, region):
    if region == "all":
        return pred, target
    mask3 = mask[:, :, None]
    if region == "non_shadow":
        mask3 = 1.0 - mask3
    return pred * mask3, target * mask3


def _lab_mae(pred, target, mask=None):
    if mask is None:
        lab_pred = cv2.cvtColor(pred, cv2.COLOR_RGB2LAB)
        lab_target = cv2.cvtColor(target, cv2.COLOR_RGB2LAB)
        return float(np.abs(lab_pred - lab_target).mean() * 3.0)
    if mask.sum() <= 0:
        return float("nan")
    lab_pred = cv2.cvtColor(pred, cv2.COLOR_RGB2LAB)
    lab_target = cv2.cvtColor(target, cv2.COLOR_RGB2LAB)
    diff = np.abs(lab_pred - lab_target)
    return float(diff[mask > 0.5].sum() / mask.sum())


def _lab_rmse(pred, target, mask=None):
    if mask is None:
        lab_pred = cv2.cvtColor(pred, cv2.COLOR_RGB2LAB)
        lab_target = cv2.cvtColor(target, cv2.COLOR_RGB2LAB)
        return float(np.sqrt(np.square(lab_pred - lab_target).sum() / (pred.shape[0] * pred.shape[1])))
    if mask.sum() <= 0:
        return float("nan")
    lab_pred = cv2.cvtColor(pred, cv2.COLOR_RGB2LAB)
    lab_target = cv2.cvtColor(target, cv2.COLOR_RGB2LAB)
    diff2 = np.square(lab_pred - lab_target)
    return float(np.sqrt(diff2[mask > 0.5].sum() / mask.sum()))


def _region_psnr(pred, target, region_mask):
    if region_mask.sum() <= 0:
        return float("nan")
    mask3 = region_mask[:, :, None]
    mse = np.square((pred - target) * mask3).sum() / (region_mask.sum() * pred.shape[2])
    if mse <= 0:
        return float("inf")
    return float(10.0 * np.log10(1.0 / mse))


def _region_ssim(pred, target, region_mask):
    if region_mask.sum() <= 0:
        return float("nan")
    _, ssim_map = ssim_loss(
        pred,
        target,
        channel_axis=-1,
        data_range=1.0,
        full=True,
    )
    if ssim_map.ndim == 3:
        mask3 = region_mask[:, :, None].astype(bool)
        mask3 = np.broadcast_to(mask3, ssim_map.shape)
        return float(ssim_map[mask3].mean())
    return float(ssim_map[region_mask > 0.5].mean())


def _compute_paper_metrics(pred, target, mask):
    """Compute DESOBA table-style full and region metrics.

    Images/masks are already resized to eval_size by this file. Shadow and
    non-shadow PSNR/SSIM/MAE/RMSE are measured on their true region pixels so
    small shadow masks cannot artificially inflate regional scores by zeroing
    the rest of the image.
    """
    shadow = mask
    non_shadow = 1.0 - shadow
    return {
        "psnr_all": float(psnr_loss(pred, target, data_range=1.0)),
        "ssim_all": float(ssim_loss(pred, target, channel_axis=-1, data_range=1.0)),
        "mae_all": _lab_mae(pred, target),
        "rmse_all": _lab_rmse(pred, target),
        "psnr_shadow": _region_psnr(pred, target, shadow),
        "ssim_shadow": _region_ssim(pred, target, shadow),
        "mae_shadow": _lab_mae(pred, target, shadow),
        "rmse_shadow": _lab_rmse(pred, target, shadow),
        "psnr_non_shadow": _region_psnr(pred, target, non_shadow),
        "ssim_non_shadow": _region_ssim(pred, target, non_shadow),
        "mae_non_shadow": _lab_mae(pred, target, non_shadow),
        "rmse_non_shadow": _lab_rmse(pred, target, non_shadow),
    }


def _paper_region_metrics(pred, target, mask, region):
    metrics = _compute_paper_metrics(pred, target, mask)
    if region == "all":
        pixels = int(pred.shape[0] * pred.shape[1])
    else:
        region_mask = mask if region == "shadow" else (1.0 - mask)
        pixels = int(region_mask.sum())
    return {
        "PSNR": metrics[f"psnr_{region}"],
        "SSIM": metrics[f"ssim_{region}"],
        "MAE": metrics[f"mae_{region}"],
        "RMSE": metrics[f"rmse_{region}"],
        "pixels": pixels,
    }


def _finite_mean(values):
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    return float(values.mean()) if values.size else float("nan")


def evaluate_region_metrics(
    test_dir,
    mask_dir,
    infer_fn,
    device,
    lpips_metric,
    eval_size=256,
    limit=0,
    logger=None,
):
    origin_dir = os.path.join(test_dir, "origin")
    gt_dir = os.path.join(test_dir, "shadow_free")
    depth_dir = os.path.join(test_dir, "depth")
    normal_dir = os.path.join(test_dir, "normal")
    if not mask_dir:
        mask_dir = os.path.join(test_dir, "shadow_mask")

    filenames = sorted(
        name for name in os.listdir(origin_dir)
        if os.path.splitext(name)[1].lower() in (".png", ".jpg", ".jpeg", ".bmp")
    )
    if limit and limit > 0:
        filenames = filenames[:limit]

    accum = {
        region: {"PSNR": [], "SSIM": [], "MAE": [], "RMSE": [], "LPIPS": [], "pixels": 0, "images": 0}
        for region in REGION_ORDER
    }

    size = (int(eval_size), int(eval_size))
    start = time.time()
    for idx, filename in enumerate(filenames, 1):
        origin_path = _find_by_stem(origin_dir, filename)
        gt_path = _find_by_stem(gt_dir, filename)
        depth_path = _find_by_stem(depth_dir, filename)
        normal_path = _find_by_stem(normal_dir, filename)

        input_tensor = _image_tensor(origin_path, size).to(device)
        point_tensor = _point_tensor(depth_path, size).to(device)
        normal_tensor = _normal_tensor(normal_path, size).to(device)
        target_tensor = _image_tensor(gt_path, size).to(device)

        with torch.no_grad():
            pred_tensor = infer_fn(input_tensor, point_tensor, normal_tensor)
            pred_tensor = pred_tensor.detach().clamp(0.0, 1.0)
            if pred_tensor.shape[-2:] != target_tensor.shape[-2:]:
                pred_tensor = F.interpolate(
                    pred_tensor,
                    size=target_tensor.shape[-2:],
                    mode="bilinear",
                    align_corners=False,
                )

        pred = _as_numpy_image(pred_tensor)
        target = _as_numpy_image(target_tensor)
        mask = _load_mask(mask_dir, filename, size)

        for region in REGION_ORDER:
            item = _paper_region_metrics(pred, target, mask, region)
            for key in ("PSNR", "SSIM", "MAE", "RMSE"):
                accum[region][key].append(item[key])
            accum[region]["pixels"] += item["pixels"]
            accum[region]["images"] += 1

            masked_pred, masked_target = _masked_rgb(pred, target, mask, region)
            accum[region]["LPIPS"].append(
                float(
                    lpips_metric(
                        _torch_image(masked_pred, device),
                        _torch_image(masked_target, device),
                    ).detach().cpu().item()
                )
            )

        if logger and (idx % 50 == 0 or idx == len(filenames)):
            logger.info(
                "Region test progress: %d/%d images, elapsed %.1fs",
                idx,
                len(filenames),
                time.time() - start,
            )

    metrics = {}
    for region, values in accum.items():
        metrics[region] = {
            "PSNR": _finite_mean(values["PSNR"]),
            "SSIM": _finite_mean(values["SSIM"]),
            "LPIPS": _finite_mean(values["LPIPS"]),
            "MAE": _finite_mean(values["MAE"]),
            "RMSE": _finite_mean(values["RMSE"]),
            "pixels": values["pixels"],
            "images": values["images"],
        }
    return metrics


def format_region_metrics(epoch, metrics):
    lines = [f"[Region Test][Epoch {epoch}] region PSNR SSIM LPIPS MAE RMSE pixels"]
    for region in REGION_ORDER:
        item = metrics[region]
        lines.append(
            "[Region Test][Epoch {epoch}] {region:10s} {PSNR:.4f} {SSIM:.4f} "
            "{LPIPS:.4f} {MAE:.6f} {RMSE:.6f} {pixels}".format(
                epoch=epoch,
                region=region,
                **item,
            )
        )
    return lines


def append_region_metrics_csv(path, epoch, metrics):
    exists = os.path.exists(path)
    with open(path, "a", newline="") as handle:
        writer = csv.writer(handle)
        if not exists:
            writer.writerow(["epoch", "region", "PSNR", "SSIM", "LPIPS", "MAE", "RMSE", "pixels", "images"])
        for region in REGION_ORDER:
            item = metrics[region]
            writer.writerow([
                epoch,
                region,
                item["PSNR"],
                item["SSIM"],
                item["LPIPS"],
                item["MAE"],
                item["RMSE"],
                item["pixels"],
                item["images"],
            ])
