import os
import time

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from skimage.metrics import structural_similarity as ssim_sk
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity as TM_LPIPS

from utils import load_depth, load_normal, depthToPoint, process_normal


REGION_KEYS = ("shadow", "non_shadow", "all")


def make_lpips(device):
    try:
        return TM_LPIPS(net_type="vgg", reduction="mean", normalize=True).to(device).eval()
    except TypeError:
        return TM_LPIPS(net_type="vgg", reduction="mean").to(device).eval()


def to_eval_image(x, eval_size=256, resize="cv2_area"):
    if resize == "finterp_area":
        t = torch.from_numpy(x).permute(2, 0, 1).unsqueeze(0)
        return F.interpolate(t, (eval_size, eval_size), mode="area")[0].permute(1, 2, 0).numpy()
    return cv2.resize(x, (eval_size, eval_size), interpolation=cv2.INTER_AREA)


def find_by_stem(root, stem):
    for ext in (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"):
        path = os.path.join(root, stem + ext)
        if os.path.isfile(path):
            return path
    return ""


def build_gt_map(gt_dir):
    gt_map = {}
    if not os.path.isdir(gt_dir):
        return gt_map
    for fn in os.listdir(gt_dir):
        stem = os.path.splitext(fn)[0]
        gt_map[stem] = fn
        gt_map[stem.rsplit("_", 1)[0]] = fn
    return gt_map


def load_geo(data_dir, stem):
    depth_dir = os.path.join(data_dir, "depth")
    normal_dir = os.path.join(data_dir, "normal")
    depth = load_depth(os.path.join(depth_dir, stem + ".npy"))
    normal = process_normal(load_normal(os.path.join(normal_dir, stem + ".npy")))
    point = depthToPoint(60, depth)
    point = point / (2 * point[:, :, 2].mean())
    return point, normal


def load_shadow_mask(mask_dir, stem, shape, eval_size):
    mask_path = find_by_stem(mask_dir, stem)
    if not mask_path:
        mask_path = find_by_stem(mask_dir, stem.rsplit("_", 1)[0])
    if not mask_path:
        raise FileNotFoundError(f"missing mask for {stem} under {mask_dir}")

    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise RuntimeError(f"failed to read mask: {mask_path}")
    if mask.shape[:2] != shape[:2]:
        mask = cv2.resize(mask, (shape[1], shape[0]), interpolation=cv2.INTER_NEAREST)
    mask = cv2.resize(mask, (eval_size, eval_size), interpolation=cv2.INTER_NEAREST)
    mask = mask > 127
    if mask.mean() > 0.85:
        mask = ~mask
    return mask


def init_meter():
    return {
        k: {"psnr": [], "mae": [], "rmse": [], "ssim": [], "lpips": [], "pixels": 0}
        for k in REGION_KEYS
    }


def psnr_from_mse(mse):
    return 99.0 if mse <= 1e-12 else float(-10.0 * np.log10(mse))


def add_pixel_metrics(meter, key, pred, gt, mask):
    if key == "all":
        sel = np.ones(mask.shape, dtype=bool)
    else:
        sel = mask if key == "shadow" else ~mask
    count = int(sel.sum())
    if count == 0:
        return
    diff_rgb = pred[sel] - gt[sel]
    lab_pred = cv2.cvtColor(pred, cv2.COLOR_RGB2LAB)
    lab_gt = cv2.cvtColor(gt, cv2.COLOR_RGB2LAB)
    diff_lab = lab_pred[sel] - lab_gt[sel]
    mse = float(np.square(diff_rgb).mean())
    meter[key]["psnr"].append(psnr_from_mse(mse))
    meter[key]["mae"].append(float(np.abs(diff_lab).mean()))
    meter[key]["rmse"].append(float(np.sqrt(np.square(diff_lab).mean())))
    meter[key]["pixels"] += count


def replace_outside(pred, gt, mask, key):
    if key == "all":
        return pred
    sel = mask if key == "shadow" else ~mask
    return np.where(sel[:, :, None], pred, gt)


def add_structure_metrics(meter, key, pred, gt, mask, lpips_fn, device):
    pred_gray = cv2.cvtColor(pred, cv2.COLOR_RGB2GRAY)
    gt_gray = cv2.cvtColor(gt, cv2.COLOR_RGB2GRAY)
    ssim_score, ssim_map = ssim_sk(gt_gray, pred_gray, data_range=1.0, full=True)
    if key == "all":
        meter[key]["ssim"].append(float(ssim_score))
    else:
        sel = mask if key == "shadow" else ~mask
        meter[key]["ssim"].append(float(ssim_map[sel].mean()))

    pred_r = replace_outside(pred, gt, mask, key)
    gt_r = gt
    pred_t = torch.from_numpy(pred_r).permute(2, 0, 1).unsqueeze(0).float().to(device)
    gt_t = torch.from_numpy(gt_r).permute(2, 0, 1).unsqueeze(0).float().to(device)
    meter[key]["lpips"].append(float(lpips_fn(pred_t, gt_t).item()))


def finalize_meter(meter):
    out = {}
    for key, vals in meter.items():
        if not vals["psnr"]:
            out[key] = {"psnr": np.nan, "ssim": np.nan, "lpips": np.nan, "mae": np.nan, "rmse": np.nan, "pixels": 0}
            continue
        out[key] = {
            "psnr": float(np.mean(vals["psnr"])),
            "ssim": float(np.mean(vals["ssim"])) if vals["ssim"] else np.nan,
            "lpips": float(np.mean(vals["lpips"])) if vals["lpips"] else np.nan,
            "mae": float(np.mean(vals["mae"])),
            "rmse": float(np.mean(vals["rmse"])),
            "pixels": vals["pixels"],
        }
    return out


def format_region_metrics(metrics, title="regional metrics"):
    lines = [
        title + " [RGB PSNR + true-region LAB MAE/RMSE per-image mean; LPIPS masked-composite]",
        "region       PSNR     SSIM    LPIPS      MAE     RMSE     pixels",
    ]
    for key in REGION_KEYS:
        m = metrics[key]
        lines.append(
            f"{key:<10} {m['psnr']:7.3f}  {m['ssim']:7.4f}  {m['lpips']:7.4f}  "
            f"{m['mae']:7.5f}  {m['rmse']:7.5f}  {m['pixels']}"
        )
    return lines


@torch.no_grad()
def evaluate_region_metrics(data_dir, infer_fn, device, eval_size=256, resize="cv2_area",
                            mask_dir="", save_dir="", limit=0, progress_every=100,
                            log=print, lpips_fn=None):
    origin_dir = os.path.join(data_dir, "origin")
    gt_dir = os.path.join(data_dir, "shadow_free")
    mask_dir = mask_dir or os.path.join(data_dir, "shadow_mask")
    if not os.path.isdir(origin_dir):
        raise FileNotFoundError(f"missing origin dir: {origin_dir}")
    if not os.path.isdir(gt_dir):
        raise FileNotFoundError(f"missing shadow_free dir: {gt_dir}")
    if not os.path.isdir(mask_dir):
        raise FileNotFoundError(f"missing shadow mask dir: {mask_dir}")
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)

    files = sorted(os.listdir(origin_dir))
    if limit:
        files = files[:limit]
    gt_map = build_gt_map(gt_dir)
    meter = init_meter()
    lpips_fn = lpips_fn or make_lpips(device)
    t0 = time.time()

    for i, fn in enumerate(files):
        stem = os.path.splitext(fn)[0]
        inp = cv2.imread(os.path.join(origin_dir, fn), cv2.IMREAD_COLOR)
        if inp is None:
            raise RuntimeError(f"failed to read input: {fn}")
        inp = cv2.cvtColor(inp, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        point, normal = load_geo(data_dir, stem)

        it = torch.from_numpy(inp).permute(2, 0, 1).unsqueeze(0).float().to(device)
        pt = torch.from_numpy(point).permute(2, 0, 1).unsqueeze(0).float().to(device)
        nm = torch.from_numpy(normal).permute(2, 0, 1).unsqueeze(0).float().to(device)
        pred = infer_fn(it, pt, nm)[0].detach().float().cpu().numpy().transpose(1, 2, 0)
        pred = np.clip(pred, 0.0, 1.0)

        if save_dir:
            out_u8 = (pred * 255.0).round().astype(np.uint8)
            cv2.imwrite(os.path.join(save_dir, fn), cv2.cvtColor(out_u8, cv2.COLOR_RGB2BGR))

        gt_fn = gt_map.get(stem, gt_map.get(stem.rsplit("_", 1)[0], fn))
        gt_path = os.path.join(gt_dir, gt_fn)
        gt = cv2.imread(gt_path, cv2.IMREAD_COLOR)
        if gt is None:
            raise RuntimeError(f"failed to read gt: {gt_path}")
        gt = cv2.cvtColor(gt, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0

        pred_e = to_eval_image(pred, eval_size, resize)
        gt_e = to_eval_image(gt, eval_size, resize)
        mask = load_shadow_mask(mask_dir, stem, gt.shape, eval_size)

        for key in REGION_KEYS:
            add_pixel_metrics(meter, key, pred_e, gt_e, mask)
            add_structure_metrics(meter, key, pred_e, gt_e, mask, lpips_fn, device)

        if progress_every and (i + 1) % progress_every == 0:
            log(f"  metrics {i+1}/{len(files)} ({time.time()-t0:.0f}s)")

    return finalize_meter(meter)
