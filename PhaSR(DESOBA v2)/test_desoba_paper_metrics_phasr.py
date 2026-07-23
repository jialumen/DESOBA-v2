#!/usr/bin/env python

import argparse
import csv
import math
import os
import sys
import time

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from skimage.metrics import peak_signal_noise_ratio as psnr_loss
from skimage.metrics import structural_similarity as ssim_loss

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.chdir(os.path.dirname(os.path.abspath(__file__)))
torch.backends.cudnn.benchmark = False

import utils
from region_metrics import build_gt_map, find_by_stem, load_geo


REGIONS = ("shadow", "nonshadow", "all")


class _Opt:
    pass


def parse_args():
    p = argparse.ArgumentParser(
        description="Evaluate PhaSR on DESOBA v2 using the HomoFormer paper metric protocol.")
    p.add_argument("--weights", required=True)
    p.add_argument("--data", required=True)
    p.add_argument("--mask_dir", default="")
    p.add_argument("--win_size", type=int, default=16)
    p.add_argument("--embed_dim", type=int, default=32)
    p.add_argument("--eval_size", type=int, default=256)
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--save_csv", default="")
    return p.parse_args()


def resize_image(image, size):
    return cv2.resize(image * 255.0, (size, size), interpolation=cv2.INTER_AREA) / 255.0


def resize_mask(mask, size):
    if mask.ndim == 3:
        mask = mask.squeeze()
    mask = cv2.resize(mask.astype(np.float32) * 255.0, (size, size), interpolation=cv2.INTER_AREA) / 255.0
    return np.where(mask < 0.001, np.zeros_like(mask), np.ones_like(mask)).astype(np.float32)


def masked_psnr(pred, gt, mask):
    if mask.sum() <= 0:
        return float("nan")
    mask3 = mask[:, :, None]
    return float(psnr_loss(pred * mask3, gt * mask3, data_range=1.0))


def masked_ssim(pred, gt, mask):
    if mask.sum() <= 0:
        return float("nan")
    if pred.ndim == 3:
        mask3 = mask[:, :, None]
        return float(ssim_loss(pred * mask3, gt * mask3, channel_axis=-1, data_range=1.0))
    return float(ssim_loss(pred * mask, gt * mask, data_range=1.0))


def lab_mae(pred, gt, mask=None):
    if mask is None:
        lab_pred = cv2.cvtColor(pred, cv2.COLOR_RGB2LAB)
        lab_gt = cv2.cvtColor(gt, cv2.COLOR_RGB2LAB)
        return float(np.abs(lab_pred - lab_gt).mean())
    if mask.sum() <= 0:
        return float("nan")
    mask3 = mask[:, :, None]
    lab_pred = cv2.cvtColor(pred * mask3, cv2.COLOR_RGB2LAB)
    lab_gt = cv2.cvtColor(gt * mask3, cv2.COLOR_RGB2LAB)
    return float(np.abs(lab_pred - lab_gt).sum() / mask3.sum())


def lab_rmse(pred, gt, mask=None):
    if mask is None:
        lab_pred = cv2.cvtColor(pred, cv2.COLOR_RGB2LAB)
        lab_gt = cv2.cvtColor(gt, cv2.COLOR_RGB2LAB)
        return float(np.sqrt(np.square(lab_pred - lab_gt).mean()))
    if mask.sum() <= 0:
        return float("nan")
    mask3 = mask[:, :, None]
    lab_pred = cv2.cvtColor(pred * mask3, cv2.COLOR_RGB2LAB)
    lab_gt = cv2.cvtColor(gt * mask3, cv2.COLOR_RGB2LAB)
    return float(np.sqrt(np.square(lab_pred - lab_gt).sum() / mask3.sum()))


def compute_paper_metrics(pred_rgb, gt_rgb, mask, size=256):
    pred = resize_image(np.clip(pred_rgb, 0.0, 1.0), size)
    gt = resize_image(np.clip(gt_rgb, 0.0, 1.0), size)
    shadow = resize_mask(mask, size)
    nonshadow = 1.0 - shadow
    return {
        "psnr_shadow": masked_psnr(pred, gt, shadow),
        "ssim_shadow": masked_ssim(pred, gt, shadow),
        "mae_shadow": lab_mae(pred, gt, shadow),
        "rmse_shadow": lab_rmse(pred, gt, shadow),
        "psnr_nonshadow": masked_psnr(pred, gt, nonshadow),
        "ssim_nonshadow": masked_ssim(pred, gt, nonshadow),
        "mae_nonshadow": lab_mae(pred, gt, nonshadow),
        "rmse_nonshadow": lab_rmse(pred, gt, nonshadow),
        "psnr_all": float(psnr_loss(pred, gt, data_range=1.0)),
        "ssim_all": float(ssim_loss(pred, gt, channel_axis=-1, data_range=1.0)),
        "mae_all": lab_mae(pred, gt),
        "rmse_all": lab_rmse(pred, gt),
    }


def psnr_from_mse(mse):
    return 99.0 if mse <= 1e-12 else float(-10.0 * np.log10(mse))


def true_region_psnr(pred, gt, mask):
    region = mask > 0.5
    if region.sum() <= 0:
        return float("nan")
    diff = pred[region] - gt[region]
    return psnr_from_mse(float(np.square(diff).mean()))


def true_region_mae(pred, gt, mask):
    region = mask > 0.5
    if region.sum() <= 0:
        return float("nan")
    lab_pred = cv2.cvtColor(pred, cv2.COLOR_RGB2LAB)
    lab_gt = cv2.cvtColor(gt, cv2.COLOR_RGB2LAB)
    return float(np.abs(lab_pred[region] - lab_gt[region]).mean())


def true_region_rmse(pred, gt, mask):
    region = mask > 0.5
    if region.sum() <= 0:
        return float("nan")
    lab_pred = cv2.cvtColor(pred, cv2.COLOR_RGB2LAB)
    lab_gt = cv2.cvtColor(gt, cv2.COLOR_RGB2LAB)
    return float(np.sqrt(np.square(lab_pred[region] - lab_gt[region]).mean()))


def true_region_ssim(pred, gt, mask):
    if mask.sum() <= 0:
        return float("nan")
    score, ssim_map = ssim_loss(gt, pred, channel_axis=-1, data_range=1.0, full=True)
    if mask.mean() >= 0.999:
        return float(score)
    if ssim_map.ndim == 3:
        ssim_map = ssim_map.mean(axis=2)
    return float((ssim_map * mask).sum() / max(mask.sum(), 1e-12))


def compute_true_region_metrics(pred_rgb, gt_rgb, mask, size=256):
    pred = resize_image(np.clip(pred_rgb, 0.0, 1.0), size)
    gt = resize_image(np.clip(gt_rgb, 0.0, 1.0), size)
    shadow = resize_mask(mask, size)
    nonshadow = 1.0 - shadow
    all_mask = np.ones_like(shadow, dtype=np.float32)
    return {
        "psnr_shadow": true_region_psnr(pred, gt, shadow),
        "ssim_shadow": true_region_ssim(pred, gt, shadow),
        "mae_shadow": true_region_mae(pred, gt, shadow),
        "rmse_shadow": true_region_rmse(pred, gt, shadow),
        "psnr_nonshadow": true_region_psnr(pred, gt, nonshadow),
        "ssim_nonshadow": true_region_ssim(pred, gt, nonshadow),
        "mae_nonshadow": true_region_mae(pred, gt, nonshadow),
        "rmse_nonshadow": true_region_rmse(pred, gt, nonshadow),
        "psnr_all": true_region_psnr(pred, gt, all_mask),
        "ssim_all": true_region_ssim(pred, gt, all_mask),
        "mae_all": true_region_mae(pred, gt, all_mask),
        "rmse_all": true_region_rmse(pred, gt, all_mask),
    }


def mean_lists(metric_lists):
    out = {}
    for key, vals in metric_lists.items():
        arr = np.asarray(vals, dtype=np.float64)
        arr = arr[np.isfinite(arr)]
        out[key] = float(arr.mean()) if arr.size else float("nan")
    return out


def load_mask(mask_dir, stem, shape):
    path = find_by_stem(mask_dir, stem)
    if not path:
        path = find_by_stem(mask_dir, stem.rsplit("_", 1)[0])
    if not path:
        raise FileNotFoundError(f"missing mask for {stem} under {mask_dir}")
    mask = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise RuntimeError(f"failed to read mask: {path}")
    if mask.shape[:2] != shape[:2]:
        mask = cv2.resize(mask, (shape[1], shape[0]), interpolation=cv2.INTER_NEAREST)
    mask = (mask > 127).astype(np.float32)
    if mask.mean() > 0.85:
        mask = 1.0 - mask
    return mask


def format_table(metrics, method):
    lines = [
        "Metric table (zero-mask compatibility; PSNR is full-image after pred*mask, not true-region PSNR):",
        f"{method:<14} {'Region':<11} {'PSNR':>8} {'SSIM':>8} {'MAE':>8} {'RMSE':>8}",
    ]
    labels = (("shadow", "Shadow"), ("nonshadow", "non-Shadow"), ("all", "All"))
    for key, label in labels:
        lines.append(
            f"{method:<14} {label:<11} {metrics['psnr_' + key]:8.3f} "
            f"{metrics['ssim_' + key]:8.3f} {metrics['mae_' + key]:8.3f} {metrics['rmse_' + key]:8.3f}"
        )
    return "\n".join(lines)


def format_true_region_table(metrics, method):
    lines = [
        "Metric table (paper-style true-region mean; RGB PSNR/SSIM, LAB MAE/RMSE on selected pixels):",
        f"{method:<14} {'Region':<11} {'PSNR':>8} {'SSIM':>8} {'MAE':>8} {'RMSE':>8}",
    ]
    labels = (("shadow", "Shadow"), ("nonshadow", "non-Shadow"), ("all", "All"))
    for key, label in labels:
        lines.append(
            f"{method:<14} {label:<11} {metrics['psnr_' + key]:8.3f} "
            f"{metrics['ssim_' + key]:8.3f} {metrics['mae_' + key]:8.5f} {metrics['rmse_' + key]:8.5f}"
        )
    return "\n".join(lines)


def save_metrics_csv(path, metrics):
    if not path:
        return
    out_dir = os.path.dirname(os.path.abspath(path))
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["region", "psnr", "ssim", "mae", "rmse"])
        for key, label in (("shadow", "Shadow"), ("nonshadow", "non-Shadow"), ("all", "All")):
            writer.writerow([
                label,
                f"{metrics['psnr_' + key]:.6f}",
                f"{metrics['ssim_' + key]:.6f}",
                f"{metrics['mae_' + key]:.6f}",
                f"{metrics['rmse_' + key]:.6f}",
            ])


def main():
    args = parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    dps = 14
    opt = _Opt()
    opt.arch = "PhaSR"
    opt.train_ps = args.eval_size
    opt.embed_dim = args.embed_dim
    opt.win_size = args.win_size
    opt.token_projection = "linear"
    opt.token_mlp = "leff"

    model = utils.get_arch(opt).to(dev).eval()
    utils.load_checkpoint(model, args.weights)
    dino = torch.hub.load("./dinov2", "dinov2_vitl14", source="local").to(dev).eval()
    print(f"[paper-test] loaded {args.weights} | eval@{args.eval_size}", flush=True)

    def dfeat(x):
        xu = F.interpolate(x, size=(int(x.shape[2] * dps / 8), int(x.shape[3] * dps / 8)),
                           mode="bilinear", align_corners=False)
        return dino.get_intermediate_layers(xu, 4, True)

    def infer(inp, pt, nm):
        m = 8 * args.win_size
        h, w = inp.shape[2], inp.shape[3]
        H = ((h + m - 1) // m) * m
        W = ((w + m - 1) // m) * m
        xi = F.pad(inp, (0, W - w, 0, H - h), "reflect")
        pp = F.pad(pt, (0, W - w, 0, H - h), "reflect")
        nn = F.pad(nm, (0, W - w, 0, H - h), "reflect")
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            out = model(xi, dfeat(xi), pp, nn)
        return out.float().clamp(0, 1)[:, :, :h, :w]

    origin_dir = os.path.join(args.data, "origin")
    gt_dir = os.path.join(args.data, "shadow_free")
    mask_dir = args.mask_dir or os.path.join(args.data, "shadow_mask")
    print(f"[paper-test] metric protocol: true-region RGB PSNR/SSIM + true-region LAB MAE/RMSE", flush=True)
    print(f"[paper-test] mask_dir: {mask_dir}", flush=True)
    files = sorted(os.listdir(origin_dir))
    if args.limit:
        files = files[:args.limit]
    gt_map = build_gt_map(gt_dir)
    true_metric_lists = {}
    t0 = time.time()
    with torch.no_grad():
        for i, fn in enumerate(files):
            stem = os.path.splitext(fn)[0]
            inp = cv2.cvtColor(cv2.imread(os.path.join(origin_dir, fn)), cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
            point, normal = load_geo(args.data, stem)
            it = torch.from_numpy(inp).permute(2, 0, 1).unsqueeze(0).float().to(dev)
            pt = torch.from_numpy(point).permute(2, 0, 1).unsqueeze(0).float().to(dev)
            nm = torch.from_numpy(normal).permute(2, 0, 1).unsqueeze(0).float().to(dev)
            pred = infer(it, pt, nm)[0].detach().cpu().numpy().transpose(1, 2, 0)
            gt_fn = gt_map.get(stem, gt_map.get(stem.rsplit("_", 1)[0], fn))
            gt = cv2.cvtColor(cv2.imread(os.path.join(gt_dir, gt_fn)), cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
            mask = load_mask(mask_dir, stem, gt.shape)
            true_sample = compute_true_region_metrics(pred, gt, mask, size=args.eval_size)
            for key, val in true_sample.items():
                true_metric_lists.setdefault(key, []).append(val)
            if (i + 1) % 100 == 0:
                print(f"  {i+1}/{len(files)} ({time.time()-t0:.0f}s)", flush=True)
    true_metrics = mean_lists(true_metric_lists)
    print(format_true_region_table(true_metrics, os.path.splitext(os.path.basename(args.weights))[0]), flush=True)
    if args.save_csv:
        save_metrics_csv(args.save_csv, true_metrics)
        print(f"[paper-test] saved csv: {args.save_csv}", flush=True)


if __name__ == "__main__":
    main()
