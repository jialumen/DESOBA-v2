import argparse
import csv
import math
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from skimage.metrics import structural_similarity
from torchvision.transforms.functional import pil_to_tensor

try:
    import cv2
except Exception:
    cv2 = None


REGIONS = ("Shadow", "non-Shadow", "all")
EVAL_SIZE = 256


def _to_numpy_image(image):
    if isinstance(image, Image.Image):
        arr = np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0
    elif isinstance(image, torch.Tensor):
        tensor = image.detach().cpu().float()
        if tensor.ndim == 4:
            tensor = tensor.squeeze(0)
        if tensor.shape[0] in (1, 3):
            tensor = tensor.permute(1, 2, 0)
        arr = tensor.numpy()
        if arr.min() < 0:
            arr = (arr + 1.0) / 2.0
        arr = np.clip(arr, 0.0, 1.0)
        if arr.shape[-1] == 1:
            arr = np.repeat(arr, 3, axis=-1)
    else:
        arr = np.asarray(image, dtype=np.float32)
        if arr.max() > 1.0:
            arr = arr / 255.0
    return arr


def _resize_rgb_image(arr, size=EVAL_SIZE):
    if arr.shape[:2] == (size, size):
        return arr
    if cv2 is not None:
        return np.clip(cv2.resize(arr.astype(np.float32), (size, size), interpolation=cv2.INTER_AREA), 0.0, 1.0)
    img = Image.fromarray((np.clip(arr, 0, 1) * 255).astype(np.uint8))
    return np.asarray(img.resize((size, size), Image.Resampling.BICUBIC), dtype=np.float32) / 255.0


def _resize_mask_array(arr, size=EVAL_SIZE):
    if arr.shape[:2] != (size, size):
        if cv2 is not None:
            arr = cv2.resize(arr.astype(np.float32) * 255.0, (size, size), interpolation=cv2.INTER_AREA) / 255.0
        else:
            img = Image.fromarray((np.clip(arr, 0, 1) * 255).astype(np.uint8))
            arr = np.asarray(img.resize((size, size), Image.Resampling.NEAREST), dtype=np.float32) / 255.0
    return arr


def _to_numpy_mask(mask, height, width):
    if isinstance(mask, Image.Image):
        arr = np.asarray(mask.convert("L"))
    elif isinstance(mask, torch.Tensor):
        tensor = mask.detach().cpu()
        if tensor.ndim == 4:
            tensor = tensor.squeeze(0)
        if tensor.ndim == 3:
            tensor = tensor.squeeze(0)
        arr = tensor.numpy()
    else:
        arr = np.asarray(mask)
    arr = arr.astype(np.float32)
    if arr.min() < 0:
        arr = (arr + 1.0) / 2.0
    if arr.max() > 1.0:
        arr = arr / 255.0
    return arr


def _select_region(pred, gt, mask, region):
    if region == "all":
        pred_sel = pred.reshape(-1, pred.shape[-1])
        gt_sel = gt.reshape(-1, gt.shape[-1])
    else:
        region_mask = mask if region == "Shadow" else ~mask
        if not region_mask.any():
            return None
        pred_sel = pred[region_mask]
        gt_sel = gt[region_mask]
    return pred_sel, gt_sel


def _lab_diff_sums(pred, gt, mask, region):
    if cv2 is None:
        selected = _select_region(pred, gt, mask, region)
        if selected is None:
            return None
        pred_sel, gt_sel = selected
        diff = (pred_sel - gt_sel) * 255.0
        pixels = pred_sel.shape[0]
        return float(np.abs(diff).sum()), float((diff * diff).sum()), int(pixels)

    pred_lab = cv2.cvtColor(np.ascontiguousarray(pred.astype(np.float32)), cv2.COLOR_RGB2LAB)
    gt_lab = cv2.cvtColor(np.ascontiguousarray(gt.astype(np.float32)), cv2.COLOR_RGB2LAB)
    if region == "all":
        diff = pred_lab - gt_lab
        pixels = pred.shape[0] * pred.shape[1]
    else:
        region_mask = mask if region == "Shadow" else ~mask
        if not region_mask.any():
            return None
        diff = pred_lab[region_mask] - gt_lab[region_mask]
        pixels = int(region_mask.sum())
    return float(np.abs(diff).sum()), float((diff * diff).sum()), pixels


def _region_mask(mask, region):
    if region == "all":
        return np.ones_like(mask, dtype=bool)
    return mask if region == "Shadow" else ~mask


def _ssim(pred, gt, mask):
    min_side = min(pred.shape[:2])
    if min_side < 3:
        return float("nan")
    win_size = min(7, min_side if min_side % 2 else min_side - 1)
    score, ssim_map = structural_similarity(
        gt, pred, channel_axis=-1, data_range=1.0, win_size=win_size, full=True
    )
    if mask is None:
        return float(score)
    if ssim_map.ndim == 3:
        ssim_map = ssim_map.mean(axis=-1)
    if not mask.any():
        return float("nan")
    return float(np.mean(ssim_map[mask]))


class RegionMetricAccumulator:
    def __init__(self, lpips_net=None, device="cuda", lpips_device=None, include_lpips=True):
        self.lpips_net = lpips_net
        self.device = device
        self.lpips_device = lpips_device or device
        self.include_lpips = include_lpips
        self.rows = {
            region: {"psnr": [], "ssim": [], "lpips": [], "mae": [], "rmse": []}
            for region in REGIONS
        }
        self.pixel_sums = {
            region: {"rgb_sse": 0.0, "rgb_count": 0, "lab_sae": 0.0, "lab_sse": 0.0, "lab_pixels": 0}
            for region in REGIONS
        }

    @classmethod
    def build(cls, device="cuda", include_lpips=True, lpips_device=None):
        lpips_device = lpips_device or device
        try:
            import lpips

            net = lpips.LPIPS(net="alex", spatial=True).to(lpips_device).eval() if include_lpips else None
        except Exception:
            net = None
        return cls(lpips_net=net, device=device, lpips_device=lpips_device, include_lpips=include_lpips)

    def update(self, gt_tensor, pred_images, mask_tensor):
        if mask_tensor is None:
            return
        for gt_item, pred_item, mask_item in zip(gt_tensor, pred_images, mask_tensor):
            gt = _resize_rgb_image(_to_numpy_image(gt_item), EVAL_SIZE)
            pred = _resize_rgb_image(_to_numpy_image(pred_item), EVAL_SIZE)
            mask = _resize_mask_array(_to_numpy_mask(mask_item, gt.shape[0], gt.shape[1]), EVAL_SIZE) > 0.001
            for region in REGIONS:
                selected = _select_region(pred, gt, mask, region)
                if selected is None:
                    continue
                pred_sel, gt_sel = selected
                diff = pred_sel - gt_sel
                mse = float(np.mean(diff * diff))
                rmse = math.sqrt(max(mse, 0.0))
                psnr = float("inf") if rmse == 0 else 20.0 * math.log10(1.0 / rmse)
                sums = self.pixel_sums[region]
                sums["rgb_sse"] += float(np.sum(diff * diff))
                sums["rgb_count"] += int(diff.size)
                lab_sums = _lab_diff_sums(pred, gt, mask, region)
                mae = float("nan")
                lab_rmse = float("nan")
                if lab_sums is not None:
                    lab_sae, lab_sse, lab_pixels = lab_sums
                    sums["lab_sae"] += lab_sae
                    sums["lab_sse"] += lab_sse
                    sums["lab_pixels"] += lab_pixels
                    mae = lab_sae / lab_pixels
                    lab_rmse = math.sqrt(max(lab_sse / lab_pixels, 0.0))
                region_mask = _region_mask(mask, region)
                ssim = _ssim(pred, gt, region_mask)
                lpips_value = self._lpips(pred, gt, region_mask)
                self.rows[region]["psnr"].append(psnr)
                self.rows[region]["ssim"].append(ssim)
                self.rows[region]["lpips"].append(lpips_value)
                self.rows[region]["mae"].append(mae)
                self.rows[region]["rmse"].append(lab_rmse)

    def _lpips(self, pred, gt, mask):
        if not self.include_lpips or self.lpips_net is None:
            return float("nan")
        pred_tensor = pil_to_tensor(Image.fromarray((np.clip(pred, 0, 1) * 255).astype(np.uint8))).float()
        gt_tensor = pil_to_tensor(Image.fromarray((np.clip(gt, 0, 1) * 255).astype(np.uint8))).float()
        pred_tensor = pred_tensor.unsqueeze(0).to(self.lpips_device) / 127.5 - 1.0
        gt_tensor = gt_tensor.unsqueeze(0).to(self.lpips_device) / 127.5 - 1.0
        with torch.no_grad():
            value = self.lpips_net(pred_tensor, gt_tensor)
        if value.numel() == 1:
            return float(value.item())
        value_map = value.detach().float().cpu().squeeze().numpy()
        if value_map.ndim == 0:
            return float(value_map)
        if value_map.ndim == 3:
            value_map = value_map.mean(axis=0)
        if value_map.shape != mask.shape:
            if cv2 is not None:
                mask_resized = cv2.resize(mask.astype(np.uint8), (value_map.shape[1], value_map.shape[0]), interpolation=cv2.INTER_NEAREST) > 0
            else:
                img = Image.fromarray(mask.astype(np.uint8) * 255)
                mask_resized = np.asarray(img.resize((value_map.shape[1], value_map.shape[0]), Image.Resampling.NEAREST)) > 0
        else:
            mask_resized = mask
        if not mask_resized.any():
            return float("nan")
        return float(np.mean(value_map[mask_resized]))

    def summary(self):
        out = {}
        for region, metrics in self.rows.items():
            out[region] = {"pixels": 0, "pooled_psnr": float("nan"), "pooled_mae": float("nan"), "pooled_rmse": float("nan")}
            sums = self.pixel_sums[region]
            out[region]["pixels"] = sums["rgb_count"] // 3
            if sums["rgb_count"] > 0:
                mse = sums["rgb_sse"] / sums["rgb_count"]
                out[region]["pooled_psnr"] = float("inf") if mse == 0 else 10.0 * math.log10(1.0 / mse)
            if sums["lab_pixels"] > 0:
                out[region]["pooled_mae"] = sums["lab_sae"] / sums["lab_pixels"]
                out[region]["pooled_rmse"] = math.sqrt(max(sums["lab_sse"] / sums["lab_pixels"], 0.0))
            for name, values in metrics.items():
                finite = [v for v in values if not math.isnan(v)]
                out[region][name] = float(np.mean(finite)) if finite else float("nan")
        return out


def format_region_summary(summary):
    lines = []
    for region in REGIONS:
        row = summary[region]
        lines.append(
            f"{region}: PSNR {row['psnr']:.4f} | SSIM {row['ssim']:.4f} | "
            f"LPIPS {row['lpips']:.4f} | MAE {row['mae']:.6f} | RMSE {row['rmse']:.6f} | pixels {row['pixels']}"
        )
    return lines


def save_region_metrics_csv(path, summary):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["region", "psnr", "ssim", "mae", "rmse", "lpips", "pixels", "pooled_psnr", "pooled_mae", "pooled_rmse"])
        for display_region, csv_region in (
                ("Shadow", "shadow"),
                ("non-Shadow", "nonshadow"),
                ("all", "all")):
            row = summary[display_region]
            writer.writerow([
                csv_region,
                "{:.6f}".format(row["psnr"]),
                "{:.6f}".format(row["ssim"]),
                "{:.6f}".format(row["mae"]),
                "{:.6f}".format(row["rmse"]),
                "{:.6f}".format(row["lpips"]),
                str(int(row.get("pixels", 0))),
                "{:.6f}".format(row.get("pooled_psnr", float("nan"))),
                "{:.6f}".format(row.get("pooled_mae", float("nan"))),
                "{:.6f}".format(row.get("pooled_rmse", float("nan"))),
            ])


def _load_rgb(path):
    return np.asarray(Image.open(path).convert("RGB"), dtype=np.float32) / 255.0


def evaluate_dirs(pred_dir, gt_dir, mask_dir, max_images=0, device="cuda", include_lpips=True, lpips_device=None):
    pred_dir = Path(pred_dir)
    gt_dir = Path(gt_dir)
    mask_dir = Path(mask_dir)
    acc = RegionMetricAccumulator.build(device=device, include_lpips=include_lpips, lpips_device=lpips_device)
    files = sorted([p for p in pred_dir.iterdir() if p.suffix.lower() in {".png", ".jpg", ".jpeg"}])
    if max_images > 0:
        files = files[:max_images]
    matched = 0
    for pred_path in files:
        gt_path = gt_dir / pred_path.name
        mask_path = mask_dir / pred_path.name
        if not gt_path.exists() or not mask_path.exists():
            continue
        pred = Image.open(pred_path).convert("RGB")
        gt = torch.from_numpy(_load_rgb(gt_path).transpose(2, 0, 1))
        mask = Image.open(mask_path).convert("L")
        acc.update(gt.unsqueeze(0), [pred], [mask])
        matched += 1
    return acc.summary(), matched


def main():
    parser = argparse.ArgumentParser(
        description="StableShadowRemoval-compatible DESOBA v2 Shadow/nonshadow/all metrics.")
    parser.add_argument("--pred_dir", required=True)
    parser.add_argument("--gt_dir", required=True)
    parser.add_argument("--mask_dir", required=True)
    parser.add_argument("--save_csv", default="")
    parser.add_argument("--max_images", type=int, default=0)
    parser.add_argument("--gpus", default="0")
    parser.add_argument("--skip_lpips", action="store_true")
    parser.add_argument("--lpips_device", default="")
    args = parser.parse_args()

    if args.gpus.lower() == "cpu":
        device = "cpu"
    else:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    lpips_device = args.lpips_device or device
    summary, count = evaluate_dirs(
        args.pred_dir,
        args.gt_dir,
        args.mask_dir,
        args.max_images,
        device=device,
        include_lpips=not args.skip_lpips,
        lpips_device=lpips_device,
    )
    print("===> Number of images:", count)
    for line in format_region_summary(summary):
        print(line)
    if args.save_csv:
        save_region_metrics_csv(args.save_csv, summary)
        print("Saved CSV:", args.save_csv)


if __name__ == "__main__":
    main()
