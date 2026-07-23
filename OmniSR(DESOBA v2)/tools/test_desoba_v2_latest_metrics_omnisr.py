import argparse
import csv
import math
import os
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from skimage.metrics import peak_signal_noise_ratio as psnr_loss
from skimage.metrics import structural_similarity as ssim_loss
from torch.utils.data import DataLoader
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import utils
from utils.eval_metrics import (
    compute_paper_metrics as paper_compute_paper_metrics,
    dilate_shadow_mask,
    format_metric_table as paper_format_metric_table,
    legacy_metric_lines as paper_legacy_metric_lines,
)
from utils.loader import get_validation_data


METRIC_KEYS = (
    "psnr_all", "ssim_all", "mae_all", "rmse_all", "lpips_all",
    "psnr_shadow", "ssim_shadow", "mae_shadow", "rmse_shadow", "lpips_shadow",
    "psnr_nonshadow", "ssim_nonshadow", "mae_nonshadow", "rmse_nonshadow", "lpips_nonshadow",
)

def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate OmniSR on DESOBA v2 with Shadow/Non-shadow/All metrics.")
    parser.add_argument("--weights", default="auto")
    parser.add_argument("--weights_root", action="append", default=None)
    parser.add_argument("--prefer_checkpoint_name", default="model_latest.pth,model_best.pth",
                        help="Comma-separated checkpoint names for --weights auto. Latest is preferred by default.")
    parser.add_argument("--input_dir", default="./data/DESOBAv2/test")
    parser.add_argument("--mask_dir", default="./data/DESOBAv2/test/shadow_mask")
    parser.add_argument("--result_dir", default="./outputs/OmniSR_DESOBAv2_test")
    parser.add_argument("--save_images", action="store_true")
    parser.add_argument("--save_csv", default="")
    parser.add_argument("--save_per_image_csv", default="")
    parser.add_argument("--metric_name", default="")
    parser.add_argument("--filename_source", default="auto", choices=["auto", "clean", "noisy"],
                        help="Which validation filename to use for mask lookup. Auto tries clean then noisy.")
    parser.add_argument("--max_images", type=int, default=0)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--gpus", default="0")
    parser.add_argument("--skip_lpips", action="store_true")
    parser.add_argument("--non_strict_checkpoint", action="store_true",
                        help="Allow partial checkpoint loading. Default is strict to catch wrong weights.")
    parser.add_argument("--metric_protocol", default="paper", choices=["paper", "train"],
                        help="paper matches the provided test_desoba_v2_latest_metrics.py; train matches train_DDP's internal validation.")
    parser.add_argument("--shadow_mask_dilate", type=int, default=0,
                        help="Odd dilation kernel size for visual/transition-region metrics. 0 keeps strict original masks.")
    parser.add_argument("--arch", default="ShadowFormer")
    parser.add_argument("--embed_dim", type=int, default=32)
    parser.add_argument("--win_size", type=int, default=16)
    parser.add_argument("--tile", type=int, default=256,
                        help="Tile size for evaluation inference. 0 disables tiling.")
    parser.add_argument("--tile_overlap", type=int, default=64,
                        help="Tile overlap for evaluation inference.")
    parser.add_argument("--token_projection", default="linear")
    parser.add_argument("--token_mlp", default="leff")
    parser.add_argument("--train_ps", type=int, default=256)
    return parser.parse_args()


def resolve_weight_path(args):
    if args.weights.lower() != "auto":
        path = Path(args.weights)
        if not path.exists():
            raise FileNotFoundError(path)
        return path
    roots = [Path(root) for root in (args.weights_root or ["./log"])]
    candidates = []
    for root in roots:
        if not root.exists():
            continue
        candidates.extend([p for p in root.rglob("*.pth") if p.stat().st_size > 1024 * 1024])
    if not candidates:
        raise FileNotFoundError("No checkpoints found under {}".format(", ".join(map(str, roots))))
    preferred = [name.strip().lower() for name in args.prefer_checkpoint_name.split(",") if name.strip()]
    for name in preferred:
        hits = [p for p in candidates if p.name.lower() == name]
        if hits:
            return max(hits, key=lambda p: p.stat().st_mtime)
    return max(candidates, key=lambda p: p.stat().st_mtime)


def mean_metrics(rows):
    out = {}
    for key in METRIC_KEYS:
        vals = np.asarray([row[key] for row in rows], dtype=np.float64)
        vals = vals[np.isfinite(vals)]
        out[key] = float(vals.mean()) if vals.size else float("nan")
    return out


def save_metrics_csv(path, metrics):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["region", "psnr", "ssim", "mae", "rmse", "lpips"])
        for region in ["shadow", "nonshadow", "all"]:
            writer.writerow([
                region,
                "{:.6f}".format(metrics["psnr_" + region]),
                "{:.6f}".format(metrics["ssim_" + region]),
                "{:.6f}".format(metrics["mae_" + region]),
                "{:.6f}".format(metrics["rmse_" + region]),
                "{:.6f}".format(metrics["lpips_" + region]),
            ])


def save_per_image_csv(path, rows):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fields = ["image_id"] + list(METRIC_KEYS)
    if any(("visual_" + key) in row for row in rows for key in METRIC_KEYS):
        fields.extend(["visual_" + key for key in METRIC_KEYS])
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def choose_mask_path(mask_dir, clean_filename, noisy_filename, filename_source):
    mask_root = Path(mask_dir)
    clean_path = mask_root / clean_filename
    noisy_path = mask_root / noisy_filename
    if filename_source == "clean":
        return clean_path, clean_filename, "clean"
    if filename_source == "noisy":
        return noisy_path, noisy_filename, "noisy"
    if clean_path.exists():
        return clean_path, clean_filename, "clean"
    return noisy_path, noisy_filename, "noisy"


_LPIPS_MODEL = None


def get_lpips_model(device):
    global _LPIPS_MODEL
    if _LPIPS_MODEL is None:
        import lpips
        _LPIPS_MODEL = lpips.LPIPS(net="alex").to(device).eval()
    return _LPIPS_MODEL


def compute_train_metric_shadow_mask(restored, target, mask, lpips_model):
    from utils.eval_metrics import compute_training_region_metrics
    return compute_training_region_metrics(restored, target, mask, lpips_model=lpips_model)


def flatten_training_metrics(metrics):
    return {
        "psnr_shadow": metrics["Shadow"]["PSNR"],
        "ssim_shadow": metrics["Shadow"]["SSIM"],
        "mae_shadow": metrics["Shadow"]["MAE"],
        "rmse_shadow": metrics["Shadow"]["RMSE"],
        "lpips_shadow": metrics["Shadow"]["LPIPS"],
        "psnr_nonshadow": metrics["non-Shadow"]["PSNR"],
        "ssim_nonshadow": metrics["non-Shadow"]["SSIM"],
        "mae_nonshadow": metrics["non-Shadow"]["MAE"],
        "rmse_nonshadow": metrics["non-Shadow"]["RMSE"],
        "lpips_nonshadow": metrics["non-Shadow"]["LPIPS"],
        "psnr_all": metrics["all"]["PSNR"],
        "ssim_all": metrics["all"]["SSIM"],
        "mae_all": metrics["all"]["MAE"],
        "rmse_all": metrics["all"]["RMSE"],
        "lpips_all": metrics["all"]["LPIPS"],
    }


def format_training_metric_table(method_name, metrics):
    return "\n".join([
        "Metric table (train_DDP validation protocol: region pixels for PSNR/MAE/RMSE; isolated crop SSIM; full-image LPIPS):",
        "{:<28} | {:^45} | {:^49} | {:^45}".format("Method", "Shadow Region", "Non-Shadow Region", "All Region"),
        "{:<28} | {:>8} {:>8} {:>8} {:>8} {:>8} | {:>8} {:>8} {:>8} {:>8} {:>8} | {:>8} {:>8} {:>8} {:>8} {:>8}".format(
            "", "PSNR", "SSIM", "MAE", "RMSE", "LPIPS",
            "PSNR", "SSIM", "MAE", "RMSE", "LPIPS",
            "PSNR", "SSIM", "MAE", "RMSE", "LPIPS"),
        "-" * 181,
        "{:<28} | {:8.2f} {:8.3f} {:8.2f} {:8.2f} {:8.4f} | {:8.2f} {:8.3f} {:8.2f} {:8.2f} {:8.4f} | {:8.2f} {:8.3f} {:8.2f} {:8.2f} {:8.4f}".format(
            method_name,
            metrics["psnr_shadow"], metrics["ssim_shadow"], metrics["mae_shadow"], metrics["rmse_shadow"], metrics["lpips_shadow"],
            metrics["psnr_nonshadow"], metrics["ssim_nonshadow"], metrics["mae_nonshadow"], metrics["rmse_nonshadow"], metrics["lpips_nonshadow"],
            metrics["psnr_all"], metrics["ssim_all"], metrics["mae_all"], metrics["rmse_all"], metrics["lpips_all"]),
    ])


def upsample_for_dino(img):
    return nn.UpsamplingBilinear2d(size=(int(img.shape[2] * 14 / 8), int(img.shape[3] * 14 / 8)))(img)


def _pad_to_multiple(tensor, multiple):
    height, width = tensor.shape[-2:]
    h_pad = ((height + multiple - 1) // multiple) * multiple - height
    w_pad = ((width + multiple - 1) // multiple) * multiple - width
    if h_pad or w_pad:
        tensor = F.pad(tensor, (0, w_pad, 0, h_pad), "reflect")
    return tensor, height, width


def _starts(length, tile, overlap):
    if length <= tile:
        return [0]
    step = max(1, tile - overlap)
    starts = list(range(0, max(length - tile, 0) + 1, step))
    last = length - tile
    if starts[-1] != last:
        starts.append(last)
    return starts


def _infer_one(model, dino, input_, point, normal, device, img_multiple_of):
    input_pad, height, width = _pad_to_multiple(input_, img_multiple_of)
    point_pad = F.pad(point, (0, input_pad.shape[-1] - point.shape[-1], 0, input_pad.shape[-2] - point.shape[-2]), "reflect") \
        if input_pad.shape[-2:] != point.shape[-2:] else point
    normal_pad = F.pad(normal, (0, input_pad.shape[-1] - normal.shape[-1], 0, input_pad.shape[-2] - normal.shape[-2]), "reflect") \
        if input_pad.shape[-2:] != normal.shape[-2:] else normal
    dino_features = dino.get_intermediate_layers(upsample_for_dino(input_pad), 4, True)
    with torch.cuda.amp.autocast(enabled=(device.type == "cuda")):
        restored = model(input_pad, dino_features, point_pad, normal_pad)
    return torch.clamp(restored[:, :, :height, :width], 0.0, 1.0)


def tiled_inference(model, dino, input_, point, normal, device, img_multiple_of, tile=256, overlap=64):
    height, width = input_.shape[-2:]
    if tile <= 0 or (height <= tile and width <= tile):
        return _infer_one(model, dino, input_, point, normal, device, img_multiple_of)
    tile = int(tile)
    overlap = max(0, min(int(overlap), tile - 1))
    output = torch.zeros((input_.shape[0], input_.shape[1], height, width), device=device)
    weight = torch.zeros((input_.shape[0], 1, height, width), device=device)
    for y in _starts(height, tile, overlap):
        y1 = min(y + tile, height)
        y0 = max(0, y1 - tile)
        for x in _starts(width, tile, overlap):
            x1 = min(x + tile, width)
            x0 = max(0, x1 - tile)
            tile_out = _infer_one(
                model, dino,
                input_[:, :, y0:y1, x0:x1],
                point[:, :, y0:y1, x0:x1],
                normal[:, :, y0:y1, x0:x1],
                device, img_multiple_of)
            output[:, :, y0:y1, x0:x1] += tile_out
            weight[:, :, y0:y1, x0:x1] += 1.0
    return torch.clamp(output / weight.clamp_min(1.0), 0.0, 1.0)


def main():
    args = parse_args()
    if args.gpus.lower() == "cpu":
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
    else:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus
    device = torch.device("cuda" if torch.cuda.is_available() and args.gpus.lower() != "cpu" else "cpu")
    weight_path = resolve_weight_path(args)
    print("===> Testing using weights:", weight_path)

    dataset = get_validation_data(args.input_dir, False)
    loader = DataLoader(dataset=dataset, batch_size=1, shuffle=False, num_workers=args.num_workers, drop_last=False)
    model = utils.get_arch(args).to(device).eval()
    utils.load_checkpoint(model, str(weight_path), strict=not args.non_strict_checkpoint)
    dino = torch.hub.load("./dinov2", "dinov2_vitl14", source="local").to(device).eval()

    img_multiple_of = 8 * args.win_size
    Path(args.result_dir).mkdir(parents=True, exist_ok=True)
    rows = []
    diagnostics = {
        "name_mismatch": 0,
        "mask_source_clean": 0,
        "mask_source_noisy": 0,
        "mask_ratio": [],
        "mask_ratio_dilated": [],
        "input_gt_white_error": [],
        "input_gt_black_error": [],
    }
    lpips_model = get_lpips_model(device) if (args.metric_protocol == "train" and not args.skip_lpips) else None
    with torch.no_grad():
        for idx, data in enumerate(tqdm(loader), 1):
            target = data[0].to(device)
            input_ = data[1].to(device)
            point = data[2].to(device)
            normal = data[3].to(device)
            clean_filename = data[4][0]
            noisy_filename = data[5][0] if len(data) > 5 else clean_filename
            if clean_filename != noisy_filename:
                diagnostics["name_mismatch"] += 1
            mask_path, filename, mask_source = choose_mask_path(
                args.mask_dir, clean_filename, noisy_filename, args.filename_source)
            diagnostics["mask_source_" + mask_source] += 1
            height, width = input_.shape[2], input_.shape[3]
            restored = tiled_inference(
                model, dino, input_, point, normal, device, img_multiple_of,
                tile=args.tile, overlap=args.tile_overlap)
            restored_np = restored.cpu().numpy().squeeze().transpose(1, 2, 0)
            target_np = target.cpu().numpy().squeeze().transpose(1, 2, 0)
            mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
            if mask is None:
                raise FileNotFoundError(mask_path)
            mask = (mask > 127).astype(np.float32)
            diagnostics["mask_ratio"].append(float(mask.mean()))
            input_np = input_.detach().cpu().numpy().squeeze().transpose(1, 2, 0)
            if mask.shape[:2] != target_np.shape[:2]:
                mask_for_diag = cv2.resize(mask, (target_np.shape[1], target_np.shape[0]),
                                           interpolation=cv2.INTER_NEAREST).astype(bool)
            else:
                mask_for_diag = mask.astype(bool)
            diff = np.mean(np.abs(input_np - target_np), axis=2)
            if np.any(mask_for_diag):
                diagnostics["input_gt_white_error"].append(float(diff[mask_for_diag].mean()))
            if np.any(~mask_for_diag):
                diagnostics["input_gt_black_error"].append(float(diff[~mask_for_diag].mean()))
            if args.metric_protocol == "paper":
                sample = paper_compute_paper_metrics(restored_np, target_np, mask, include_lpips=not args.skip_lpips)
            else:
                sample = compute_train_metric_shadow_mask(restored_np, target_np, mask, lpips_model=lpips_model)
                sample = flatten_training_metrics(sample)
            if args.shadow_mask_dilate > 1:
                mask_dilated = dilate_shadow_mask(mask, args.shadow_mask_dilate)
                diagnostics["mask_ratio_dilated"].append(float(mask_dilated.mean()))
                if args.metric_protocol == "paper":
                    sample_dilated = paper_compute_paper_metrics(
                        restored_np, target_np, mask_dilated, include_lpips=not args.skip_lpips)
                else:
                    sample_dilated = compute_train_metric_shadow_mask(
                        restored_np, target_np, mask_dilated, lpips_model=lpips_model)
                    sample_dilated = flatten_training_metrics(sample_dilated)
            else:
                sample_dilated = None
            row = {"image_id": Path(filename).stem}
            row.update(sample)
            if sample_dilated is not None:
                for key, value in sample_dilated.items():
                    row["visual_" + key] = value
            rows.append(row)
            if args.save_images:
                utils.save_img(np.clip(restored_np * 255.0, 0, 255).round().astype(np.uint8),
                               str(Path(args.result_dir) / filename))
            if args.max_images > 0 and idx >= args.max_images:
                break

    metrics = mean_metrics(rows)
    visual_metrics = None
    if args.shadow_mask_dilate > 1:
        visual_metrics = {}
        for key in METRIC_KEYS:
            vals = np.asarray([row.get("visual_" + key, np.nan) for row in rows], dtype=np.float64)
            vals = vals[np.isfinite(vals)]
            visual_metrics[key] = float(vals.mean()) if vals.size else float("nan")
    if diagnostics["mask_ratio"]:
        ratio = np.asarray(diagnostics["mask_ratio"], dtype=np.float64)
        ratio_dilated = np.asarray(diagnostics["mask_ratio_dilated"], dtype=np.float64)
        white_error = np.asarray(diagnostics["input_gt_white_error"], dtype=np.float64)
        black_error = np.asarray(diagnostics["input_gt_black_error"], dtype=np.float64)
        print("Diagnostics:")
        print("  filename mismatches clean/noisy: {}".format(diagnostics["name_mismatch"]))
        print("  mask source clean/noisy: {}/{}".format(
            diagnostics["mask_source_clean"], diagnostics["mask_source_noisy"]))
        print("  mask white ratio mean/min/max: {:.6f}/{:.6f}/{:.6f}".format(
            float(np.nanmean(ratio)), float(np.nanmin(ratio)), float(np.nanmax(ratio))))
        if ratio_dilated.size:
            print("  dilated mask ratio mean/min/max (kernel {}): {:.6f}/{:.6f}/{:.6f}".format(
                args.shadow_mask_dilate,
                float(np.nanmean(ratio_dilated)), float(np.nanmin(ratio_dilated)), float(np.nanmax(ratio_dilated))))
        print("  input-vs-gt error white/black: {:.6f}/{:.6f}".format(
            float(np.nanmean(white_error)) if white_error.size else float("nan"),
            float(np.nanmean(black_error)) if black_error.size else float("nan")))
        if white_error.size and black_error.size and np.nanmean(white_error) < np.nanmean(black_error):
            print("  WARNING: mask polarity may be inverted; white-mask input error is lower than black-mask error.")
    for line in paper_legacy_metric_lines(metrics):
        print(line)
    print("")
    if args.metric_protocol == "paper":
        print(paper_format_metric_table((args.metric_name or weight_path.stem) + "_strict", metrics))
        if visual_metrics is not None:
            print("")
            print(paper_format_metric_table(
                "{}_visual_dilate{}".format(args.metric_name or weight_path.stem, args.shadow_mask_dilate),
                visual_metrics))
    else:
        print(format_training_metric_table((args.metric_name or weight_path.stem) + "_strict", metrics))
        if visual_metrics is not None:
            print("")
            print(format_training_metric_table(
                "{}_visual_dilate{}".format(args.metric_name or weight_path.stem, args.shadow_mask_dilate),
                visual_metrics))
    if args.save_csv:
        save_metrics_csv(args.save_csv, metrics)
        print("Saved CSV:", args.save_csv)
    if args.save_per_image_csv:
        save_per_image_csv(args.save_per_image_csv, rows)
        print("Saved per-image CSV:", args.save_per_image_csv)


if __name__ == "__main__":
    main()
