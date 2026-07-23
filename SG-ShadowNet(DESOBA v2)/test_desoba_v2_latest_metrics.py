import argparse
import csv
import os

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from data.desoba_v2 import DESOBAv2Dataset, lab_tensor_to_rgb
from models.model import ConGenerator_S2F, ConRefineNet
from utils.eval_metrics import METRIC_KEYS, compute_paper_metrics, format_metric_table, mean_metric_lists, merge_metric_lists


def load_models(weights, device):
    checkpoint = torch.load(weights, map_location="cpu")
    net_g1, net_g2 = ConGenerator_S2F(), ConRefineNet()
    if isinstance(checkpoint, dict) and "netG_1" in checkpoint:
        net_g1.load_state_dict(checkpoint["netG_1"])
        net_g2.load_state_dict(checkpoint["netG_2"])
    else:
        raise RuntimeError("Expected a SG checkpoint containing netG_1 and netG_2: {}".format(weights))
    return net_g1.to(device).eval(), net_g2.to(device).eval()


def evaluate(net_g1, net_g2, loader, device, max_images=0, progress=True):
    metric_lists = {key: [] for key in METRIC_KEYS}
    seen = 0
    iterator = tqdm(loader, desc="DESOBA v2 evaluation", dynamic_ncols=True) if progress else loader
    net_g1.eval()
    net_g2.eval()
    with torch.no_grad():
        for batch in iterator:
            shadow = batch["shadow"].to(device, non_blocking=True)
            target = batch["target"]
            mask = batch["mask"].to(device, non_blocking=True)
            coarse = net_g1(shadow, mask)
            refined_input = shadow * (1.0 - mask) + coarse * mask
            prediction = net_g2(refined_input, mask)
            for i in range(prediction.shape[0]):
                sample = compute_paper_metrics(
                    lab_tensor_to_rgb(prediction[i]),
                    lab_tensor_to_rgb(target[i]),
                    batch["mask"][i].numpy().squeeze(),
                    size=256,
                )
                merge_metric_lists(metric_lists, sample)
                seen += 1
                if max_images and seen >= max_images:
                    return mean_metric_lists(metric_lists)
    return mean_metric_lists(metric_lists)


def save_metrics_csv(path, metrics):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["region", "psnr", "ssim", "mae", "rmse", "lpips"])
        for region in ("shadow", "nonshadow", "all"):
            writer.writerow([region] + ["{:.6f}".format(metrics["{}_{}".format(key, region)])
                                        for key in ("psnr", "ssim", "mae", "rmse", "lpips")])


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate SG-ShadowNet on DESOBA v2 with the reference protocol.")
    parser.add_argument("--weights", default="best_path.pth")
    parser.add_argument("--annotation_root", default="./data/DESOBAv2_extended_annotations")
    parser.add_argument("--image_root", default="./data/DESOBAv2_work")
    parser.add_argument("--split", default="test")
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--max_images", type=int, default=0)
    parser.add_argument("--save_csv", default="")
    parser.add_argument("--metric_name", default="SG-ShadowNet")
    return parser.parse_args()


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset = DESOBAv2Dataset(args.annotation_root, args.image_root, split=args.split, augment=False)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers,
                        pin_memory=device.type == "cuda")
    net_g1, net_g2 = load_models(args.weights, device)
    metrics = evaluate(net_g1, net_g2, loader, device, max_images=args.max_images)
    print(format_metric_table(args.metric_name, metrics), flush=True)
    if args.save_csv:
        save_metrics_csv(args.save_csv, metrics)


if __name__ == "__main__":
    main()
