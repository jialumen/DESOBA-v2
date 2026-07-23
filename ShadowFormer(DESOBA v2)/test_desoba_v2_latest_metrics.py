
import numpy as np
import os,sys
import argparse
from tqdm import tqdm
from einops import rearrange, repeat

import torch.nn as nn
import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F
# from ptflops import get_model_complexity_info

import scipy.io as sio
from utils.loader import get_validation_data
import utils
import cv2
from model import UNet

from skimage import img_as_float32, img_as_ubyte
from utils.eval_metrics import compute_region_metrics, format_metric_table, mean_metric_lists, merge_metric_lists, METRIC_KEYS

parser = argparse.ArgumentParser(description='ShadowFormer DESOBA v2 evaluation with Shadow/non-Shadow/all metrics')
parser.add_argument('--input_dir', default='./data/DESOBAv2_test/',
    type=str, help='Directory of DESOBA v2 validation images')
parser.add_argument('--result_dir', default='./results/',
    type=str, help='Directory for results')
parser.add_argument('--weights', default='best_path.pth',
    type=str, help='Path to weights')
parser.add_argument('--gpus', default='0', type=str, help='CUDA_VISIBLE_DEVICES')
parser.add_argument('--arch', default='ShadowFormer', type=str, help='arch')
parser.add_argument('--batch_size', default=1, type=int, help='Batch size for dataloader')
parser.add_argument('--num_workers', default=0, type=int, help='Number of dataloader workers')
parser.add_argument('--save_images', action='store_true', help='Save denoised images in result directory')
parser.add_argument('--cal_metrics', action='store_true', help='Measure denoised images with GT')
parser.add_argument('--embed_dim', type=int, default=32, help='number of data loading workers')    
parser.add_argument('--win_size', type=int, default=8, help='window size')
parser.add_argument('--token_projection', type=str, default='linear', help='linear/conv token projection')
parser.add_argument('--token_mlp', type=str,default='leff', help='ffn/leff token mlp')
# args for vit
parser.add_argument('--vit_dim', type=int, default=256, help='vit hidden_dim')
parser.add_argument('--vit_depth', type=int, default=12, help='vit depth')
parser.add_argument('--vit_nheads', type=int, default=8, help='vit hidden_dim')
parser.add_argument('--vit_mlp_dim', type=int, default=512, help='vit mlp_dim')
parser.add_argument('--vit_patch_size', type=int, default=16, help='vit patch_size')
parser.add_argument('--global_skip', action='store_true', default=False, help='global skip connection')
parser.add_argument('--local_skip', action='store_true', default=False, help='local skip connection')
parser.add_argument('--vit_share', action='store_true', default=False, help='share vit module')
parser.add_argument('--train_ps', type=int, default=320, help='patch size of training sample')
parser.add_argument('--tile', type=int, default=None, help='Tile size (e.g 720). None means testing on the original resolution image')
parser.add_argument('--tile_overlap', type=int, default=32, help='Overlapping of different tiles')
args = parser.parse_args()


os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus

utils.mkdir(args.result_dir)

test_dataset = get_validation_data(args.input_dir)
test_loader = DataLoader(dataset=test_dataset, batch_size=1, shuffle=False, num_workers=args.num_workers, drop_last=False)

model_restoration = utils.get_arch(args)
model_restoration = torch.nn.DataParallel(model_restoration)

utils.load_checkpoint(model_restoration, args.weights)
print("===>Testing using weights: ", args.weights)

model_restoration.cuda()
model_restoration.eval()

img_multiple_of = 8 * args.win_size

with torch.no_grad():
    metrics = {key: [] for key in METRIC_KEYS}
    for ii, data_test in enumerate(tqdm(test_loader), 0):
        rgb_gt = data_test[0].numpy().squeeze().transpose((1, 2, 0))
        rgb_noisy = data_test[1].cuda()
        mask = data_test[2].cuda()
        filenames = data_test[3]

        # Pad the input if not_multiple_of win_size * 8
        height, width = rgb_noisy.shape[2], rgb_noisy.shape[3]
        H, W = ((height + img_multiple_of) // img_multiple_of) * img_multiple_of, (
                    (width + img_multiple_of) // img_multiple_of) * img_multiple_of
        padh = H - height if height % img_multiple_of != 0 else 0
        padw = W - width if width % img_multiple_of != 0 else 0
        rgb_noisy = F.pad(rgb_noisy, (0, padw, 0, padh), 'reflect')
        mask = F.pad(mask, (0, padw, 0, padh), 'reflect')

        if args.tile is None:
            rgb_restored = model_restoration(rgb_noisy, mask)
        else:
            # test the image tile by tile
            b, c, h, w = rgb_noisy.shape
            tile = min(args.tile, h, w)
            assert tile % 8 == 0, "tile size should be multiple of 8"
            tile_overlap = args.tile_overlap

            stride = tile - tile_overlap
            h_idx_list = list(range(0, h - tile, stride)) + [h - tile]
            w_idx_list = list(range(0, w - tile, stride)) + [w - tile]
            E = torch.zeros(b, c, h, w).type_as(rgb_noisy)
            W = torch.zeros_like(E)

            for h_idx in h_idx_list:
                for w_idx in w_idx_list:
                    in_patch = rgb_noisy[..., h_idx:h_idx + tile, w_idx:w_idx + tile]
                    mask_patch = mask[..., h_idx:h_idx + tile, w_idx:w_idx + tile]
                    out_patch = model_restoration(in_patch, mask_patch)
                    out_patch_mask = torch.ones_like(out_patch)

                    E[..., h_idx:(h_idx + tile), w_idx:(w_idx + tile)].add_(out_patch)
                    W[..., h_idx:(h_idx + tile), w_idx:(w_idx + tile)].add_(out_patch_mask)
            rgb_restored = E.div_(W)

        rgb_restored = torch.clamp(rgb_restored, 0, 1).cpu().numpy().squeeze().transpose((1, 2, 0))

        # Unpad the output
        rgb_restored = rgb_restored[:height, :width, :]

        if args.cal_metrics:
            sample = compute_region_metrics(
                rgb_restored,
                rgb_gt,
                mask.cpu().numpy().squeeze(),
                size=256)
            merge_metric_lists(metrics, sample)

        if args.save_images:
            utils.save_img(rgb_restored*255.0, os.path.join(args.result_dir, filenames[0]))

if args.cal_metrics:
    mean_metrics = mean_metric_lists(metrics)
    print(format_metric_table("ShadowFormer", mean_metrics))

