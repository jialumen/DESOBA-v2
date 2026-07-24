import argparse
import csv
import json
import math
import os
import random
from itertools import islice
from pathlib import Path

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import numpy as np
import cv2
import torch
import torch.nn.functional as F
from skimage import img_as_ubyte
from torch.utils.data import DataLoader
from tqdm import tqdm

import utils
from relation_prior import RELATION_META_KEYS, RELATION_SCALAR_KEYS, RELATION_SPATIAL_KEYS
from utils.eval_metrics import (compute_paper_metrics, format_metric_table,
                                legacy_metric_lines, mean_metric_lists,
                                merge_metric_lists)
from utils.image_utils import mergeimage, splitimage
from utils.loader import get_shadow_relation_data


CHECKPOINT_SUFFIXES = {".pth", ".pt", ".ckpt"}
DEFAULT_EXCLUDE_NAMES = {"istd_p.pth", "srd.pth"}


def compute_paper_metrics_compat(pred_rgb, gt_rgb, mask, include_lpips=True):
    try:
        return compute_paper_metrics(
            pred_rgb, gt_rgb, mask, include_lpips=include_lpips)
    except TypeError as exc:
        if "include_lpips" not in str(exc):
            raise
        sample = compute_paper_metrics(pred_rgb, gt_rgb, mask)
        if not include_lpips and "lpips_shadow" not in sample:
            sample["lpips_shadow"] = float("nan")
        return sample


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate latest HomoFormer DESOBA v2 checkpoint with Shadow/Non-shadow/All metrics.")
    parser.add_argument("--weights", default="auto",
                        help="Checkpoint path. Use auto to pick the latest checkpoint under --weights_root.")
    parser.add_argument("--weights_root", action="append", default=None,
                        help="Root directory to search when --weights auto. Can be passed multiple times.")
    parser.add_argument("--prefer_checkpoint_name", default="model_best_paper.pth,model_best.pth,model_latest.pth",
                        help="Comma-separated checkpoint names to prefer while searching; empty disables this preference.")
    parser.add_argument("--allow_base_checkpoint", action="store_true",
                        help="Allow checkpoints without relation/DESOBA branch weights.")
    parser.add_argument("--exclude_names", default="ISTD_P.pth,SRD.pth",
                        help="Comma-separated checkpoint file names to ignore during auto search.")

    parser.add_argument("--relation_annotation_root", default="./data/DESOBAv2_extended_annotations",
                        help="DESOBA v2 relation annotation root.")
    parser.add_argument("--relation_image_root", default="./data/DESOBAv2_work/release256",
                        help="DESOBA v2 image root.")
    parser.add_argument("--split", default="test", help="DESOBA v2 split to evaluate.")
    parser.add_argument("--result_dir", default="./results_desoba_v2_latest",
                        help="Directory for optional output images.")
    parser.add_argument("--save_images", action="store_true", help="Save restored images.")
    parser.add_argument("--save_csv", default="", help="Optional CSV path for final metrics.")
    parser.add_argument("--save_per_image_csv", default="",
                        help="Optional CSV path for per-image metrics and v68 activation statistics.")
    parser.add_argument("--save_v68_aux", action="store_true",
                        help="Save merged/TTA-averaged v68 gates, residuals, masks, and error maps.")
    parser.add_argument("--save_v68_aux_hard_only", action="store_true",
                        help="With --save_v68_aux, save images only for --hard_case_ids.")
    parser.add_argument("--roi_manifest", default="",
                        help="JSON manifest containing fixed hard-case ROI coordinates.")
    parser.add_argument("--build_roi_manifest", action="store_true",
                        help="Build --roi_manifest from this run for the fixed hard cases.")
    parser.add_argument("--hard_case_ids", default="13977_0,13225_0",
                        help="Comma-separated image stems used for ROI construction and reporting.")
    parser.add_argument("--eval_image_ids", default="",
                        help="Optional comma-separated image ids/stems to evaluate instead of the full split.")
    parser.add_argument("--eval_image_list", default="",
                        help="Optional TSV/text image allow-list; useful for internal validation manifests.")
    parser.add_argument("--roi_size", type=int, default=96)
    parser.add_argument("--roi_count", type=int, default=3)
    parser.add_argument("--metric_name", default="", help="Method name shown in the metric table.")
    parser.add_argument("--max_images", type=int, default=0, help="Debug only: stop after this many images.")
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--gpus", default="0", help="CUDA_VISIBLE_DEVICES value, or cpu.")
    parser.add_argument("--no_dataparallel", action="store_true")
    parser.add_argument("--seed", type=int, default=123,
                        help="Deterministic evaluation seed shared by dataset and model initialization.")
    parser.add_argument("--non_deterministic_eval", action="store_true",
                        help="Allow non-deterministic CUDA kernels (not valid for strict v68 comparisons).")

    parser.add_argument("--arch", default="HomoFormer")
    parser.add_argument("--embed_dim", type=int, default=32)
    parser.add_argument("--win_size", type=int, default=8)
    parser.add_argument("--token_projection", default="linear")
    parser.add_argument("--token_mlp", default="leff")
    parser.add_argument("--train_ps", type=int, default=256)
    parser.add_argument("--tile", type=int, default=0,
                        help="Tile size. 0 chooses 384/256/128 based on image size.")
    parser.add_argument("--tile_overlap", type=int, default=30)
    parser.add_argument("--repeat", type=int, default=1)
    parser.add_argument("--base_repeat", type=int, default=1)
    parser.add_argument("--tta_d4", action="store_true",
                        help="Average the 8 D4 test-time augmentations.")
    parser.add_argument("--blend_kernels", default="",
                        help="Comma-separated output shadow blend kernels to average at eval time.")
    parser.add_argument("--blend_floors", default="",
                        help="Comma-separated output shadow blend floors to average at eval time.")
    parser.add_argument("--blend_modes", default="",
                        help="Comma-separated output shadow blend modes to average at eval time.")
    parser.add_argument("--skip_lpips", action="store_true",
                        help="Skip LPIPS for faster PSNR/SSIM/MAE/RMSE sweeps.")

    parser.add_argument("--use_priors", action="store_true", default=False)
    parser.add_argument("--prior_root", default="./datasets/priors/SRD")
    parser.add_argument("--prior_quality_root", default="./datasets/SRD_quality_tta")
    parser.add_argument("--prior_semantic_subdir", default="semantic")
    parser.add_argument("--prior_confidence_root", default="")
    parser.add_argument("--prior_embed_dim", type=int, default=16)
    parser.add_argument("--prior_semantic_dim", type=int, default=384)
    parser.add_argument("--prior_semantic_size", type=int, default=32)
    parser.add_argument("--prior_dropout", type=float, default=0.0)
    parser.add_argument("--prior_shadow_floor", type=float, default=0.0)
    parser.add_argument("--prior_token_max_modulation", type=float, default=0.02)
    parser.add_argument("--prior_max_residual", type=float, default=0.05)
    parser.add_argument("--prior_max_modulation", type=float, default=0.02)
    parser.add_argument("--use_illumination_prior", action="store_true", default=False)
    parser.add_argument("--illumination_max_log_ratio", type=float, default=0.8)
    parser.add_argument("--illumination_max_bias", type=float, default=0.03)
    parser.add_argument("--use_shadow_interaction_attention", action="store_true", default=False)
    parser.add_argument("--shadow_interaction_heads", type=int, default=8)
    parser.add_argument("--shadow_interaction_max_modulation", type=float, default=0.03)
    parser.add_argument("--use_shadow_texture_refine", action="store_true", default=False)
    parser.add_argument("--shadow_texture_max_residual", type=float, default=0.12)
    parser.add_argument("--shadow_texture_gain_limit", type=float, default=2.0)
    parser.add_argument("--shadow_texture_boundary_kernel", type=int, default=11)
    parser.add_argument("--shadow_texture_phase_restore_strength", type=float, default=0.0)
    parser.add_argument("--shadow_texture_phase_restore_line_boost", type=float, default=0.0)
    parser.add_argument("--shadow_texture_phase_restore_boundary", type=float, default=0.0)
    parser.add_argument("--shadow_texture_line_gate_bias", type=float, default=-4.2)
    parser.add_argument("--shadow_texture_line_structure_floor", type=float, default=0.0)
    parser.add_argument("--shadow_texture_line_ratio_floor", type=float, default=0.0)
    parser.add_argument("--use_shadow_texture_periodic_inference_prior",
                        action="store_true", default=False)
    parser.add_argument("--shadow_texture_periodic_prior_strength", type=float, default=0.0)
    parser.add_argument("--shadow_texture_periodic_prior_floor", type=float, default=0.0)
    parser.add_argument("--shadow_texture_periodic_prior_gate_bias", type=float, default=-3.2)
    parser.add_argument("--use_shadow_texture_v68", action="store_true", default=False)
    parser.add_argument("--no_shadow_texture_v68_c_teacher",
                        dest="shadow_texture_v68_use_c_teacher",
                        action="store_false", default=True)
    parser.add_argument("--shadow_texture_v68_c_mode", default="correct",
                        choices=["correct", "wrong_random", "wrong_semantic_mismatch"])
    parser.add_argument("--shadow_texture_v68_c_feature_dropout", type=float, default=0.0)
    parser.add_argument("--shadow_texture_v68_c_region_dropout", type=float, default=0.0)
    parser.add_argument("--shadow_texture_v68_c_full_dropout", type=float, default=0.0)
    parser.add_argument("--shadow_texture_v68_gate_floor", type=float, default=0.0)
    parser.add_argument("--shadow_texture_v68_head_variant", default="simple",
                        choices=["simple", "lap_decoder", "guided_hf"])
    parser.add_argument("--shadow_texture_v68_struct_only", action="store_true", default=False)
    parser.add_argument("--shadow_texture_v68_force_gate", type=float, default=-1.0)
    parser.add_argument("--shadow_texture_v68_bypass_residual_gate",
                        action="store_true", default=False)
    parser.add_argument("--shadow_texture_v68_alpha_struct", type=float, default=0.05)
    parser.add_argument("--shadow_texture_v68_alpha_micro", type=float, default=0.025)
    parser.add_argument("--shadow_texture_v68_residual_init_std", type=float, default=0.0)
    parser.add_argument("--shadow_texture_v68_guided_gain_max", type=float, default=2.0)
    parser.add_argument("--shadow_texture_v68_guided_free_scale", type=float, default=0.03)
    parser.add_argument("--use_v69_shallow_lap_refiner", action="store_true", default=False)
    parser.add_argument("--shadow_lap_max_residual", type=float, default=0.10)
    parser.add_argument("--shadow_lap_hidden_dim", type=int, default=48)
    parser.add_argument("--shadow_lap_write_conf_floor", type=float, default=0.15)
    parser.add_argument("--use_v70_reflectance_lap_refiner", action="store_true", default=False)
    parser.add_argument("--train_v70_stage", default="all",
                        choices=["all", "illum_only", "reflectance_only"])
    parser.add_argument("--shadow_reflectance_disable_residual", action="store_true", default=False)
    parser.add_argument("--shadow_reflectance_max_residual", type=float, default=0.08)
    parser.add_argument("--shadow_reflectance_max_log_gain", type=float, default=0.35)
    parser.add_argument("--shadow_reflectance_hidden_dim", type=int, default=48)
    parser.add_argument("--shadow_reflectance_write_conf_floor", type=float, default=0.15)
    parser.add_argument("--use_v71_pawr_refiner", action="store_true", default=False)
    parser.add_argument("--shadow_pawr_gain_max", type=float, default=2.0)
    parser.add_argument("--shadow_pawr_free_scale", type=float, default=0.03)
    parser.add_argument("--shadow_pawr_hidden_dim", type=int, default=48)
    parser.add_argument("--shadow_pawr_write_conf_floor", type=float, default=0.15)
    parser.add_argument("--use_v72_boundary_matte_illumination", action="store_true", default=False)
    parser.add_argument("--use_v73_guided_matte_illumination", action="store_true", default=False)
    parser.add_argument("--train_v72_stage", default="matte_illum",
                        choices=["matte_illum", "pawr_core", "joint"])
    parser.add_argument("--disable_pawr", action="store_true", default=False)
    parser.add_argument("--use_v72_matte_as_output_blend", action="store_true", default=False)
    parser.add_argument("--shadow_matte_hidden_dim", type=int, default=48)
    parser.add_argument("--shadow_matte_max_log_gain", type=float, default=0.35)
    parser.add_argument("--shadow_matte_residual_logit_scale", type=float, default=1.0)
    parser.add_argument("--use_v74_boundary_material_harmonizer", action="store_true", default=False)
    parser.add_argument("--use_v75_gated_harmonizer", action="store_true", default=False)
    parser.add_argument("--shadow_harmonizer_hidden_dim", type=int, default=48)
    parser.add_argument("--shadow_harmonizer_max_delta", type=float, default=0.04)
    parser.add_argument("--shadow_harmonizer_material_floor", type=float, default=0.0)
    parser.add_argument("--shadow_harmonizer_material_blur_kernel", type=int, default=0)
    parser.add_argument("--shadow_harmonizer_material_dilate_kernel", type=int, default=0)
    parser.add_argument("--disable_shadow_texture_at_runtime", action="store_true", default=False,
                        help="Instantiate and load the texture modules, then bypass them during forward.")
    parser.add_argument("--output_shadow_blend_kernel", type=int, default=15)
    parser.add_argument("--output_shadow_blend_floor", type=float, default=0.02)
    parser.add_argument("--output_shadow_blend_mode", default="dilate",
                        choices=["dilate", "soft", "soft_boundary", "boundary"])

    parser.add_argument("--use_relation", action="store_true", default=True)
    parser.add_argument("--relation_prior_root", default="")
    parser.add_argument("--relation_prior_cache_mode", default="build")
    parser.add_argument("--relation_embed_dim", type=int, default=16)
    parser.add_argument("--relation_dropout", type=float, default=0.0)
    parser.add_argument("--relation_max_residual", type=float, default=0.18)
    parser.add_argument("--relation_max_modulation", type=float, default=0.05)
    parser.add_argument("--relation_late_only", action="store_true", default=True)
    parser.add_argument("--no_relation_late_only", dest="relation_late_only", action="store_false")
    parser.add_argument("--relation_residual_init", type=float, default=0.0)
    parser.add_argument("--relation_global_residual_scale", type=float, default=0.15)
    parser.add_argument("--relation_control_max_residual", type=float, default=0.05)
    parser.add_argument("--relation_gate_dilate", type=int, default=11)
    parser.add_argument("--relation_global_enable", action="store_true", default=True)
    parser.add_argument("--no_relation_global_enable", dest="relation_global_enable", action="store_false")
    parser.add_argument("--relation_legacy_global_enable", action="store_true", default=True)
    parser.add_argument("--no_relation_legacy_global_enable", dest="relation_legacy_global_enable", action="store_false")
    parser.add_argument("--relation_control_enable", action="store_true", default=False)
    parser.add_argument("--no_relation_control_enable", dest="relation_control_enable", action="store_false")
    parser.add_argument("--relation_control_apply_to_global", action="store_true", default=False)
    parser.add_argument("--no_relation_control_apply_to_global", dest="relation_control_apply_to_global", action="store_false")
    parser.add_argument("--use_relation_global_expert", action="store_true", default=False)
    parser.add_argument("--no_relation_global_expert", dest="use_relation_global_expert", action="store_false")
    parser.add_argument("--relation_global_expert_max_residual", type=float, default=0.03)
    parser.add_argument("--relation_global_use_tokens", action="store_true", default=True)
    parser.add_argument("--no_relation_global_use_tokens", dest="relation_global_use_tokens", action="store_false")
    parser.add_argument("--relation_global_use_stack", action="store_true", default=True)
    parser.add_argument("--no_relation_global_use_stack", dest="relation_global_use_stack", action="store_false")
    parser.add_argument("--use_global_psnr_head", action="store_true", default=False)
    parser.add_argument("--no_global_psnr_head", dest="use_global_psnr_head", action="store_false")
    parser.add_argument("--global_psnr_max_residual", type=float, default=0.02)
    parser.add_argument("--use_boundary_micro_head", action="store_true", default=False)
    parser.add_argument("--no_boundary_micro_head", dest="use_boundary_micro_head", action="store_false")
    parser.add_argument("--boundary_micro_max_residual", type=float, default=0.015)
    parser.add_argument("--relation_use_tokens", action="store_true", default=True)
    parser.add_argument("--no_relation_use_tokens", dest="relation_use_tokens", action="store_false")
    parser.add_argument("--relation_token_layers", type=int, default=2)
    parser.add_argument("--relation_token_heads", type=int, default=4)
    parser.add_argument("--relation_bypass_global", action="store_true", default=True)
    parser.add_argument("--no_relation_bypass_global", dest="relation_bypass_global", action="store_false")
    parser.add_argument("--relation_task_mode", default="global",
                        help="global/remove_selected/preserve_selected/remove_receiver.")
    parser.add_argument("--preserve_selected", action="store_true", default=False,
                        help="Preserve selected C/S pair and remove the remaining shadows.")
    parser.add_argument("--selected_pair_policy", default="first")
    parser.add_argument("--selected_pair_id", type=int, default=-1)
    parser.add_argument("--strict_checkpoint", action="store_true",
                        help="Load checkpoint with strict=True.")
    return parser.parse_args()


def resolve_roots(args):
    roots = args.weights_root if args.weights_root else ["./log"]
    return [Path(root).expanduser().resolve() for root in roots]


def checkpoint_state_dict(path):
    checkpoint = torch.load(path, map_location="cpu")
    if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        return checkpoint["state_dict"]
    return checkpoint


def assert_shadow_texture_checkpoint_compatible(model, weight_path, args):
    base_model = model.module if hasattr(model, "module") else model
    print("===> use_shadow_texture_refine:", bool(args.use_shadow_texture_refine))
    print("===> has shadow_texture_refiner:", hasattr(base_model, "shadow_texture_refiner"))
    print("===> use_v69_shallow_lap_refiner:", bool(getattr(args, "use_v69_shallow_lap_refiner", False)))
    print("===> has shadow_lap_refiner:", hasattr(base_model, "shadow_lap_refiner"))
    print("===> use_v70_reflectance_lap_refiner:", bool(getattr(args, "use_v70_reflectance_lap_refiner", False)))
    print("===> has shadow_reflectance_refiner:", hasattr(base_model, "shadow_reflectance_refiner"))
    print("===> use_v71_pawr_refiner:", bool(getattr(args, "use_v71_pawr_refiner", False)))
    print("===> has shadow_pawr_refiner:", hasattr(base_model, "shadow_pawr_refiner"))
    print("===> use_v72_boundary_matte_illumination:", bool(getattr(args, "use_v72_boundary_matte_illumination", False)))
    print("===> use_v73_guided_matte_illumination:", bool(getattr(args, "use_v73_guided_matte_illumination", False)))
    print("===> has shadow_matte_refiner:", hasattr(base_model, "shadow_matte_refiner"))
    print("===> use_v74_boundary_material_harmonizer:", bool(getattr(args, "use_v74_boundary_material_harmonizer", False)))
    print("===> has shadow_harmonizer:", hasattr(base_model, "shadow_harmonizer"))
    if (not args.use_shadow_texture_refine and
            not getattr(args, "use_v69_shallow_lap_refiner", False) and
            not getattr(args, "use_v70_reflectance_lap_refiner", False) and
            not getattr(args, "use_v71_pawr_refiner", False) and
            not getattr(args, "use_v72_boundary_matte_illumination", False) and
            not getattr(args, "use_v73_guided_matte_illumination", False) and
            not getattr(args, "use_v74_boundary_material_harmonizer", False)):
        return
    if args.use_shadow_texture_refine and not hasattr(base_model, "shadow_texture_refiner"):
        raise RuntimeError(
            "Eval model did not instantiate shadow_texture_refiner; texture evaluation is invalid.")
    if getattr(args, "use_v69_shallow_lap_refiner", False) and not hasattr(base_model, "shadow_lap_refiner"):
        raise RuntimeError("Eval model did not instantiate the v69 shallow Laplacian refiner.")
    if getattr(args, "use_v70_reflectance_lap_refiner", False) and not hasattr(base_model, "shadow_reflectance_refiner"):
        raise RuntimeError("Eval model did not instantiate the v70 reflectance-Laplacian refiner.")
    if getattr(args, "use_v71_pawr_refiner", False) and not hasattr(base_model, "shadow_pawr_refiner"):
        raise RuntimeError("Eval model did not instantiate the v71 PAWR refiner.")
    if ((getattr(args, "use_v72_boundary_matte_illumination", False) or
         getattr(args, "use_v73_guided_matte_illumination", False)) and
            not hasattr(base_model, "shadow_matte_refiner")):
        raise RuntimeError("Eval model did not instantiate the v72 boundary matte refiner.")
    if getattr(args, "use_v74_boundary_material_harmonizer", False) and not hasattr(base_model, "shadow_harmonizer"):
        raise RuntimeError("Eval model did not instantiate the v74 boundary/material harmonizer.")
    if args.use_shadow_texture_v68:
        print("===> has shadow_texture_v68_refiner:",
              hasattr(base_model, "shadow_texture_v68_refiner"))
        if not hasattr(base_model, "shadow_texture_v68_refiner"):
            raise RuntimeError("Eval model did not instantiate the v68 texture refiner.")
        if not getattr(base_model.shadow_texture_v68_refiner, "use_v68", False):
            raise RuntimeError("Eval model did not enable the v68 texture path.")

    checkpoint_keys = set(checkpoint_state_dict(weight_path).keys())
    model_keys = set(model.state_dict().keys())
    checkpoint_has_module = any(key.startswith("module.") for key in checkpoint_keys)
    model_has_module = any(key.startswith("module.") for key in model_keys)
    if checkpoint_has_module and not model_has_module:
        checkpoint_keys = {key[7:] if key.startswith("module.") else key for key in checkpoint_keys}
    elif model_has_module and not checkpoint_has_module:
        checkpoint_keys = {"module." + key for key in checkpoint_keys}

    texture_model_keys = {
        key for key in model_keys
        if "shadow_texture_" in key or "shadow_lap_refiner." in key or
        "shadow_reflectance_refiner." in key or "shadow_pawr_refiner." in key or
        "shadow_matte_refiner." in key or "shadow_harmonizer." in key
    }
    texture_checkpoint_keys = {
        key for key in checkpoint_keys
        if "shadow_texture_" in key or "shadow_lap_refiner." in key or
        "shadow_reflectance_refiner." in key or "shadow_pawr_refiner." in key or
        "shadow_matte_refiner." in key or "shadow_harmonizer." in key
    }
    missing_texture = sorted(texture_model_keys - checkpoint_keys)
    unexpected_texture = sorted(texture_checkpoint_keys - model_keys)
    if args.use_shadow_texture_v68:
        invalid_missing = missing_texture
    else:
        invalid_missing = [
            key for key in missing_texture
            if not (getattr(args, "use_v69_shallow_lap_refiner", False) and
                    key.replace("module.", "").startswith("shadow_lap_refiner."))
            and not (getattr(args, "use_v70_reflectance_lap_refiner", False) and
                     key.replace("module.", "").startswith("shadow_reflectance_refiner."))
            and not (getattr(args, "use_v71_pawr_refiner", False) and
                     key.replace("module.", "").startswith("shadow_pawr_refiner."))
            and not ((getattr(args, "use_v72_boundary_matte_illumination", False) or
                      getattr(args, "use_v73_guided_matte_illumination", False)) and
                     key.replace("module.", "").startswith("shadow_matte_refiner."))
            and not (getattr(args, "use_v74_boundary_material_harmonizer", False) and
                     key.replace("module.", "").startswith("shadow_harmonizer."))
        ]
        if missing_texture:
            print("===> texture model-only keys initialized:", len(missing_texture))
    if invalid_missing or unexpected_texture:
        raise RuntimeError(
            "Shadow texture checkpoint mismatch: missing={}, unexpected={}".format(
                invalid_missing[:20], unexpected_texture[:20]))
    print("===> shadow texture checkpoint keys matched:", len(texture_checkpoint_keys))


def _masked_mean(value, mask):
    expanded = mask.expand_as(value)
    return float((value * expanded).sum().item() / expanded.sum().clamp_min(1.0).item())


def _sobel_gradients(image):
    channels = image.shape[1]
    kernel_x = image.new_tensor([
        [-1.0, 0.0, 1.0],
        [-2.0, 0.0, 2.0],
        [-1.0, 0.0, 1.0],
    ]).view(1, 1, 3, 3) / 8.0
    kernel_y = kernel_x.transpose(-1, -2)
    kernel_x = kernel_x.expand(channels, 1, 3, 3)
    kernel_y = kernel_y.expand(channels, 1, 3, 3)
    grad_x = F.conv2d(image, kernel_x, padding=1, groups=channels)
    grad_y = F.conv2d(image, kernel_y, padding=1, groups=channels)
    return grad_x, grad_y


def _laplacian_of_gaussian(image):
    channels = image.shape[1]
    gaussian = image.new_tensor([
        [1.0, 4.0, 6.0, 4.0, 1.0],
        [4.0, 16.0, 24.0, 16.0, 4.0],
        [6.0, 24.0, 36.0, 24.0, 6.0],
        [4.0, 16.0, 24.0, 16.0, 4.0],
        [1.0, 4.0, 6.0, 4.0, 1.0],
    ]).view(1, 1, 5, 5) / 256.0
    laplacian = image.new_tensor([
        [0.0, 1.0, 0.0],
        [1.0, -4.0, 1.0],
        [0.0, 1.0, 0.0],
    ]).view(1, 1, 3, 3)
    blurred = F.conv2d(
        image, gaussian.expand(channels, 1, 5, 5), padding=2, groups=channels)
    return F.conv2d(
        blurred, laplacian.expand(channels, 1, 3, 3), padding=1, groups=channels)


def compute_texture_detail_metrics(restored, target, shadow_mask, boundary_mask=None):
    pred = torch.from_numpy(restored.transpose(2, 0, 1)).unsqueeze(0).float()
    gt = torch.from_numpy(target.transpose(2, 0, 1)).unsqueeze(0).float()
    mask = torch.from_numpy(shadow_mask).view(1, 1, *shadow_mask.shape).float().clamp(0, 1)

    pred_log = _laplacian_of_gaussian(pred)
    gt_log = _laplacian_of_gaussian(gt)
    hfen = math.sqrt(_masked_mean((pred_log - gt_log).square(), mask))

    pred_grad_x, pred_grad_y = _sobel_gradients(pred)
    gt_grad_x, gt_grad_y = _sobel_gradients(gt)
    grad_error = 0.5 * (
        _masked_mean((pred_grad_x - gt_grad_x).abs(), mask) +
        _masked_mean((pred_grad_y - gt_grad_y).abs(), mask))

    pred_high = pred - F.avg_pool2d(pred, kernel_size=7, stride=1, padding=3)
    gt_high = gt - F.avg_pool2d(gt, kernel_size=7, stride=1, padding=3)
    pred_energy = F.avg_pool2d(pred_high.abs(), kernel_size=7, stride=1, padding=3)
    gt_energy = F.avg_pool2d(gt_high.abs(), kernel_size=7, stride=1, padding=3)
    energy_error = _masked_mean((pred_energy - gt_energy).abs(), mask)

    if boundary_mask is None:
        dilated = F.max_pool2d(mask, kernel_size=7, stride=1, padding=3)
        eroded = 1.0 - F.max_pool2d(1.0 - mask, kernel_size=7, stride=1, padding=3)
        boundary = (dilated - eroded).clamp(0, 1)
    else:
        boundary = torch.from_numpy(boundary_mask).view(
            1, 1, *boundary_mask.shape).float().clamp(0, 1)
    boundary_grad_error = 0.5 * (
        _masked_mean((pred_grad_x - gt_grad_x).abs(), boundary) +
        _masked_mean((pred_grad_y - gt_grad_y).abs(), boundary))
    return {
        "hfen_shadow": hfen,
        "grad_l1_shadow": grad_error,
        "energy_error_shadow": energy_error,
        "boundary_grad_error": boundary_grad_error,
    }


def checkpoint_has_relation_weights(path):
    state_dict = checkpoint_state_dict(path)
    if not hasattr(state_dict, "keys"):
        return False
    return any("relation_" in key or ".relation" in key or "relation" in key
               for key in state_dict.keys())


def iter_checkpoint_candidates(roots, exclude_names):
    exclude_names = {name.strip().lower() for name in exclude_names.split(",") if name.strip()}
    for root in roots:
        if not root.exists():
            continue
        if root.is_file():
            candidates = [root]
        else:
            candidates = root.rglob("*")
        for path in candidates:
            if not path.is_file():
                continue
            if path.suffix.lower() not in CHECKPOINT_SUFFIXES:
                continue
            if path.name.lower() in exclude_names:
                continue
            if path.stat().st_size < 1024 * 1024:
                continue
            yield path


def find_latest_checkpoint(args):
    roots = resolve_roots(args)
    candidates = list(iter_checkpoint_candidates(roots, args.exclude_names))
    if not candidates:
        searched = ", ".join(str(root) for root in roots)
        raise FileNotFoundError("No checkpoint found under: {}".format(searched))

    preferred_names = [
        name.strip().lower()
        for name in (args.prefer_checkpoint_name or "").split(",")
        if name.strip()
    ]
    for preferred_name in preferred_names:
        preferred = [path for path in candidates if path.name.lower() == preferred_name]
        if preferred:
            return max(preferred, key=lambda path: path.stat().st_mtime)
    return max(candidates, key=lambda path: path.stat().st_mtime)


def resolve_weight_path(args):
    if args.weights.lower() == "auto":
        path = find_latest_checkpoint(args)
    else:
        path = Path(args.weights).expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError("Checkpoint not found: {}".format(path))
    if not args.allow_base_checkpoint and not checkpoint_has_relation_weights(path):
        raise RuntimeError(
            "Checkpoint has no relation/DESOBA branch weights: {}. "
            "Pass --allow_base_checkpoint only if this is intentional.".format(path))
    return path


def condition_enabled(args):
    return args.use_priors or args.use_relation


def move_condition_to_device(condition, device, repeat=1):
    moved = {}
    for key, value in condition.items():
        if not torch.is_tensor(value):
            moved[key] = value
            continue
        if repeat != 1:
            value = value.repeat(*([repeat] + [1] * (value.dim() - 1)))
        moved[key] = value.to(device, non_blocking=True)
    return moved


def metric_shadow_mask(condition, fallback_mask, image_root="", filename=""):
    metric_mask = None
    if isinstance(condition, dict):
        metric_mask = condition.get("metric_shadow")
        if torch.is_tensor(metric_mask):
            if metric_mask.shape[0] == 1 and fallback_mask.shape[0] > 1:
                metric_mask = metric_mask.repeat(fallback_mask.shape[0], 1, 1, 1)
            if metric_mask.shape[-2:] != fallback_mask.shape[-2:]:
                metric_mask = F.interpolate(
                    metric_mask.float(), size=fallback_mask.shape[-2:],
                    mode="bilinear", align_corners=False)
            metric_mask = metric_mask.to(
                device=fallback_mask.device, dtype=fallback_mask.dtype).clamp(0, 1)
    if metric_mask is None:
        metric_mask = fallback_mask
    empty = metric_mask.flatten(1).sum(dim=1) <= 0.5
    if empty.any():
        metric_mask = metric_mask.clone()
        metric_mask[empty] = fallback_mask[empty]
        empty = metric_mask.flatten(1).sum(dim=1) <= 0.5
    if empty.any() and image_root and filename:
        disk_path = Path(image_root) / "shadow_masks" / Path(filename).name
        disk_mask = cv2.imread(str(disk_path), cv2.IMREAD_GRAYSCALE)
        if disk_mask is not None:
            disk_mask = torch.from_numpy(
                (disk_mask > 127).astype(np.float32)).view(1, 1, *disk_mask.shape)
            disk_mask = F.interpolate(
                disk_mask, size=fallback_mask.shape[-2:], mode="nearest")
            disk_mask = disk_mask.to(
                device=fallback_mask.device, dtype=fallback_mask.dtype)
            if disk_mask.flatten(1).sum().item() > 0.5:
                metric_mask[empty] = disk_mask.expand_as(metric_mask)[empty]
    return metric_mask


def d4_transform_tensor(tensor, transform_id):
    if transform_id == 0:
        return tensor
    if transform_id == 1:
        return torch.rot90(tensor, k=1, dims=[-1, -2])
    if transform_id == 2:
        return torch.rot90(tensor, k=2, dims=[-1, -2])
    if transform_id == 3:
        return torch.rot90(tensor, k=3, dims=[-1, -2])
    if transform_id == 4:
        return tensor.flip(-2)
    if transform_id == 5:
        return torch.rot90(tensor, k=1, dims=[-1, -2]).flip(-2)
    if transform_id == 6:
        return torch.rot90(tensor, k=2, dims=[-1, -2]).flip(-2)
    if transform_id == 7:
        return torch.rot90(tensor, k=3, dims=[-1, -2]).flip(-2)
    raise ValueError("Unknown D4 transform id {}".format(transform_id))


def d4_inverse_tensor(tensor, transform_id):
    if transform_id == 0:
        return tensor
    if transform_id == 1:
        return torch.rot90(tensor, k=3, dims=[-1, -2])
    if transform_id == 2:
        return torch.rot90(tensor, k=2, dims=[-1, -2])
    if transform_id == 3:
        return torch.rot90(tensor, k=1, dims=[-1, -2])
    if transform_id == 4:
        return tensor.flip(-2)
    if transform_id == 5:
        return torch.rot90(tensor.flip(-2), k=3, dims=[-1, -2])
    if transform_id == 6:
        return torch.rot90(tensor.flip(-2), k=2, dims=[-1, -2])
    if transform_id == 7:
        return torch.rot90(tensor.flip(-2), k=1, dims=[-1, -2])
    raise ValueError("Unknown D4 transform id {}".format(transform_id))


def d4_transform_condition(condition, transform_id):
    if condition is None:
        return None
    spatial_keys = set(RELATION_SPATIAL_KEYS) | {
        "depth", "depth_quality", "confidence", "semantic", "semantic_quality",
    }
    transformed = {}
    for key, value in condition.items():
        if torch.is_tensor(value) and key in spatial_keys and value.dim() >= 3:
            transformed[key] = d4_transform_tensor(value, transform_id)
        else:
            transformed[key] = value
    return transformed


def _parse_int_list(value):
    return [int(item.strip()) for item in str(value).split(",") if item.strip()]


def _parse_float_list(value):
    return [float(item.strip()) for item in str(value).split(",") if item.strip()]


def _parse_string_list(value):
    return [item.strip() for item in str(value).split(",") if item.strip()]


def model_core(model):
    return model.module if hasattr(model, "module") else model


class BlendScope:
    def __init__(self, model, kernel, floor, mode):
        self.core = model_core(model)
        self.kernel = int(kernel)
        self.floor = float(floor)
        self.mode = str(mode)

    def __enter__(self):
        if not hasattr(self.core, "output_shadow_blend_kernel"):
            self.enabled = False
            return
        self.enabled = True
        self.old_kernel = self.core.output_shadow_blend_kernel
        self.old_floor = self.core.output_shadow_blend_floor
        self.old_mode = self.core.output_shadow_blend_mode
        self.core.output_shadow_blend_kernel = self.kernel
        self.core.output_shadow_blend_floor = self.floor
        self.core.output_shadow_blend_mode = self.mode

    def __exit__(self, exc_type, exc, tb):
        if not getattr(self, "enabled", False):
            return
        self.core.output_shadow_blend_kernel = self.old_kernel
        self.core.output_shadow_blend_floor = self.old_floor
        self.core.output_shadow_blend_mode = self.old_mode


def blend_configs(args, model):
    core = model_core(model)
    base_kernel = int(getattr(core, "output_shadow_blend_kernel", args.output_shadow_blend_kernel))
    base_floor = float(getattr(core, "output_shadow_blend_floor", args.output_shadow_blend_floor))
    base_mode = str(getattr(core, "output_shadow_blend_mode", args.output_shadow_blend_mode))
    kernels = _parse_int_list(args.blend_kernels) or [base_kernel]
    floors = _parse_float_list(args.blend_floors) or [base_floor]
    modes = _parse_string_list(args.blend_modes) or [base_mode]
    return [(kernel, floor, mode) for kernel in kernels for floor in floors for mode in modes]


def split_lowres_prior(prior, starts, crop_size, full_size, output_size):
    _, _, ph, pw = prior.shape
    H, W = full_size
    split_data = []
    for hs, ws in starts:
        r0 = int(math.floor(hs * ph / H))
        c0 = int(math.floor(ws * pw / W))
        r1 = int(math.ceil((hs + crop_size) * ph / H))
        c1 = int(math.ceil((ws + crop_size) * pw / W))
        r0 = max(0, min(r0, ph - 1))
        c0 = max(0, min(c0, pw - 1))
        r1 = max(r0 + 1, min(r1, ph))
        c1 = max(c0 + 1, min(c1, pw))
        crop = prior[:, :, r0:r1, c0:c1]
        crop = F.interpolate(crop, size=(output_size, output_size), mode="bilinear", align_corners=False)
        split_data.append(crop)
    return split_data


def split_priors(priors, starts, crop_size, full_size, semantic_size, tile_overlap):
    depth_data, _ = splitimage(priors["depth"], crop_size=crop_size, overlap_size=tile_overlap)
    depth_quality_data, _ = splitimage(priors["depth_quality"], crop_size=crop_size, overlap_size=tile_overlap)
    semantic_data = split_lowres_prior(priors["semantic"], starts, crop_size, full_size, semantic_size)
    semantic_quality_data = split_lowres_prior(
        priors["semantic_quality"], starts, crop_size, full_size, semantic_size)
    confidence_data = None
    if "confidence" in priors:
        confidence_data, _ = splitimage(priors["confidence"], crop_size=crop_size, overlap_size=tile_overlap)

    split_items = [
        {
            "depth": depth_data[i],
            "depth_quality": depth_quality_data[i],
            "semantic": semantic_data[i],
            "semantic_quality": semantic_quality_data[i],
        }
        for i in range(len(starts))
    ]
    if confidence_data is not None:
        for i, item in enumerate(split_items):
            item["confidence"] = confidence_data[i]
    return split_items


def split_condition(condition, starts, crop_size, full_size, semantic_size, tile_overlap):
    split_items = [{} for _ in range(len(starts))]
    if condition is None:
        return None
    if "depth" in condition and "semantic" in condition:
        prior_items = split_priors(condition, starts, crop_size, full_size, semantic_size, tile_overlap)
        for i, item in enumerate(prior_items):
            split_items[i].update(item)

    for key in RELATION_SPATIAL_KEYS:
        if key not in condition:
            continue
        split_data, _ = splitimage(condition[key], crop_size=crop_size, overlap_size=tile_overlap)
        for i, value in enumerate(split_data):
            split_items[i][key] = value
    for key in RELATION_SCALAR_KEYS + RELATION_META_KEYS:
        if key in condition:
            for item in split_items:
                item[key] = condition[key]
    return split_items


V68_SPATIAL_AUX_KEYS = (
    "shadow_texture_base",
    "shadow_texture_v68_inner_mask",
    "shadow_texture_v68_boundary_mask",
    "shadow_texture_v68_structure_residual",
    "shadow_texture_v68_micro_residual",
    "shadow_texture_v68_structure_gate",
    "shadow_texture_v68_micro_gate",
    "shadow_texture_v68_texture_gate",
    "shadow_texture_v68_raw_residual",
    "shadow_texture_v68_gated_residual",
    "shadow_texture_v68_texture_residual",
    "shadow_texture_v68_texture_output_raw",
    "shadow_texture_v68_texture_output_gated",
    "shadow_texture_v68_texture_output",
    "shadow_texture_base",
    "shadow_lap_base",
    "shadow_lap_inner_mask",
    "shadow_lap_boundary_mask",
    "shadow_lap_soft_mask",
    "shadow_lap_texture_conf",
    "shadow_lap_input_highpass",
    "shadow_lap_base_highpass",
    "shadow_lap_lap_s1",
    "shadow_lap_lap_s2",
    "shadow_lap_lap_s3",
    "shadow_lap_pred_residual_raw",
    "shadow_lap_pred_residual",
    "shadow_lap_texture_output",
    "shadow_reflectance_base",
    "shadow_reflectance_illum_output",
    "shadow_reflectance_illum_delta",
    "shadow_reflectance_inner_mask",
    "shadow_reflectance_boundary_mask",
    "shadow_reflectance_soft_mask",
    "shadow_reflectance_texture_conf",
    "shadow_reflectance_input_highpass",
    "shadow_reflectance_base_highpass",
    "shadow_reflectance_illum_highpass",
    "shadow_reflectance_lap_s1",
    "shadow_reflectance_lap_s2",
    "shadow_reflectance_lap_s3",
    "shadow_reflectance_pred_residual_raw",
    "shadow_reflectance_pred_residual",
    "shadow_reflectance_pred_residual_unapplied",
    "shadow_reflectance_texture_output",
    "shadow_pawr_base",
    "shadow_pawr_inner_mask",
    "shadow_pawr_boundary_mask",
    "shadow_pawr_soft_mask",
    "shadow_pawr_wave_mask",
    "shadow_pawr_input_lh",
    "shadow_pawr_input_hl",
    "shadow_pawr_input_hh",
    "shadow_pawr_illum_lh",
    "shadow_pawr_illum_hl",
    "shadow_pawr_illum_hh",
    "shadow_pawr_delta_lh",
    "shadow_pawr_delta_hl",
    "shadow_pawr_delta_hh",
    "shadow_pawr_gain_x",
    "shadow_pawr_gain_y",
    "shadow_pawr_free",
    "shadow_pawr_pred_residual",
    "shadow_pawr_texture_output",
    "shadow_matte_base",
    "shadow_matte_alpha",
    "shadow_matte_alpha_raw",
    "shadow_matte_alpha_prior",
    "shadow_matte_guided_prior",
    "shadow_matte_core",
    "shadow_matte_delta_log_gain",
    "shadow_matte_inner_mask",
    "shadow_matte_boundary_mask",
    "shadow_matte_soft_mask",
    "shadow_matte_illum_output",
    "shadow_matte_pawr_core_mask",
    "shadow_harmonizer_base",
    "shadow_harmonizer_boundary_mask",
    "shadow_harmonizer_material_mask",
    "shadow_harmonizer_delta_boundary",
    "shadow_harmonizer_delta_material",
    "shadow_harmonizer_gate_boundary",
    "shadow_harmonizer_gate_material",
    "shadow_harmonizer_local_stat_error",
    "shadow_harmonizer_nonregion_identity_mask",
    "shadow_harmonizer_receiver_mask",
    "shadow_harmonizer_output",
)


def _merge_tile_aux(tile_aux, starts, tile, resolution):
    merged = {}
    for key in V68_SPATIAL_AUX_KEYS:
        values = [item[key].detach().cpu() for item in tile_aux if key in item]
        if len(values) != len(tile_aux):
            continue
        if values[0].shape[-2:] != (tile, tile):
            values = [
                F.interpolate(value.float(), size=(tile, tile), mode="bilinear", align_corners=False)
                for value in values
            ]
        channels = values[0].shape[1]
        merged[key] = mergeimage(
            values, starts, crop_size=tile,
            resolution=(resolution[0], channels, resolution[2], resolution[3]))
    return merged


def _append_inverse_aux(aux_lists, merged_aux, transform_id):
    for key, value in merged_aux.items():
        aux_lists.setdefault(key, []).append(d4_inverse_tensor(value, transform_id))


def _mean_aux_lists(aux_lists):
    return {
        key: torch.mean(torch.cat(values, dim=0), dim=0, keepdim=True)
        for key, values in aux_lists.items() if values
    }


def _masked_tensor_stats(value, mask):
    if value is None or mask is None:
        return float("nan"), float("nan")
    if value.shape[-2:] != mask.shape[-2:]:
        mask = F.interpolate(mask.float(), size=value.shape[-2:], mode="nearest")
    if mask.shape[1] == 1 and value.shape[1] > 1:
        mask = mask.expand(-1, value.shape[1], -1, -1)
    selected = value.detach().float().abs()[mask > 0.5]
    if selected.numel() == 0:
        return float("nan"), float("nan")
    return float(selected.mean().item()), float(torch.quantile(selected, 0.90).item())


def _mask_inner_boundary(mask):
    tensor = torch.from_numpy(mask).view(1, 1, *mask.shape).float().clamp(0, 1)
    eroded = 1.0 - F.max_pool2d(1.0 - tensor, kernel_size=7, stride=1, padding=3)
    dilated = F.max_pool2d(tensor, kernel_size=7, stride=1, padding=3)
    inner = eroded.clamp(0, 1)
    boundary = (dilated - eroded).clamp(0, 1)
    return inner.numpy().squeeze(), boundary.numpy().squeeze()


def _highpass_numpy(image):
    tensor = torch.from_numpy(image.transpose(2, 0, 1)).unsqueeze(0).float()
    high = tensor - F.avg_pool2d(tensor, kernel_size=7, stride=1, padding=3)
    return high.abs().mean(dim=1).numpy().squeeze()


def _gradient_energy_numpy(image):
    tensor = torch.from_numpy(image.transpose(2, 0, 1)).unsqueeze(0).float()
    grad_x, grad_y = _sobel_gradients(tensor)
    return (grad_x.square() + grad_y.square()).sqrt().mean(dim=1).numpy().squeeze()


def build_hard_case_rois(baseline, target, shadow_mask, roi_size=96, roi_count=3):
    height, width = shadow_mask.shape
    window = max(32, min(int(roi_size), height, width))
    inner, _ = _mask_inner_boundary(shadow_mask)
    score_map = (
        (_highpass_numpy(baseline) - _highpass_numpy(target)).__abs__() * 0.60 +
        _gradient_energy_numpy(target) * 0.40
    ) * inner
    score = torch.from_numpy(score_map).view(1, 1, height, width)
    valid = torch.from_numpy(inner).view(1, 1, height, width)
    kernel = torch.ones((1, 1, window, window), dtype=score.dtype)
    summed = F.conv2d(score, kernel)
    coverage = F.conv2d(valid, kernel) / float(window * window)
    summed[coverage < 0.40] = -1.0
    candidates = []
    flat_order = torch.argsort(summed.flatten(), descending=True)
    out_width = summed.shape[-1]
    for flat_index in flat_order.tolist():
        if float(summed.flatten()[flat_index].item()) < 0:
            break
        y0 = flat_index // out_width
        x0 = flat_index % out_width
        box = [int(x0), int(y0), int(x0 + window), int(y0 + window)]
        overlaps = False
        for existing in candidates:
            ix = max(0, min(box[2], existing[2]) - max(box[0], existing[0]))
            iy = max(0, min(box[3], existing[3]) - max(box[1], existing[1]))
            if ix * iy > 0.10 * window * window:
                overlaps = True
                break
        if not overlaps:
            candidates.append(box)
        if len(candidates) >= int(roi_count):
            break
    return candidates


def load_roi_manifest(path):
    if not path or not os.path.exists(path):
        return {}
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def save_roi_manifest(path, manifest):
    if not path:
        return
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(manifest, handle, ensure_ascii=False, indent=2)


def _save_rgb(path, rgb):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    utils.save_img(img_as_ubyte(np.clip(rgb, 0, 1)), str(path))


def _tensor_local_highpass(image, kernel_size=7):
    kernel_size = max(3, int(kernel_size) | 1)
    padding = kernel_size // 2
    padded = F.pad(image, (padding, padding, padding, padding), mode="replicate")
    return image - F.avg_pool2d(padded, kernel_size=kernel_size, stride=1)


def _target_residual_tensor(aux, target):
    base = aux.get("shadow_pawr_base")
    inner = aux.get("shadow_pawr_inner_mask")
    if base is None:
        base = aux.get("shadow_reflectance_illum_output")
    if inner is None:
        inner = aux.get("shadow_reflectance_inner_mask")
    if base is None:
        base = aux.get("shadow_lap_base")
    if inner is None:
        inner = aux.get("shadow_lap_inner_mask")
    if base is None:
        base = aux.get("shadow_texture_base")
    if inner is None:
        inner = aux.get("shadow_texture_v68_inner_mask")
    if base is None or inner is None:
        return None
    base = base.detach().float().cpu()
    inner = inner.detach().float().cpu()
    target_tensor = torch.from_numpy(
        np.asarray(target, dtype=np.float32).transpose(2, 0, 1)
    ).unsqueeze(0)
    if target_tensor.shape[-2:] != base.shape[-2:]:
        target_tensor = F.interpolate(
            target_tensor, size=base.shape[-2:], mode="bilinear", align_corners=False)
    if inner.shape[-2:] != base.shape[-2:]:
        inner = F.interpolate(inner, size=base.shape[-2:], mode="bilinear", align_corners=False)
    return (_tensor_local_highpass(target_tensor, 7) - _tensor_local_highpass(base, 7)) * inner.clamp(0, 1)


def _pseudo_alpha_from_np(input_image, target_image, mask_tensor):
    if input_image is None or target_image is None or mask_tensor is None:
        return None
    input_tensor = torch.from_numpy(
        np.asarray(input_image, dtype=np.float32).transpose(2, 0, 1)
    ).unsqueeze(0)
    target_tensor = torch.from_numpy(
        np.asarray(target_image, dtype=np.float32).transpose(2, 0, 1)
    ).unsqueeze(0)
    mask = mask_tensor.detach().float()
    if mask.dim() == 4:
        mask = mask[:1, :1]
    elif mask.dim() == 3:
        mask = mask[:1].unsqueeze(0)
    else:
        return None
    if mask.shape[-2:] != input_tensor.shape[-2:]:
        mask = F.interpolate(mask, size=input_tensor.shape[-2:], mode="bilinear", align_corners=False)
    padded_input = F.pad(input_tensor, (7, 7, 7, 7), mode="replicate")
    padded_target = F.pad(target_tensor, (7, 7, 7, 7), mode="replicate")
    low_input = F.avg_pool2d(padded_input, kernel_size=15, stride=1)
    low_target = F.avg_pool2d(padded_target, kernel_size=15, stride=1)
    diff = (low_target - low_input).abs().mean(dim=1, keepdim=True)
    dilated = F.max_pool2d(mask.clamp(0, 1), kernel_size=21, stride=1, padding=10)
    alpha = diff * dilated
    denom = alpha.flatten(1).amax(dim=1).view(-1, 1, 1, 1).clamp_min(1.0e-6)
    alpha = (alpha / denom).clamp(0, 1)
    alpha = F.avg_pool2d(F.pad(alpha, (7, 7, 7, 7), mode="replicate"), kernel_size=15, stride=1)
    return alpha[0, 0].numpy()


def save_v68_aux_images(result_dir, filename, aux, target, input_image=None):
    stem = Path(filename).stem
    output_dir = Path(result_dir) / "aux" / stem
    maps = {
        "texture_gate": aux.get("shadow_texture_v68_texture_gate"),
        "structure_gate": aux.get("shadow_texture_v68_structure_gate"),
        "micro_gate": aux.get("shadow_texture_v68_micro_gate"),
        "M_inner": aux.get("shadow_texture_v68_inner_mask"),
        "M_boundary": aux.get("shadow_texture_v68_boundary_mask"),
        "shadow_lap_M_inner": aux.get("shadow_lap_inner_mask"),
        "shadow_lap_M_boundary": aux.get("shadow_lap_boundary_mask"),
        "shadow_lap_M_soft": aux.get("shadow_lap_soft_mask"),
        "shadow_lap_write_conf": aux.get("shadow_lap_texture_conf"),
        "shadow_reflectance_M_inner": aux.get("shadow_reflectance_inner_mask"),
        "shadow_reflectance_M_boundary": aux.get("shadow_reflectance_boundary_mask"),
        "shadow_reflectance_M_soft": aux.get("shadow_reflectance_soft_mask"),
        "shadow_reflectance_write_conf": aux.get("shadow_reflectance_texture_conf"),
        "shadow_pawr_M_inner": aux.get("shadow_pawr_inner_mask"),
        "shadow_pawr_M_boundary": aux.get("shadow_pawr_boundary_mask"),
        "shadow_pawr_M_soft": aux.get("shadow_pawr_soft_mask"),
        "shadow_pawr_wave_mask": aux.get("shadow_pawr_wave_mask"),
        "shadow_matte_A_shadow": aux.get("shadow_matte_alpha"),
        "shadow_matte_A_raw": aux.get("shadow_matte_alpha_raw"),
        "shadow_matte_A_prior": aux.get("shadow_matte_alpha_prior"),
        "shadow_matte_guided_prior": aux.get("shadow_matte_guided_prior"),
        "shadow_matte_A_core": aux.get("shadow_matte_core"),
        "shadow_matte_M_inner": aux.get("shadow_matte_inner_mask"),
        "shadow_matte_M_boundary": aux.get("shadow_matte_boundary_mask"),
        "shadow_matte_M_soft": aux.get("shadow_matte_soft_mask"),
        "shadow_matte_pawr_core_mask": aux.get("shadow_matte_pawr_core_mask"),
        "shadow_harmonizer_B_soft": aux.get("shadow_harmonizer_boundary_mask"),
        "shadow_harmonizer_M_local": aux.get("shadow_harmonizer_material_mask"),
        "shadow_harmonizer_local_stat_error": aux.get("shadow_harmonizer_local_stat_error"),
        "shadow_harmonizer_nonregion_identity_mask": aux.get("shadow_harmonizer_nonregion_identity_mask"),
        "shadow_harmonizer_receiver_mask": aux.get("shadow_harmonizer_receiver_mask"),
    }
    for name, value in maps.items():
        if value is None:
            continue
        image = value[0].detach().float().mean(dim=0).numpy()
        _save_rgb(output_dir / (name + ".png"), np.repeat(image[:, :, None], 3, axis=2))

    alpha_mask = aux.get("shadow_matte_soft_mask")
    if alpha_mask is None:
        alpha_mask = aux.get("shadow_matte_alpha")
    alpha_pseudo = _pseudo_alpha_from_np(input_image, target, alpha_mask)
    if alpha_pseudo is not None:
        _save_rgb(output_dir / "shadow_matte_alpha_gt_pseudo.png",
                  np.repeat(alpha_pseudo[:, :, None], 3, axis=2))

    residual = aux.get("shadow_texture_v68_texture_residual")
    if residual is None:
        residual = aux.get("shadow_lap_pred_residual")
    if residual is None:
        residual = aux.get("shadow_reflectance_pred_residual")
    if residual is None:
        residual = aux.get("shadow_pawr_pred_residual")
    if residual is not None:
        residual_np = residual[0].detach().float().numpy().transpose(1, 2, 0)
        residual_abs = np.abs(residual_np).mean(axis=2) / 0.05
        _save_rgb(output_dir / "texture_residual_abs.png",
                  np.repeat(residual_abs[:, :, None], 3, axis=2))
        _save_rgb(output_dir / "texture_residual_rgb.png", residual_np / 0.10 + 0.5)

    for name, key in (
            ("hp_input_abs", "shadow_texture_v68_input_highpass"),
            ("hp_base_abs", "shadow_texture_v68_base_highpass"),
            ("raw_unbounded_residual_abs", "shadow_texture_v68_raw_unbounded_residual"),
            ("raw_residual_abs", "shadow_texture_v68_raw_residual"),
            ("gated_residual_abs", "shadow_texture_v68_gated_residual"),
            ("shadow_lap_input_highpass_abs", "shadow_lap_input_highpass"),
            ("shadow_lap_base_highpass_abs", "shadow_lap_base_highpass"),
            ("shadow_lap_pred_residual_raw_abs", "shadow_lap_pred_residual_raw"),
            ("shadow_lap_lap_s1_abs", "shadow_lap_lap_s1"),
            ("shadow_lap_lap_s2_abs", "shadow_lap_lap_s2"),
            ("shadow_lap_lap_s3_abs", "shadow_lap_lap_s3"),
            ("shadow_reflectance_input_highpass_abs", "shadow_reflectance_input_highpass"),
            ("shadow_reflectance_base_highpass_abs", "shadow_reflectance_base_highpass"),
            ("shadow_reflectance_illum_highpass_abs", "shadow_reflectance_illum_highpass"),
            ("shadow_reflectance_pred_residual_raw_abs", "shadow_reflectance_pred_residual_raw"),
            ("shadow_reflectance_pred_residual_unapplied_abs", "shadow_reflectance_pred_residual_unapplied"),
            ("shadow_reflectance_lap_s1_abs", "shadow_reflectance_lap_s1"),
            ("shadow_reflectance_lap_s2_abs", "shadow_reflectance_lap_s2"),
            ("shadow_reflectance_lap_s3_abs", "shadow_reflectance_lap_s3"),
            ("shadow_reflectance_illum_delta_abs", "shadow_reflectance_illum_delta"),
            ("shadow_pawr_input_lh_abs", "shadow_pawr_input_lh"),
            ("shadow_pawr_input_hl_abs", "shadow_pawr_input_hl"),
            ("shadow_pawr_input_hh_abs", "shadow_pawr_input_hh"),
            ("shadow_pawr_illum_lh_abs", "shadow_pawr_illum_lh"),
            ("shadow_pawr_illum_hl_abs", "shadow_pawr_illum_hl"),
            ("shadow_pawr_illum_hh_abs", "shadow_pawr_illum_hh"),
            ("shadow_pawr_delta_lh_abs", "shadow_pawr_delta_lh"),
            ("shadow_pawr_delta_hl_abs", "shadow_pawr_delta_hl"),
            ("shadow_pawr_delta_hh_abs", "shadow_pawr_delta_hh"),
            ("shadow_pawr_gain_x_abs", "shadow_pawr_gain_x"),
            ("shadow_pawr_gain_y_abs", "shadow_pawr_gain_y"),
            ("shadow_pawr_free_abs", "shadow_pawr_free"),
            ("shadow_matte_delta_log_gain_abs", "shadow_matte_delta_log_gain"),
            ("shadow_harmonizer_delta_boundary_abs", "shadow_harmonizer_delta_boundary"),
            ("shadow_harmonizer_delta_material_abs", "shadow_harmonizer_delta_material")):
        value = aux.get(key)
        if value is None:
            continue
        value_np = value[0].detach().float().numpy().transpose(1, 2, 0)
        value_abs = np.abs(value_np).mean(axis=2) / 0.05
        _save_rgb(output_dir / (name + ".png"),
                  np.repeat(value_abs[:, :, None], 3, axis=2))

    target_residual = _target_residual_tensor(aux, target)
    if target_residual is not None:
        target_np = target_residual[0].numpy().transpose(1, 2, 0)
        target_abs = np.abs(target_np).mean(axis=2) / 0.05
        _save_rgb(output_dir / "target_residual_abs.png",
                  np.repeat(target_abs[:, :, None], 3, axis=2))
        _save_rgb(output_dir / "target_residual_rgb.png", target_np / 0.10 + 0.5)

    for name, key in (
            ("base_rgb", "shadow_texture_base"),
            ("texture_output_raw", "shadow_texture_v68_texture_output_raw"),
            ("texture_output_gated", "shadow_texture_v68_texture_output_gated"),
            ("texture_output_final", "shadow_texture_v68_texture_output"),
            ("shadow_lap_base_rgb", "shadow_lap_base"),
            ("shadow_lap_texture_output", "shadow_lap_texture_output"),
            ("base_error_map", "shadow_texture_base"),
            ("shadow_lap_base_error_map", "shadow_lap_base"),
            ("shadow_lap_texture_error_map", "shadow_lap_texture_output"),
            ("shadow_reflectance_base_rgb", "shadow_reflectance_base"),
            ("shadow_reflectance_illum_output", "shadow_reflectance_illum_output"),
            ("shadow_reflectance_texture_output", "shadow_reflectance_texture_output"),
            ("shadow_reflectance_base_error_map", "shadow_reflectance_base"),
            ("shadow_reflectance_illum_error_map", "shadow_reflectance_illum_output"),
            ("shadow_reflectance_texture_error_map", "shadow_reflectance_texture_output"),
            ("shadow_pawr_base_rgb", "shadow_pawr_base"),
            ("shadow_pawr_texture_output", "shadow_pawr_texture_output"),
            ("shadow_pawr_base_error_map", "shadow_pawr_base"),
            ("shadow_pawr_texture_error_map", "shadow_pawr_texture_output"),
            ("shadow_matte_base_rgb", "shadow_matte_base"),
            ("shadow_matte_illum_output", "shadow_matte_illum_output"),
            ("shadow_matte_base_error_map", "shadow_matte_base"),
            ("shadow_matte_illum_error_map", "shadow_matte_illum_output"),
            ("shadow_harmonizer_base_rgb", "shadow_harmonizer_base"),
            ("shadow_harmonizer_output", "shadow_harmonizer_output"),
            ("shadow_harmonizer_base_error_map", "shadow_harmonizer_base"),
            ("shadow_harmonizer_error_map", "shadow_harmonizer_output"),
            ("texture_error_map", "shadow_texture_v68_texture_output")):
        value = aux.get(key)
        if value is None:
            continue
        image = value[0].detach().float().numpy().transpose(1, 2, 0)
        if name.endswith("_map"):
            error = np.abs(image - target).mean(axis=2) / 0.10
            _save_rgb(output_dir / (name + ".png"), np.repeat(error[:, :, None], 3, axis=2))
        else:
            _save_rgb(output_dir / (name + ".png"), image)


def compute_roi_rows(filename, restored, target, shadow_mask, boxes):
    rows = []
    _, global_boundary = _mask_inner_boundary(shadow_mask)
    for index, box in enumerate(boxes):
        x0, y0, x1, y1 = [int(value) for value in box]
        roi = np.zeros_like(shadow_mask, dtype=np.float32)
        roi[y0:y1, x0:x1] = 1.0
        roi_shadow = roi * shadow_mask
        paper = compute_paper_metrics_compat(restored, target, roi_shadow, include_lpips=False)
        texture = compute_texture_detail_metrics(
            restored, target, roi_shadow, boundary_mask=global_boundary * roi)
        rows.append({
            "image_id": Path(filename).stem,
            "roi_index": index,
            "x0": x0, "y0": y0, "x1": x1, "y1": y1,
            "roi_psnr": paper["psnr_shadow"],
            "roi_hfen": texture["hfen_shadow"],
            "roi_grad_l1": texture["grad_l1_shadow"],
            "roi_energy_error": texture["energy_error_shadow"],
            "roi_boundary_grad": texture["boundary_grad_error"],
        })
    return rows


def auto_tile_size(height, width, requested_tile):
    if requested_tile > 0:
        return min(requested_tile, height, width)
    min_side = min(height, width)
    if min_side >= 384:
        return 384
    if min_side >= 256:
        return 256
    if min_side >= 128:
        return 128
    raise ValueError("Image is too small for tiled evaluation: {}x{}".format(height, width))


def build_dataset(args):
    task_mode = "preserve_selected" if args.preserve_selected else args.relation_task_mode
    return get_shadow_relation_data(
        args.relation_annotation_root,
        args.relation_image_root,
        {"patch_size": args.train_ps, "semantic_size": args.prior_semantic_size},
        split=args.split,
        preserve_selected_prob=0.0,
        preserve_selected=True if args.preserve_selected else None,
        selected_pair_policy=args.selected_pair_policy,
        selected_pair_id=args.selected_pair_id,
        remove_selected_prob=0.0,
        receiver_selected_prob=0.0,
        relation_task_mode=task_mode,
        relation_prior_root=args.relation_prior_root,
        relation_prior_cache_mode=args.relation_prior_cache_mode,
        use_priors=args.use_priors,
        prior_root=args.prior_root,
        prior_quality_root=args.prior_quality_root,
        prior_semantic_subdir=args.prior_semantic_subdir,
        prior_confidence_root=args.prior_confidence_root)


def save_metrics_csv(path, metrics):
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["region", "psnr", "ssim", "mae", "rmse", "lpips"])
        regions = ["shadow", "nonshadow", "all"]
        for region in ("relation_target", "relation_protect", "relation_protect_input"):
            if "psnr_" + region in metrics:
                regions.append(region)
        for region in regions:
            writer.writerow([
                region,
                "{:.6f}".format(metrics["psnr_" + region]),
                "{:.6f}".format(metrics["ssim_" + region]),
                "{:.6f}".format(metrics["mae_" + region]),
                "{:.6f}".format(metrics["rmse_" + region]),
                "{:.6f}".format(metrics["lpips_" + region]),
            ])
        writer.writerow([])
        writer.writerow(["texture_metric", "value"])
        for key in ("hfen_shadow", "grad_l1_shadow", "energy_error_shadow", "boundary_grad_error"):
            if key in metrics:
                writer.writerow([key, "{:.8f}".format(metrics[key])])


def save_dict_rows_csv(path, rows):
    if not path or not rows:
        return
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys())
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _to_numpy_mask(value):
    if value is None or not torch.is_tensor(value):
        return None
    if value.dim() >= 4:
        value = value[0]
    return value.detach().cpu().numpy().squeeze().astype(np.float32)


def _merge_dynamic_metric_lists(dst, src):
    for key, value in src.items():
        if isinstance(value, list):
            dst.setdefault(key, []).extend(value)
        else:
            dst.setdefault(key, []).append(value)


def _mean_dynamic_metric_lists(metrics):
    out = {}
    for key, values in metrics.items():
        values = np.asarray(values, dtype=np.float64)
        values = values[np.isfinite(values)]
        out[key] = float(values.mean()) if values.size else float("nan")
    return out


def _region_metric_from_shadow_slot(prefix, pred_rgb, gt_rgb, region_mask, include_lpips=True):
    sample = compute_paper_metrics_compat(pred_rgb, gt_rgb, region_mask, include_lpips=include_lpips)
    return {
        "psnr_" + prefix: sample["psnr_shadow"],
        "ssim_" + prefix: sample["ssim_shadow"],
        "mae_" + prefix: sample["mae_shadow"],
        "rmse_" + prefix: sample["rmse_shadow"],
        "lpips_" + prefix: sample["lpips_shadow"],
    }


def compute_relation_control_metrics(pred_rgb, gt_rgb, input_rgb, condition, include_lpips=True):
    if not isinstance(condition, dict):
        return {}
    out = {}
    remove_mask = _to_numpy_mask(condition.get("remove_shadow"))
    protect_mask = _to_numpy_mask(condition.get("protect_shadow"))
    if remove_mask is not None and remove_mask.sum() > 0:
        out.update(_region_metric_from_shadow_slot(
            "relation_target", pred_rgb, gt_rgb, remove_mask, include_lpips=include_lpips))
    if protect_mask is not None and protect_mask.sum() > 0:
        out.update(_region_metric_from_shadow_slot(
            "relation_protect", pred_rgb, gt_rgb, protect_mask, include_lpips=include_lpips))
        out.update(_region_metric_from_shadow_slot(
            "relation_protect_input", pred_rgb, input_rgb, protect_mask, include_lpips=include_lpips))
    return out


def format_relation_metric_table(method_name, metrics):
    rows = []
    for label, key in (
            ("Target/Edit", "relation_target"),
            ("Protect-vs-GT", "relation_protect"),
            ("Protect-vs-Input", "relation_protect_input")):
        metric_key = "psnr_" + key
        if metric_key not in metrics:
            continue
        rows.append("{:<18} | {:8.2f} {:8.3f} {:8.2f} {:8.2f} {:8.4f}".format(
            label,
            metrics["psnr_" + key], metrics["ssim_" + key],
            metrics["mae_" + key], metrics["rmse_" + key],
            metrics["lpips_" + key]))
    if not rows:
        return ""
    return "\n".join([
        "Relation control table (target/protect masks from DESOBA relation protocol):",
        "{:<18} | {:>8} {:>8} {:>8} {:>8} {:>8}".format(
            method_name, "PSNR", "SSIM", "MAE", "RMSE", "LPIPS"),
        "-" * 68,
    ] + rows)


def main():
    args = parse_args()
    if getattr(args, "use_v75_gated_harmonizer", False):
        args.use_v74_boundary_material_harmonizer = True
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    if not args.non_deterministic_eval:
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
    if args.gpus.lower() != "cpu":
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus

    weight_path = resolve_weight_path(args)
    device = torch.device("cuda" if args.gpus.lower() != "cpu" and torch.cuda.is_available() else "cpu")
    utils.mkdir(args.result_dir)

    dataset = build_dataset(args)
    dataset.is_train = False
    wanted = {item.strip() for item in args.eval_image_ids.split(",") if item.strip()}
    if args.eval_image_list:
        with open(args.eval_image_list, "r", encoding="utf-8") as stream:
            header = stream.readline().rstrip("\n").split("\t")
            id_index = header.index("image_id") if "image_id" in header else 0
            for line in stream:
                parts = line.rstrip("\n").split("\t")
                if len(parts) > id_index and parts[id_index]:
                    wanted.add(parts[id_index])
    if wanted:
        filtered = []
        matched = set()
        for record in dataset.records:
            image_id = str(record.get("image_id", ""))
            source_stem = Path(record.get("source_file", image_id)).stem
            target_stem = Path(record.get("target_file", image_id)).stem
            names = {image_id, source_stem, target_stem}
            if wanted & names:
                filtered.append(record)
                matched.update(wanted & names)
        missing = sorted(wanted - matched)
        if missing:
            raise RuntimeError("Requested --eval_image_ids not found: {}".format(", ".join(missing)))
        dataset.records = filtered
        dataset.tar_size = len(filtered)
        preview = ",".join(sorted(wanted)[:8])
        if len(wanted) > 8:
            preview += ",..."
        print("===> Filtered eval images: {} requested, {}".format(len(wanted), preview))
    loader = DataLoader(dataset=dataset, batch_size=1, shuffle=False,
                        num_workers=args.num_workers, drop_last=False)

    model = utils.get_arch(args)
    if device.type == "cuda" and not args.no_dataparallel:
        model = torch.nn.DataParallel(model)
    assert_shadow_texture_checkpoint_compatible(model, str(weight_path), args)
    utils.load_checkpoint(model, str(weight_path), strict=args.strict_checkpoint)
    base_model = model.module if hasattr(model, "module") else model
    if args.disable_shadow_texture_at_runtime:
        if not hasattr(base_model, "shadow_texture_v68_refiner"):
            raise RuntimeError(
                "--disable_shadow_texture_at_runtime requires an instantiated v68 texture refiner.")
        base_model.use_shadow_texture_v68 = False
        print("===> v68 texture refiner disabled at runtime; legacy v42 texture remains active.")
    model = model.to(device)
    model.eval()

    print("===> DESOBA v2 split:", args.split)
    print("===> Number of images:", len(dataset) if args.max_images <= 0 else min(args.max_images, len(dataset)))
    print("===> Testing using latest weights:", weight_path)
    print("===> Device:", device)
    if args.use_relation:
        print("===> Relation global bypass:", args.relation_bypass_global)
    print("===> Metric mask: condition['metric_shadow'] with relation mask fallback")
    print("===> D4 TTA:", bool(args.tta_d4))
    print("===> Blend configs:", blend_configs(args, model))

    metric_lists = {}
    relation_metric_lists = {}
    texture_metric_lists = {}
    per_image_rows = []
    roi_rows = []
    roi_manifest = load_roi_manifest(args.roi_manifest)
    hard_case_ids = set(_parse_string_list(args.hard_case_ids))
    need_texture_aux = bool(
        ((args.use_shadow_texture_v68 and not args.disable_shadow_texture_at_runtime) or
         getattr(args, "use_v69_shallow_lap_refiner", False) or
         getattr(args, "use_v70_reflectance_lap_refiner", False) or
         getattr(args, "use_v71_pawr_refiner", False) or
         getattr(args, "use_v72_boundary_matte_illumination", False) or
         getattr(args, "use_v73_guided_matte_illumination", False) or
         getattr(args, "use_v74_boundary_material_harmonizer", False)) and
        (args.save_v68_aux or args.save_per_image_csv or args.roi_manifest))
    include_lpips = not bool(args.skip_lpips)
    total_images = len(dataset) if args.max_images <= 0 else min(args.max_images, len(dataset))
    eval_loader = loader if args.max_images <= 0 else islice(loader, args.max_images)
    with torch.no_grad():
        for ii, data_test in tqdm(enumerate(eval_loader, 0), total=total_images):

            filename = data_test[4][0]
            _, _, H, W = data_test[1].shape
            tile = auto_tile_size(H, W, args.tile)

            rgb_gt = data_test[0].numpy().squeeze().transpose((1, 2, 0))
            rgb_input = data_test[1].numpy().squeeze().transpose((1, 2, 0))
            metric_mask = metric_shadow_mask(
                data_test[3] if condition_enabled(args) else None,
                data_test[2], args.relation_image_root, filename)
            mask_gt = metric_mask.numpy().squeeze()
            restored_list = []
            aux_lists = {}

            repeat = max(1, int(args.repeat))
            base_repeat = max(1, min(int(args.base_repeat), repeat))
            transform_ids = list(range(8)) if args.tta_d4 else [0]
            for kernel, floor, mode in blend_configs(args, model):
                with BlendScope(model, kernel, floor, mode):
                    for transform_id in transform_ids:
                        done = 0
                        while done < repeat:
                            chunk = min(base_repeat, repeat - done)
                            done += chunk

                            input_cpu = d4_transform_tensor(data_test[1], transform_id)
                            mask_cpu = d4_transform_tensor(data_test[2], transform_id)
                            condition_cpu = d4_transform_condition(
                                data_test[3] if condition_enabled(args) else None, transform_id)
                            input_tensor = input_cpu.repeat(chunk, 1, 1, 1).to(device, non_blocking=True)
                            mask_tensor = mask_cpu.repeat(chunk, 1, 1, 1).to(device, non_blocking=True)
                            condition = move_condition_to_device(
                                condition_cpu, device, repeat=chunk) if condition_enabled(args) else None
                            B, C, H, W = input_tensor.shape

                            split_data, starts = splitimage(input_tensor, crop_size=tile, overlap_size=args.tile_overlap)
                            mask_data, _ = splitimage(mask_tensor, crop_size=tile, overlap_size=args.tile_overlap)
                            condition_data = split_condition(
                                condition, starts, tile, (H, W), args.prior_semantic_size, args.tile_overlap)

                            tile_aux = []
                            for i, (tile_input, tile_mask) in enumerate(zip(split_data, mask_data)):
                                if need_texture_aux:
                                    tile_output, tile_output_aux = model(
                                        tile_input, tile_mask, condition_data[i], return_aux=True)
                                    split_data[i] = tile_output.cpu()
                                    tile_aux.append(tile_output_aux)
                                else:
                                    split_data[i] = model(tile_input, tile_mask, condition_data[i]).cpu()

                            restored = mergeimage(split_data, starts, crop_size=tile, resolution=(B, C, H, W))
                            restored_list.append(d4_inverse_tensor(restored, transform_id))
                            if need_texture_aux:
                                merged_aux = _merge_tile_aux(tile_aux, starts, tile, (B, C, H, W))
                                _append_inverse_aux(aux_lists, merged_aux, transform_id)

            restored = torch.mean(torch.cat(restored_list, dim=0), dim=0, keepdim=True)
            merged_aux = _mean_aux_lists(aux_lists)
            rgb_restored = torch.clamp(restored, 0, 1).cpu().numpy().squeeze().transpose((1, 2, 0))
            sample_metrics = compute_paper_metrics_compat(
                rgb_restored, rgb_gt, mask_gt, include_lpips=include_lpips)
            sample_texture_metrics = compute_texture_detail_metrics(rgb_restored, rgb_gt, mask_gt)
            merge_metric_lists(metric_lists, sample_metrics)
            _merge_dynamic_metric_lists(
                texture_metric_lists, sample_texture_metrics)
            _merge_dynamic_metric_lists(
                relation_metric_lists,
                compute_relation_control_metrics(
                    rgb_restored, rgb_gt, rgb_input, data_test[3],
                    include_lpips=include_lpips))

            if args.save_images:
                utils.save_img(img_as_ubyte(rgb_restored), os.path.join(args.result_dir, filename))

            inner_aux = merged_aux.get("shadow_lap_inner_mask")
            if inner_aux is None:
                inner_aux = merged_aux.get("shadow_pawr_inner_mask")
            if inner_aux is None:
                inner_aux = merged_aux.get("shadow_reflectance_inner_mask")
            if inner_aux is None:
                inner_aux = merged_aux.get("shadow_texture_v68_inner_mask")
            gate_mean, gate_p90 = _masked_tensor_stats(
                merged_aux.get("shadow_texture_v68_texture_gate"), inner_aux)
            residual_tensor = merged_aux.get("shadow_lap_pred_residual")
            if residual_tensor is None:
                residual_tensor = merged_aux.get("shadow_pawr_pred_residual")
            if residual_tensor is None:
                residual_tensor = merged_aux.get("shadow_reflectance_pred_residual")
            if residual_tensor is None:
                residual_tensor = merged_aux.get("shadow_texture_v68_texture_residual")
            residual_mean, residual_p90 = _masked_tensor_stats(
                residual_tensor, inner_aux)
            target_residual = _target_residual_tensor(merged_aux, rgb_gt)
            target_residual_mean, target_residual_p90 = _masked_tensor_stats(
                target_residual, inner_aux)
            if getattr(args, "use_v71_pawr_refiner", False):
                activation_status = "pawr_active" if np.isfinite(residual_mean) and residual_mean >= 1e-5 else "pawr_residual_zero"
            elif getattr(args, "use_v70_reflectance_lap_refiner", False):
                activation_status = "reflectance_active" if np.isfinite(residual_mean) and residual_mean >= 1e-5 else "reflectance_residual_zero"
            elif getattr(args, "use_v69_shallow_lap_refiner", False):
                activation_status = "lap_active" if np.isfinite(residual_mean) and residual_mean >= 1e-5 else "lap_residual_zero"
            elif np.isfinite(gate_mean) and gate_mean < 1e-4:
                activation_status = "gate_off"
            elif np.isfinite(residual_mean) and residual_mean < 1e-5:
                activation_status = "residual_zero"
            elif np.isfinite(gate_mean):
                activation_status = "active"
            else:
                activation_status = "v68_disabled"
            per_image_rows.append({
                "image_id": Path(filename).stem,
                "psnr_shadow": sample_metrics["psnr_shadow"],
                "hfen_shadow": sample_texture_metrics["hfen_shadow"],
                "grad_l1_shadow": sample_texture_metrics["grad_l1_shadow"],
                "energy_error_shadow": sample_texture_metrics["energy_error_shadow"],
                "boundary_grad_error": sample_texture_metrics["boundary_grad_error"],
                "psnr_nonshadow": sample_metrics["psnr_nonshadow"],
                "mae_nonshadow": sample_metrics["mae_nonshadow"],
                "gate_mean_inner": gate_mean,
                "gate_p90_inner": gate_p90,
                "residual_abs_mean_inner": residual_mean,
                "residual_abs_p90_inner": residual_p90,
                "target_residual_abs_mean_inner": target_residual_mean,
                "target_residual_abs_p90_inner": target_residual_p90,
                "residual_target_ratio_inner": (
                    residual_mean / target_residual_mean
                    if np.isfinite(residual_mean) and np.isfinite(target_residual_mean) and target_residual_mean > 0
                    else float("nan")),
                "activation_status": activation_status,
            })

            image_id = Path(filename).stem
            if args.build_roi_manifest and image_id in hard_case_ids:
                boxes = build_hard_case_rois(
                    rgb_restored, rgb_gt, mask_gt,
                    roi_size=args.roi_size, roi_count=args.roi_count)
                roi_manifest[image_id] = {
                    "source": "v42_baseline_gt_fixed",
                    "image_size": [int(W), int(H)],
                    "boxes": boxes,
                }
                save_roi_manifest(args.roi_manifest, roi_manifest)
            if image_id in roi_manifest:
                boxes = roi_manifest[image_id].get("boxes", [])
                roi_rows.extend(compute_roi_rows(
                    filename, rgb_restored, rgb_gt, mask_gt, boxes))
                for roi_index, (x0, y0, x1, y1) in enumerate(boxes):
                    roi_dir = Path(args.result_dir) / "roi" / image_id
                    _save_rgb(roi_dir / ("roi_{:02d}_output.png".format(roi_index)),
                              rgb_restored[y0:y1, x0:x1])
                    _save_rgb(roi_dir / ("roi_{:02d}_gt.png".format(roi_index)),
                              rgb_gt[y0:y1, x0:x1])
            if (args.save_v68_aux and merged_aux and
                    (not args.save_v68_aux_hard_only or image_id in hard_case_ids)):
                save_v68_aux_images(args.result_dir, filename, merged_aux, rgb_gt, input_image=rgb_input)

    metrics = mean_metric_lists(metric_lists)
    metrics.update(_mean_dynamic_metric_lists(relation_metric_lists))
    metrics.update(_mean_dynamic_metric_lists(texture_metric_lists))
    for line in legacy_metric_lines(metrics):
        print(line)

    method_name = args.metric_name or weight_path.stem
    print("")
    print(format_metric_table(method_name, metrics))
    relation_table = format_relation_metric_table(method_name, metrics)
    if relation_table:
        print("")
        print(relation_table)
    print("")
    print("Texture detail metrics (lower is better):")
    for key in ("hfen_shadow", "grad_l1_shadow", "energy_error_shadow", "boundary_grad_error"):
        print("  {}: {:.8f}".format(key, metrics[key]))

    if args.save_csv:
        save_metrics_csv(args.save_csv, metrics)
        print("Saved CSV:", args.save_csv)
    if args.save_per_image_csv:
        save_dict_rows_csv(args.save_per_image_csv, per_image_rows)
        print("Saved per-image CSV:", args.save_per_image_csv)
    if roi_rows:
        roi_csv = os.path.join(args.result_dir, "roi_metrics.csv")
        save_dict_rows_csv(roi_csv, roi_rows)
        print("Saved ROI CSV:", roi_csv)
    if args.build_roi_manifest:
        save_roi_manifest(args.roi_manifest, roi_manifest)
        missing = sorted(hard_case_ids - set(roi_manifest))
        if missing:
            raise RuntimeError("Hard-case ROI images were not found: {}".format(", ".join(missing)))
        print("Saved ROI manifest:", args.roi_manifest)


if __name__ == "__main__":
    main()
