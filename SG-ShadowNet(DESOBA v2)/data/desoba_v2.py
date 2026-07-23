import glob
import json
import os
import random

import cv2
import numpy as np
import torch
from skimage import color, io
from torch.utils.data import Dataset


def _rgb(path):
    image = io.imread(path)
    if image.ndim == 2:
        image = np.repeat(image[..., None], 3, axis=2)
    if image.shape[2] == 4:
        image = image[:, :, :3]
    if image.dtype != np.uint8:
        image = np.clip(image * (255.0 if image.max() <= 1.0 else 1.0), 0, 255).astype(np.uint8)
    return image


def rgb_to_lab_tensor(image):
    lab = color.rgb2lab(image).astype(np.float32)
    lab[:, :, 0] = lab[:, :, 0] / 50.0 - 1.0
    lab[:, :, 1:] = 2.0 * (lab[:, :, 1:] + 128.0) / 255.0 - 1.0
    return torch.from_numpy(lab.transpose(2, 0, 1).copy()).float()


def lab_tensor_to_rgb(tensor):
    lab = tensor.detach().float().cpu().numpy().transpose(1, 2, 0).copy()
    lab[:, :, 0] = 50.0 * (lab[:, :, 0] + 1.0)
    lab[:, :, 1:] = 255.0 * (lab[:, :, 1:] + 1.0) / 2.0 - 128.0
    return np.clip(color.lab2rgb(lab), 0.0, 1.0).astype(np.float32)


class DESOBAv2Dataset(Dataset):
    """Paired DESOBA v2 loader using the official annotation JSON split field."""

    def __init__(self, annotation_root, image_root, split="train", augment=False, dilate_size=51):
        self.annotation_root = annotation_root
        self.image_root = image_root
        self.augment = augment
        self.dilate_size = int(dilate_size)
        annotation_dir = os.path.join(annotation_root, "annotations")
        self.samples = []
        for path in sorted(glob.glob(os.path.join(annotation_dir, "*.json"))):
            with open(path, "r", encoding="utf-8") as handle:
                record = json.load(handle)
            if str(record.get("split", "")).lower() != split.lower():
                continue
            name = record.get("source_file") or os.path.basename(record["source_shadow_file"])
            mask_paths = [os.path.join(annotation_root, pair["shadow_mask"])
                          for pair in record.get("pairs", []) if pair.get("shadow_mask")]
            self.samples.append({"name": name, "mask_paths": mask_paths})
        if not self.samples:
            raise RuntimeError("No DESOBA v2 samples found for split={!r} under {}".format(split, annotation_dir))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        sample = self.samples[index]
        name = sample["name"]
        shadow = _rgb(os.path.join(self.image_root, "shadow_imgs", name))
        target = _rgb(os.path.join(self.image_root, "shadowfree_imgs", name))
        # Match the reference DataLoaderShadowRelation global-removal protocol:
        # union all per-pair PNG shadow masks from the extended annotations.
        mask = np.zeros(shadow.shape[:2], dtype=np.float32)
        for mask_path in sample["mask_paths"]:
            mask_raw = io.imread(mask_path)
            if mask_raw.ndim == 3:
                mask_raw = mask_raw[:, :, 0]
            threshold = 0.5 if np.issubdtype(mask_raw.dtype, np.floating) else 127
            mask = np.maximum(mask, (mask_raw > threshold).astype(np.float32))

        if self.augment and random.random() < 0.5:
            shadow = np.fliplr(shadow).copy()
            target = np.fliplr(target).copy()
            mask = np.fliplr(mask).copy()

        mask_dilated = mask
        if self.dilate_size > 1:
            kernel = np.ones((self.dilate_size, self.dilate_size), dtype=np.uint8)
            mask_dilated = cv2.dilate(mask.astype(np.uint8), kernel).astype(np.float32)

        return {
            "shadow": rgb_to_lab_tensor(shadow),
            "target": rgb_to_lab_tensor(target),
            "mask": torch.from_numpy(mask[None].copy()).float(),
            "mask_dilated": torch.from_numpy(mask_dilated[None].copy()).float(),
            "name": name,
        }
