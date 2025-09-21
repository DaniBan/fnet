from pathlib import Path

import numpy as np
import torch
from PIL import Image
from PIL import ImageOps
from torch.utils.data import Dataset
from torchvision.transforms import ToTensor

def _resize_with_aspect_ratio(image: Image.Image, bboxes: list[list[int]], target_size: tuple[int, int]):
    target_w, target_h = target_size
    img_w, img_h = image.size

    scale = min(target_w / img_w, target_h / img_h)
    scaled_w, scaled_h = int(img_w * scale), int(img_h * scale)

    image = image.resize((scaled_w, scaled_h))

    delta_w, delta_h = target_w - scaled_w, target_h - scaled_h
    left, top = delta_w // 2, delta_h // 2
    right, bottom = delta_w - left, delta_h - top

    _bboxes = [[
        round(x1 * scale) + left,
        round(y1 * scale) + top,
        round(x2 * scale) + left,
        round(y2 * scale) + top
    ] for x1, y1, x2, y2 in bboxes]

    return ImageOps.expand(image, border=(left, top, right, bottom)), _bboxes


class FDDBDataset(Dataset):

    def __init__(self, data: dict[Path, list[list[int]]], to_tensor=False, target_size=None):
        self.paths = list(data.keys())
        self.to_tensor = to_tensor
        self.target_size = target_size
        self.annotations = data

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index: int):
        img_path = self.paths[index]
        img = Image.open(img_path)
        bboxes = self.annotations.get(img_path, [])

        if self.target_size is not None:
            img, bboxes = _resize_with_aspect_ratio(img, bboxes, target_size=self.target_size)

        # Build target
        labels = np.array([1] * len(bboxes))
        target = {
            "boxes": torch.tensor(bboxes, dtype=torch.float32) if self.to_tensor else bboxes,
            "labels": torch.tensor(labels, dtype=torch.int64) if self.to_tensor else labels
        }

        if self.to_tensor:
            to_tensor = ToTensor()
            img = to_tensor(img)

        return img, target
