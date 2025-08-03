from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import Dataset


class FDDBDataset(Dataset):

    def __init__(self, data: dict[Path, list[list[int]]], transform=None, target_shape=None):
        self.paths = list(data.keys())
        self.transform = transform
        self.target_shape = target_shape
        self.annotations = data

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index: int):
        img_path = self.paths[index]
        img = Image.open(img_path)
        bboxes = self.annotations.get(img_path, [])
        labels = torch.tensor([0] * len(bboxes), dtype=torch.int64)

        if self.target_shape is not None:
            orig_size = torch.tensor(img.size)
            new_size = torch.tensor(self.target_shape)
            scale_factor = new_size / orig_size
            bboxes = torch.Tensor(bboxes) * torch.cat((scale_factor, scale_factor))

        target = {"boxes": bboxes, "labels": labels}

        return img if self.transform is None else self.transform(img), target
