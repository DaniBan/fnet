import pathlib
import pandas as pd
import os

from PIL import Image
from torch import Tensor
from torch.utils.data import Dataset


class FaceDataset(Dataset):

    def __init__(self, target_dir, transform=None, target_transform=None):
        self.paths = list(pathlib.Path(target_dir).glob("images/*.jpg"))
        self.label_df = pd.read_csv(os.path.join(target_dir, "labels", "labels.csv"))
        self.transform = transform
        self.target_transform = target_transform

    def load_image(self, index: int) -> Image.Image:
        image_path = self.paths[index]
        return Image.open(image_path)

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index) -> (Tensor, Tensor):
        img = self.load_image(index)
        w, h = img.size
        label = Tensor(self.label_df.loc[index, :].values[:18].tolist())
        return img if self.transform is None else self.transform(img), \
            label if self.target_transform is None else self.target_transform(label, w, h)
