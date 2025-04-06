from pathlib import Path

from PIL import Image
from torch.utils.data import Dataset


class FDDBDataset(Dataset):

    def __init__(self, target_dir: Path, transform=None, target_transform=None):
        if not target_dir.exists():
            raise FileNotFoundError(f"'{target_dir}' not found")
        if not target_dir.is_dir():
            raise NotADirectoryError(f"'{target_dir}' exists but is not a directory")

        self.paths = []
        self.transform = transform
        self.target_transform = transform
        self.img_source_dir = target_dir / "images"

        if not self.img_source_dir.exists():
            raise FileNotFoundError(f"No 'images' subdirectory found in '{target_dir}'")
        if not self.img_source_dir.is_dir():
            raise NotADirectoryError(f"'{self.img_source_dir}' exists but is not a directory")

        self.annotations = {}
        curr_img = None
        with open(target_dir / "label.txt", "r") as f:
            for line in f:
                line = line.strip()

                if not line:
                    continue

                if line.startswith("#"):
                    curr_img = Path(line.lstrip("# "))
                    self.paths.append(curr_img)
                    self.annotations[curr_img] = []
                else:
                    bbox = list(map(int, line.split()))
                    self.annotations[curr_img].append(bbox)

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index: int):
        img_path: Path = self.paths[index]
        img = Image.open(self.img_source_dir / img_path)
        annotation = self.annotations.get(img_path, [])

        return img if self.transform is None else self.transform(img), annotation
