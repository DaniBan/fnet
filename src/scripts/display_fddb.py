from src.dataset.fddb.fddb import FDDBDataset
from src.dataset.fddb.utils import display_random_items, display_item
from pathlib import Path
from torchvision import transforms

if __name__ == "__main__":
    data_path = Path("C:\\Users\\big_b\\.cache\\kagglehub\\datasets\\ngoduy\\dataset-for-face-detection\\versions\\1\\"
                     "Dataset_FDDB\\Dataset_FDDB")

    data_transform = transforms.Compose([
        transforms.ToTensor()
    ])

    dataset: FDDBDataset = FDDBDataset(target_dir=data_path, transform=data_transform, target_transform=data_transform)
    display_random_items(dataset, n=4, m=4, seed=42)
    # display_item(dataset, 1)
