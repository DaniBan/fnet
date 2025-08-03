import random
from pathlib import Path
from typing import Tuple, List, Dict

import matplotlib.patches as patches
import matplotlib.pyplot as plt
import torch
from torch.utils.data import Dataset

LABEL_FILE_NAME = "label.txt"


def display_item(dataset: Dataset, index: int):
    img, target = dataset[index]
    bboxes = target["boxes"].tolist()
    img = img.permute(1, 2, 0)
    fig, ax = plt.subplots(1, figsize=(9, 9))
    ax.imshow(img)
    for bbox in bboxes:
        x1, y1, x2, y2 = bbox
        rect = patches.Rectangle((x1, y1), x2 - x1, y2 - y1, linewidth=2, edgecolor="green", fill=False)
        ax.add_patch(rect)

    ax.set_xticks([])
    ax.set_yticks([])
    plt.show()


def display_random_items(dataset: Dataset, n: int = 2, m: int = 2, seed: int = 42):
    torch.manual_seed(seed)
    indexes = torch.randint(0, len(dataset), size=[n * m]).tolist()

    fig = plt.figure(figsize=(9, 9))

    for i, idx in enumerate(indexes):
        img, target = dataset[idx]
        bboxes = target["boxes"].tolist()
        img = img.permute(1, 2, 0)
        ax = fig.add_subplot(n, m, i + 1)

        ax.imshow(img)
        ax.set_xticks([])
        ax.set_yticks([])

        for bbox in bboxes:
            x1, y1, x2, y2 = bbox
            rect = patches.Rectangle((x1, y1), x2 - x1, y2 - y1, linewidth=2, edgecolor="green", fill=False)
            ax.add_patch(rect)

    plt.show()


def split_train_test(src: Path, train_ratio: float = 0.8, shuffle: bool = False, seed: int = None):
    """
    Splits a dataset of image files and their annotations into training and testing sets.

    This function divides a set of images and their corresponding annotations into two
    subsets: training and testing, based on a given ratio. It supports optional shuffling
    of the dataset to ensure randomness in the split.

    Args:
        src (Path): The path to the directory containing image files and annotations.
        train_ratio (float, optional): The ratio of the dataset to allocate for training.
            Must be between 0.0 and 1.0. Defaults to 0.8.
        shuffle (bool, optional): Whether to shuffle the dataset before splitting.
            Defaults to False.
        seed (int, optional): The seed value for the random number generator, used when
            shuffle is True to ensure deterministic results. Defaults to None.

    Raises:
        FileNotFoundError: If the source directory does not exist.
        NotADirectoryError: If the provided source path is not a directory.
        ValueError: If the train_ratio is not within the range [0.0, 1.0].

    Returns:
        Tuple[Dict[Path, Any], Dict[Path, Any]]: A tuple of two dictionaries. The first
        dictionary contains the training data with image paths as keys and annotations as
        values. The second dictionary contains the testing data with the same format.
    """
    if not src.exists():
        raise FileNotFoundError(f"Source '{src}' not found.")
    if not src.is_dir():
        raise NotADirectoryError(f"Source '{src}' is not a directory.")
    if not 0.0 <= train_ratio <= 1.0:
        raise ValueError("Split ratio must be between 0.0 and 1.0.")

    image_paths, image_annotations = _load_annotations(src)

    if shuffle:
        random_ = random.Random(seed)
        random_.shuffle(image_paths)

    # Split data
    split_idx = int(len(image_paths) * train_ratio)
    train_paths = image_paths[:split_idx]
    test_paths = image_paths[split_idx:]

    train_data = {path: image_annotations[path] for path in train_paths}
    test_data = {path: image_annotations[path] for path in test_paths}

    return train_data, test_data


def _load_annotations(src: Path) -> Tuple[List[Path], Dict[Path, List[List[int]]]]:
    """Loads annotations from the `label.txt` file."""
    image_paths = []
    image_annotations = {}
    curr_img = None
    with open(src / LABEL_FILE_NAME, "r") as f:
        for line in f:
            line = line.strip()
            if line.startswith("#"):
                curr_img = src / "images" / Path(line.lstrip("# "))
                image_paths.append(curr_img)
                image_annotations[curr_img] = []
            else:
                bbox = list(map(int, line.split()))
                image_annotations[curr_img].append(bbox)
    return image_paths, image_annotations
