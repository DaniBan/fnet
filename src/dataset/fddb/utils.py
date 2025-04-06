import matplotlib.patches as patches
import matplotlib.pyplot as plt
import torch
from torch.utils.data import Dataset


def display_item(dataset: Dataset, index: int):
    img, bboxes = dataset[index]
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
        img, bboxes = dataset[idx]
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
