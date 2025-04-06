from timeit import default_timer as timer
from torch import Tensor
from typing import Dict, List

import matplotlib.pyplot as plt
import torch.utils.data.dataset


def print_image(image: Tensor):
    image_reshaped = image.permute([1, 2, 0])
    plt.figure()
    plt.imshow(image_reshaped, cmap='gray')
    plt.axis(False)
    plt.show()


def display_item(dataset: torch.utils.data.dataset.Dataset,
                 i: int):
    """
    Displays an image from a dataset at a specified index and scatters the facial point labels.

    :param dataset: torch dataset
    :param i: index
    """
    start = timer()

    img, label = dataset[i]
    img_adjust = img.permute(1, 2, 0)

    label_x = label[::2]
    label_y = label[1::2]

    end = timer()
    print(f"Time to get item from dataset: {end - start:.4f}")

    print(label_x)
    print(label_y)
    plt.figure(figsize=(9, 9))
    plt.imshow(img_adjust)
    plt.scatter(label_x, label_y, s=8, c="r")
    plt.axis("off")
    plt.show()


def display_random_items(dataset: torch.utils.data.dataset.Dataset, n=2, m=2, seed=42):
    """
    Displays random images on n rows and m cols and scatters facial point labels.

    :param dataset: torch dataset
    :param n: number of rows
    :param m: number of cols
    :param seed: random seed
    """
    torch.manual_seed(seed)
    indexes = torch.randint(0, len(dataset), size=[n * m]).tolist()
    fig = plt.figure(figsize=(9, 9))

    for i in range(n * m):
        img, label = dataset[indexes[i]]
        img_adjust = img.permute(1, 2, 0)
        label_x = label[::2]
        label_y = label[1::2]

        fig.add_subplot(n, m, i + 1)
        plt.imshow(img_adjust, cmap="gray")
        plt.scatter(label_x, label_y, s=8, c="r")
        plt.axis(False)

    plt.show()


def plot_prediction(img, preds):
    img_adjust = img.permute(1, 2, 0)

    label_x = preds[::2]
    label_y = preds[1::2]
    print(label_x)
    print(label_y)
    plt.figure(figsize=(9, 9))
    plt.imshow(img_adjust)
    plt.scatter(label_x, label_y, s=8, c="r")
    plt.axis("off")
    plt.show()


def plot_loss_curves(results: Dict[str, List[float]]):
    plt.figure(figsize=(9, 9))
    train_loss = results["train_loss"]
    test_loss = results["test_loss"]

    epochs = range(len(train_loss))

    plt.plot(epochs, train_loss, label="train loss")
    plt.plot(epochs, test_loss, label="test loss")
    plt.title("Loss")
    plt.xlabel("Epochs")
    plt.legend()

    plt.show()


def plot_lr(lr: list):
    epochs = range(len(lr))
    plt.figure(figsize=(4, 8))
    plt.plot(epochs, lr, label="learning rate")
    plt.xlabel("Epochs")
    plt.legend()

    plt.show()


def plot_results(results: Dict[str, List[float]], lr):
    fig = plt.figure(figsize=(9, 9))
    train_loss = results["train_loss"]
    test_loss = results["test_loss"]

    epochs = range(len(train_loss))

    fig.add_subplot(2, 1, 1)
    plt.plot(epochs, train_loss, label="train loss")
    plt.plot(epochs, test_loss, label="test loss")
    plt.title("Loss")
    plt.xlabel("Epochs")
    plt.legend()

    fig.add_subplot(2, 1, 2)
    # plt.figure(figsize=(8, 4))
    plt.plot(epochs, lr, label="learning rate")
    plt.xlabel("Epochs")
    plt.legend()

    plt.show()
    pass
