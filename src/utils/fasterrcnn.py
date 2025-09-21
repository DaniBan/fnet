import logging
import math

import matplotlib.patches as patches
import matplotlib.pyplot as plt
import torch

logger = logging.getLogger(__name__)


def run_inference(model, image, target, device):
    """
    Runs inference using a pre-trained model on a given data and target.

    This function performs inference on a single data sample using the given model,
    and compares the predicted bounding box with the ground truth information provided
    in the target parameter. It visualizes the results, including the predicted and ground
    truth bounding boxes, using matplotlib.

    Args:
        model: A pre-trained PyTorch model used for inference.
        image: A single input data sample, formatted as a tensor.
        target: A dictionary containing the ground truth bounding box ("boxes") and
            other relevant ground truth information for the input data.
        device: The device to be used for inference (e.g., "cpu" or "cuda").
    """
    model = model.to(device)
    image = image.to(device)
    model.eval()
    with torch.inference_mode():
        y_pred = model(image.unsqueeze(dim=0))

        if y_pred[0]["boxes"].shape[0] == 0:
            logger.info("No bounding boxes found.")
            return

        y_pred = [{
            "boxes": sub_pred["boxes"].cpu(),
            "labels": sub_pred["labels"].cpu(),
            "scores": sub_pred["scores"].cpu()
        } for sub_pred in y_pred]
        pred_elem = y_pred[0]
        logger.info(f"Predicted bounding boxes: {pred_elem['boxes']}")

        # Plot prediction
        image = image.cpu()
        _, ax = plt.subplots(1, figsize=(9, 9))
        ax.imshow(image.permute(1, 2, 0))

        # Predicted bounding boxes
        label_set = False
        for x1, y1, x2, y2 in pred_elem["boxes"][:min(len(target["boxes"]), len(pred_elem["boxes"]))]:
            rect = patches.Rectangle((x1, y1), x2 - x1, y2 - y1, linewidth=2, edgecolor="blue", fill=False,
                                     label="Prediction" if not label_set else None)
            label_set = True
            ax.add_patch(rect)

        # Ground truth bounding boxes
        label_set = False
        for x1, y1, x2, y2 in target["boxes"]:
            rect = patches.Rectangle((x1, y1), x2 - x1, y2 - y1, linewidth=2, edgecolor="green", fill=False,
                                     label="Ground Truth" if not label_set else None)
            label_set = True
            ax.add_patch(rect)

        # Add legend
        ax.legend(loc="upper right")

        ax.set_xticks([])
        ax.set_yticks([])
        plt.show()


def run_inference_n(model: torch.nn.Module, images: list, targets: list[dict], device: torch.device = "cpu"):
    # Move model and data to device
    model = model.to(device)
    for i in range(len(images)):
        images[i] = images[i].to(device)

    # Make predictions
    predictions = []
    model.eval()
    with torch.inference_mode():
        for item in images:
            y_pred = model(item.to(device).unsqueeze(dim=0))
            predictions.append(y_pred[0])

    if device == "cuda":
        for i in range(len(images)):
            images[i] = images[i].cpu()

    # Plot predictions
    n = len(images)
    rows = math.floor(math.sqrt(n))
    while n % rows != 0:
        rows -= 1
    cols = n // rows

    _, ax = plt.subplots(nrows=rows, ncols=cols, figsize=(9, 9))
    for r in range(rows):
        for c in range(cols):
            i = r * cols + c
            if i < n:
                # TODO: solve bug when the number of elements go on 1 line
                ax[r, c].imshow(images[i].permute(1, 2, 0))

                if len(predictions[i]) == 0:
                    continue

                # Predicted bounding boxes
                nb_bboxes_to_plot = min(len(targets[i]["boxes"]), len(predictions[i]["boxes"]))
                for x1, y1, x2, y2 in predictions[i]["boxes"][:nb_bboxes_to_plot]:
                    x1_orig, y1_orig, x2_orig, y2_orig = targets[i]["boxes"][0]
                    rect = patches.Rectangle((x1, y1), x2 - x1, y2 - y1, linewidth=2, edgecolor="blue",
                                             fill=False, label="Prediction")
                    ax[r, c].add_patch(rect)

                # Ground truth bounding boxes
                for x1, y1, x2, y2 in targets[i]["boxes"]:
                    rect = patches.Rectangle((x1, y1), x2 - x1, y2 - y1, linewidth=2, edgecolor="green",
                                              fill=False, label="Ground Truth")
                    ax[r, c].add_patch(rect)

                ax[r, c].set_xticks([])
                ax[r, c].set_yticks([])
    plt.show()


def plot_results(results):
    mean_ap = results["test_mean_ap"]
    train_loss = results["train_loss"]
    map_signals_to_plot = ["map", "map_50", "map_75", "map_small", "map_medium", "map_large"]

    fig = plt.figure(figsize=(16, 9))

    fig.add_subplot(2, 1, 1)
    for key, value in mean_ap.items():
        if key in map_signals_to_plot:
            plt.plot(value, marker="o", label=key)
    plt.xlabel("Epoch")
    plt.ylabel("Mean Average Precision")
    plt.legend()

    fig.add_subplot(2, 1, 2)
    for key, value in train_loss.items():
        plt.plot(value, label=key)
    plt.xlabel("Epoch")
    plt.ylabel("Train Loss")
    plt.legend()

    plt.show()
