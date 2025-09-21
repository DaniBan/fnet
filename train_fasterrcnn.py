import json
import logging
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import torch  # root package
import torch.nn as nn  # neural networks
import torch.optim as optim  # optimizers e.g. gradient descent, ADAM, etc.
from torch.amp import autocast, GradScaler
from torch.utils.data import DataLoader  # dataset representation and loading
from torchmetrics.detection.mean_ap import MeanAveragePrecision
from torchmetrics.metric import Metric
from torchvision.models.detection.faster_rcnn import fasterrcnn_resnet50_fpn, FasterRCNN_ResNet50_FPN_Weights
from tqdm import tqdm
from yacs.config import CfgNode

from src.dataset.fddb.fddb import FDDBDataset
from src.dataset.fddb.utils import split_train_test
from src.utils.fasterrcnn import plot_results
from src.utils.train import save_state

# Configure logging
logging.basicConfig(
    level=logging.INFO,  # Set the minimum logging level
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",  # Format for log messages
)
logger = logging.getLogger(__name__)

scaler = GradScaler(device="cuda")  # Initialize the gradient scaler


def _move_data_to_device(features, target, device):
    features = tuple(feat.to(device, non_blocking=True) for feat in features)
    target = [{k: v.to(device, non_blocking=True) for k, v in t.items()} for t in target]

    return features, target


def train_step(model: nn.Module,
               dataloader: DataLoader,
               optimizer: optim.Optimizer,
               device: torch.device):
    model.train()
    loss_dict = {
        "loss_classifier": 0,
        "loss_box_reg": 0,
        "loss_objectness": 0,
        "loss_rpn_box_reg": 0,
        "loss_total": 0
    }
    for X, y in dataloader:
        # Move data to device
        X, y = _move_data_to_device(X, y, device)

        # Compute prediction and loss
        with autocast(device_type="cuda" if device.type == "cuda" else "cpu"):
            pred_loss = model(X, y)
            loss = 0
            for k, v in pred_loss.items():
                loss_dict[k] += v.item()
                loss += v
            # loss = sum(loss for loss in pred_loss.values())

        # Backpropagation
        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        loss_dict["loss_total"] += loss.item()

    for k, v in loss_dict.items():
        loss_dict[k] /= len(dataloader)
    return loss_dict


def test_step(model: nn.Module,
              dataloader: DataLoader,
              metric: Metric,
              device: torch.device):
    model.eval()
    with torch.no_grad():
        for batch, (X, y) in enumerate(dataloader):
            # Move data to device
            X, y = _move_data_to_device(X, y, device)

            with autocast(device_type="cuda" if device.type == "cuda" else "cpu"):
                y_pred = model(X)
                metric.update(y_pred, y)

    return metric.compute()


def custom_collate_fn(batch):
    return tuple(zip(*batch))


def main():
    # Load configurations
    with open("config/fddb.yml") as f:
        config = CfgNode.load_cfg(f)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Load data
    train_data, test_data = split_train_test(Path(config.target_dir))
    logger.info(f"# train: {len(train_data)} | # test: {len(test_data)}")

    new_size = (450, 450)  # (H, W)
    train_dataset = FDDBDataset(train_data, to_tensor=True, target_size=new_size)
    test_dataset = FDDBDataset(test_data, to_tensor=True, target_size=new_size)

    train_dataloader = DataLoader(dataset=train_dataset,
                                  batch_size=config.batch_size,
                                  num_workers=config.num_workers,
                                  shuffle=False,
                                  pin_memory=device.type == "cuda",
                                  collate_fn=custom_collate_fn)
    test_dataloader = DataLoader(dataset=test_dataset,
                                 batch_size=config.batch_size,
                                 num_workers=config.num_workers,
                                 pin_memory=device.type == "cuda",
                                 collate_fn=custom_collate_fn)

    # Load model
    model = fasterrcnn_resnet50_fpn(weights=FasterRCNN_ResNet50_FPN_Weights.DEFAULT)
    model = model.to(device)

    # Freeze feature extractor layers
    for param in model.backbone.parameters():
        param.requires_grad = False

    # Set up the optimizer
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=config.lr, momentum=config.momentum, weight_decay=config.weight_decay)

    # Learning rate scheduler
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10)

    # Set up test metric
    metric = MeanAveragePrecision(iou_type="bbox")

    results = {
        "train_loss": defaultdict(list),
        "test_mean_ap": defaultdict(list)
    }
    current_datetime = datetime.now().strftime("%y_%m_%dT%H_%M_%S")
    for epoch in tqdm(range(config.num_epochs)):
        train_loss = train_step(model, dataloader=train_dataloader, optimizer=optimizer, device=device)
        for key, value in train_loss.items():
            results["train_loss"][key].append(value)

        mean_ap = test_step(model, dataloader=test_dataloader, metric=metric, device=device)
        for key, value in mean_ap.items():
            results["test_mean_ap"][key].append(value.tolist())

        lr_scheduler.step()

        # Save model snapshot
        if (epoch + 1) % 10 == 0 and epoch != config.num_epochs - 1:
            save_state(model.state_dict(), config, model_tag=f"fasterrcnn_resnet50_fpn",
                       experiment_tag=current_datetime, snapshot_tag=f"epoch_{epoch}", results=results)
            logger.info(f"mAp at epoch {epoch}:\n{results["test_mean_ap"]}")

    logger.info(f"Results:\n{json.dumps(results, indent=4)}")
    plot_results(results)
    save_state(model.state_dict(), config, model_tag="fasterrcnn_resnet50_fpn", experiment_tag=current_datetime,
               results=results)


if __name__ == "__main__":
    main()
