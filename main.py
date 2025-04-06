import json
import os
from datetime import datetime
from pathlib import Path
from timeit import default_timer as timer

import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data
from torchvision import transforms
from tqdm import tqdm

from src.dataset.lfw.data_factory import build_datasets
from src.models.vgg import TinyVGG
from src.scripts.dataloader import build_dataloader
from src.utils.visualization import plot_results
from src.models.resnet import res_net_50

from src.models.losses import MSExp
from yacs.config import CfgNode

def train_step(model: nn.Module,
               dataloader: torch.utils.data.DataLoader,
               loss_fn: nn.Module,
               optimizer: optim.Optimizer,
               device=None):
    model.train()
    total_loss = 0
    if device is None:
        device = torch.device("cpu")

    for batch, (X, y) in enumerate(dataloader):
        # Move data to device
        X, y = X.to(device), y.to(device)

        y_pred = model(X)
        loss = loss_fn(y_pred, y)
        total_loss += loss.item()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    total_loss = total_loss / len(dataloader)
    return total_loss


def test_step(model: nn.Module,
              dataloader: torch.utils.data.DataLoader,
              loss_fn,
              device=None):
    model.eval()
    total_loss = 0
    if device is None:
        device = "cpu"

    with torch.inference_mode():
        for batch, (X, y) in enumerate(dataloader):
            # Move data to device
            X, y = X.to(device), y.to(device)

            y_pred = model(X)
            loss = loss_fn(y_pred, y)
            total_loss += loss.item()

    total_loss = total_loss / len(dataloader)
    return total_loss


def save_state(state_dict, config, results=None, tag=None):
    state_path = Path("states")
    state_path.mkdir(parents=True,
                     exist_ok=True)

    state_name = datetime.now().strftime("%y_%m_%d__%H_%M_%S")
    snapshot_path = state_path / Path(state_name)
    snapshot_path.mkdir(parents=True,
                        exist_ok=True)

    if tag is None:
        tag = "model"
    model_save_path = snapshot_path / f"{tag}.pth"

    # save state dict
    torch.save(obj=state_dict,
               f=model_save_path)

    # save config
    with open(os.path.join(snapshot_path, "config.json"), "w") as f:
        json.dump(config, f, indent=2)

    # save results
    if results is not None:
        with open(os.path.join(snapshot_path, "results"), "w") as f:
            json.dump(results, f, indent=2)

def main():
    # Fetch dataset
    data_transform = transforms.Compose([
        transforms.ToTensor()
    ])
    train_dataset, test_dataset = build_datasets(target_dir="data",
                                                 transform_train=data_transform,
                                                 transform_test=data_transform)

    # Load configurations
    with open("config/fddb.yml") as f:
        config = CfgNode.load_cfg(f)



    # Create train and test dataloaders
    train_dataloader = build_dataloader(train_dataset,
                                        batch_size=config.batch_size,
                                        num_workers=config.num_workers,
                                        shuffle=True)
    test_dataloader = build_dataloader(test_dataset,
                                       batch_size=config.batch_size,
                                       num_workers=config.num_workers)

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    # model_tiny_vgg = TinyVGG(input_shape=3, hidden_units=10, output_shape=18).to(device=device)
    # model_vgg16 = VGG16(input_shape=3, output_shape=18).to(device=device)
    model_rn50 = res_net_50()

    # Set criterion and optimizer
    loss_fn = MSExp(1.05)
    # optimizer = optim.Adam(params=model_rn50.parameters(), lr=lr)
    optimizer = optim.SGD(params=model_rn50.parameters(), lr=config.lr)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=config.lr_step, gamma=config.lr_gamma)

    results = {
        "train_loss": [],
        "test_loss": [],
        "time": {
            "seconds": 0.0,
            "minutes": 0.0
        }
    }
    lr = []

    start_time = timer()
    for epoch in tqdm(range(config.num_epochs)):
        train_loss = train_step(model_rn50, train_dataloader, loss_fn, optimizer=optimizer, device=device)
        test_loss = test_step(model_rn50, test_dataloader, loss_fn, device=device)
        lr.append(scheduler.get_last_lr())
        scheduler.step()

        print(f"Epoch: {epoch}")
        print(f"Train loss: {train_loss}")
        print(f"Test loss: {test_loss}")
        print(f"Learning rate: {scheduler.get_last_lr()}")

        results["train_loss"].append(train_loss)
        results["test_loss"].append(test_loss)

    end_time = timer()
    print("========================\nTrain time:")
    print(f"Seconds: {end_time - start_time:.3f}")
    print(f"Minutes: {(end_time - start_time) / 60:.2f}")
    results["time"]["seconds"] = end_time - start_time
    results["time"]["minutes"] = (end_time - start_time) / 60

    tag = "tinyVgg" if isinstance(model_rn50, TinyVGG) else "resnet50"
    save_state(model_rn50.state_dict(), config, results, tag)
    plot_results(results, lr)


if __name__ == "__main__":
    main()
