import json
import os
from datetime import datetime
from pathlib import Path

import torch


def load_config() -> dict:
    """
    Loads configuration settings from a JSON file.

    This function reads configuration data from a JSON file named `configs.json`
    located in the `config` directory under the current working directory.
    The configurations are returned as a Python dictionary.
    """
    with open(os.path.join(os.getcwd(), "config", "configs.json")) as f:
        configs = json.load(f)

    return configs


def save_state(state_dict: dict,
               config: dict,
               model_tag: str,
               experiment_tag: str | None = None,
               snapshot_tag: str | None = None,
               results: dict | None = None) -> None:
    """
    Saves the model state, configuration, and optional results to disk in a structured directory format, creating
    necessary directories if they do not exist. The function ensures that snapshots are organized under specific
    tags for models, experiments, and snapshots, allowing for easy retrieval and management of saved states.

    Args:
        state_dict (dict): A dictionary containing the model's state, typically obtained from PyTorch's `state_dict()`
            method.
        config (dict): A dictionary containing the model and training configuration settings.
        model_tag (str): The tag or identifier for the model, used as part of the directory structure for saving files.
        experiment_tag (str | None, optional): An optional tag to identify the specific experiment. If not provided,
            the current timestamp is used to create a unique tag format (YY_MM_DD__HH_MM_SS).
        snapshot_tag (str | None, optional): An optional tag to label the specific snapshot within the experiment. If not
            provided, no snapshot subdirectory is created, and files are saved directly under the experiment directory.
        results (dict | None, optional): An optional dictionary containing results or additional metadata to be saved.
            If provided, it will be saved as a JSON file in the same directory.
    """
    state_path = Path("states")
    state_path.mkdir(parents=True,
                     exist_ok=True)

    if experiment_tag is None:
        experiment_tag = datetime.now().strftime("%y_%m_%d__%H_%M_%S")

    if snapshot_tag is not None:
        snapshot_path = state_path / model_tag / experiment_tag / snapshot_tag
    else:
        snapshot_path = state_path / model_tag / experiment_tag

    snapshot_path.mkdir(parents=True,
                        exist_ok=True)

    model_save_path = snapshot_path / "state.pth"

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
