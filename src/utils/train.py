import json
import os
from datetime import datetime
from pathlib import Path

import torch

_DEFAULT_MODEL_FILENAME = "model.pth"
_CONFIG_FILENAME = "config.json"
_RESULTS_FILENAME = "results.json"


def _save_json_file(data: dict, file_path: Path):
    """Save a dictionary as a JSON file."""
    with open(file_path, "w") as file:
        json.dump(data, file, indent=2)  # noqa


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
    Saves the current state of the model, including configurations and results,
    into a structured directory format. This method helps in managing model
    snapshots effectively for experiments and checkpoints.

    Args:
        state_dict (dict): The state dictionary of the model to be saved.
            It typically contains model weights and buffers.
        config (dict): The configuration dictionary containing essential
            parameters and settings related to the model or experiment.
        model_tag (str): A string identifier for the model. This is used to
            organize snapshots under a specific model directory.
        experiment_tag (str | None): A custom string identifier for the
            experiment directory. If not provided, a timestamped tag is
            automatically generated.
        snapshot_tag (str | None): An optional tag to name the snapshot file
            containing the model weights and configuration. Defaults to
            "_DEFAULT_MODEL_FILENAME" if not provided.
        results (dict | None): An optional dictionary containing results,
            metrics, or additional outcomes to be saved alongside the model
            and configuration. If not provided, no results are stored.
    """
    # Define experiment-related paths
    state_path = Path("states")
    state_path.mkdir(parents=True,
                     exist_ok=True)

    if experiment_tag is None:
        experiment_tag = datetime.now().strftime("%y_%m_%d__%H_%M_%S")

    snapshot_path = state_path / model_tag / experiment_tag
    snapshot_path.mkdir(parents=True,
                        exist_ok=True)

    # Save the state dict
    model_filename = f"{snapshot_tag}.pth" if snapshot_tag else _DEFAULT_MODEL_FILENAME
    model_save_path = snapshot_path / model_filename

    torch.save(obj=state_dict,
               f=model_save_path)

    # Save configs
    config_path = snapshot_path / _CONFIG_FILENAME
    _save_json_file(config, config_path)

    # Save results if any
    if results is not None:
        results_path = snapshot_path / _RESULTS_FILENAME
        _save_json_file(results, results_path)
