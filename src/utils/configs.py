import json
import os


def load_config():
    with open(os.path.join(os.getcwd(), "config", "configs.json")) as f:
        configs = json.load(f)

    return configs
