import yaml
import os
import json
import torch


def load_config(config_path="config.yaml"):
    with open(config_path, "r") as file:
        config = yaml.safe_load(file)
    return config


def ensure_dir(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)


def save_json(data, file_path):
    with open(file_path, "w") as file:
        json.dump(data, file, indent=4)


def print_config(config):
    print("Loaded configuration:")
    for section, values in config.items():
        print(f"\n[{section}]")
        if isinstance(values, dict):
            for key, value in values.items():
                print(f"{key}: {value}")
        else:
            print(values)


def get_torch_device():
    if torch.cuda.is_available():
        return torch.device("cuda")

    try:
        import torch_directml

        return torch_directml.device()
    except ImportError:
        pass

    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")

    return torch.device("cpu")
