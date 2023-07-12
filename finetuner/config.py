import os

import torch

# Default to no XPU
HAS_XPU = False


def set_config(device):
    """Attempt to set CPU configuration for torch."""
    global HAS_XPU
    if device == torch.device("xpu"):
        os.environ["IPEX_TILE_AS_DEVICE"] = "0"
        HAS_XPU = True
    try:
        import psutil

        num_physical_cores = psutil.cpu_count(logical=False)
        os.environ["OMP_NUM_THREADS"] = str(num_physical_cores)
        print(f"OMP_NUM_THREADS set to: {num_physical_cores}")
    except ImportError:
        print("psutil not found. Unable to set OMP_NUM_THREADS.")

def set_seed(seed_value=42):
    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    if torch.xpu.is_available():
        torch.xpu.manual_seed(seed_value)
        torch.xpu.manual_seed_all(seed_value)
        torch.backends.mkldnn.deterministic = True
        torch.backends.mkldnn.benchmark = False

def set_device(overide=False,device="cpu"):
    """Attempt to import torch and ipex. Set device depending on availability."""
    if overide:
        return torch.device(device)
    try:
        import torch
        import intel_extension_for_pytorch as ipex
        if torch.xpu.is_available():
            device = f"xpu:{torch.xpu.device_count()}"
            print(f"XPU devices available, using {device}")
            print(f"XPU device name: {torch.xpu.get_device_name(0)}")
        else:
            device = "cpu"
    except ImportError as error:
        print("Failed to import torch / ipex.")
        device = "cpu"
    return torch.device(device)


os.environ["KMP_AFFINITY"] = "granularity=fine,compact,1,0"
os.environ["KMP_BLOCKTIME"] = "1"
device = set_device(False,"") # True, "device_name" if need to set a specific device.
set_config(device)

# Other imports
import fnmatch
import pathlib
import random
import time

import matplotlib.pyplot as plt
import numpy as np
import torch.nn as nn


def seed_everything(seed: int = 4242):
    """set all random seeds using `seed`"""
    print(f"seed set to: {seed}")
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    if HAS_XPU:
        torch.xpu.set_random_seed()


def ncores() -> int:
    """Get number of physical cores"""
    return psutil.cpu_count(logical=False)
