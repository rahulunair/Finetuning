import warnings
import os

warnings.filterwarnings("ignore")
os.environ["IPEX_TILE_AS_DEVICE"] = "0"

import torch
import intel_extension_for_pytorch

import numpy as np
import random


def set_seed(seed_value=42):
    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    if torch.xpu.is_available():
        torch.xpu.manual_seed(seed_value)
        torch.xpu.manual_seed_all(seed_value)
        torch.backends.mkldnn.deterministic = True
        torch.backends.mkldnn.benchmark = False


import pathlib

set_seed(42)

import gc
import time

import wandb
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from batch_finder import optimum_batch_size
from config import device
from data_loader import (
    TRAIN_DIR,
    VALID_DIR,
    TRAIN_AUGMENTED_DIR,
    VALID_AUGMENTED_DIR,
    augment_and_save,
    data_distribution,
    imagenet_stats,
    img_transforms,
    plot_data_distribution,
    show_data,
)
from metrics import Metrics
from model import FireFinder
from trainer import Trainer

EPOCHS = 200
LR = 2.14e-4
SWEEP = False


def optm_lr(model, batch_size, dataloader):
    device = "xpu" if torch.xpu.is_available() else "cpu"
    trainer = Trainer(model, lr=LR, epochs=EPOCHS, device=device, use_wandb=True)
    print(f"Attempt to find an optimal learning rate...")
    with wandb.init(project="FireFinder_finetune") as run:
        trainer.lr_range_test(dataloader, start_lr=1e-5, end_lr=1e-2, num_iter=100)
        optimal_lr = trainer.lr
        print(f"Identified optimal LR: {optimal_lr}")
        return optimal_lr


def sweep_config(init_lr=0.0001):
    print(f"Sweep config for learning rate beween {init_lr} to {init_lr * 10}")
    print(f"Fintuning the FireFinder model...")
    sweep_config = {
        "method": "random",
        "metric": {"name": "Validation Acc", "goal": "maximize"},
        "parameters": {
            # "epochs": {"values": [5, 10, 15, 20]},
            "lr": {"values": [init_lr * 100, init_lr * 10, init_lr, init_lr / 10, init_lr / 100]},
        },
    }
    sweep_id = wandb.sweep(sweep_config, project="FireFinder_finetune")
    return sweep_id


def train(model, trainer, config=None):
    if config is None:
        config = {"lr": LR}
    with wandb.init(config=config, project="FireFinder_finetune") as run:
        train_data = datasets.ImageFolder(TRAIN_DIR, transform=img_transforms["train"])
        valid_data = datasets.ImageFolder(VALID_DIR, transform=img_transforms["valid"])
        train_dataloader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
        valid_dataloader = DataLoader(valid_data, batch_size=batch_size)
        data_dir = pathlib.Path("./data/output/")
        show_data(train_dataloader, TRAIN_DIR)
        show_data(valid_dataloader, VALID_DIR)
        train_dist, valid_dist = data_distribution(
            train_data, TRAIN_DIR
        ), data_distribution(valid_data, VALID_DIR)
        plot_data_distribution(train_dist)
        plot_data_distribution(valid_dist)
        print(f"______________")
        lr = config["lr"]
        print(f"  - Sweep: {SWEEP} with LR: {lr}")
        trainer.lr = lr
        print("Default lr being used is: {lr}")
        trainer.update_learning_rate(lr)
        start = time.time()
        val_acc = trainer.fine_tune(train_dataloader, valid_dataloader)
        model_save_path = f"./models/model_acc_{val_acc}_device_{device}_lr_{trainer.lr}_epochs_{EPOCHS}.pt"
        torch.save(
            model.state_dict(),
            model_save_path,
        )
        print(f"Time elapsed: {time.time() - start} seconds.")


def sweep_train(model, input_size, batch_size, valid_dataloader):
    device = "xpu" if torch.xpu.is_available() else "cpu"
    trainer = Trainer(model, lr=LR, epochs=EPOCHS, device=device, use_wandb=True)
    init_lr = optm_lr(model, batch_size, valid_dataloader)
    sweep_id = sweep_config(init_lr)
    wandb.agent(sweep_id, function=train, args=(model, trainer))


if __name__ == "__main__":
    aug = False
    if aug:
        print("Augmenting dataset...")
        augment_and_save(TRAIN_DIR, TRAIN_AUGMENTED_DIR)
        augment_and_save(VALID_DIR, VALID_AUGMENTED_DIR)
        print("Done Augmenting...")
    TRAIN_DIR = TRAIN_AUGMENTED_DIR
    VALID_DIR = VALID_AUGMENTED_DIR
    model = FireFinder(simple=False, dropout=0.5)
    input_size = (3, 224, 224)
    print(f"Finding optimum batch size...")
    # batch_size = optimum_batch_size(model, input_size)
    batch_size = 64
    print(f"Optimal batch size: {batch_size}")
    if SWEEP:
        valid_data = datasets.ImageFolder(VALID_DIR, transform=img_transforms["valid"])
        valid_dataloader = DataLoader(valid_data, batch_size=batch_size)
        show_data(valid_dataloader)
        sweep_train(model, input_size, batch_size, valid_dataloader)
    else:
        print("Running without sweep...")
        device = "xpu" if torch.xpu.is_available() else "cpu"
        trainer = Trainer(model, lr=LR, epochs=EPOCHS, device=device, use_wandb=True)
        train(model, trainer, config={"lr": LR})
