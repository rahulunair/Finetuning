import os
import pathlib
import warnings
import random
import time
import gc

warnings.filterwarnings("ignore")

import torch
import numpy as np
import intel_extension_for_pytorch
import matplotlib.pyplot as plt
import wandb

from torchvision import datasets, transforms
from torch.utils.data import DataLoader

from batch_finder import optimum_batch_size
from config import set_seed, device
from data_loader import (
    TRAIN_DIR,
    VALID_DIR,
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
from lr_finder import LearningRateFinder
from torch import optim


# Constants
EPOCHS = 10
LR = 2.14e-4


def create_dataloader(directory, batch_size, shuffle=False, transform=None):
    data = datasets.ImageFolder(directory, transform=transform)
    return DataLoader(data, batch_size=batch_size, shuffle=shuffle)


def setup_dataloaders(config):
    return create_dataloader(
        TRAIN_DIR, config["batch_size"], shuffle=True, transform=img_transforms["train"]
    ), create_dataloader(
        VALID_DIR, config["batch_size"], transform=img_transforms["valid"]
    )


def find_lr(model, optimizer, dataloader):
    lr_finder = LearningRateFinder(model, optimizer, device)
    best_lr = lr_finder.lr_range_test(dataloader, start_lr=1e-5, end_lr=1e-2)
    return best_lr


def train(model, trainer, config):
    train_dataloader, valid_dataloader = setup_dataloaders(config)
    print("training data")
    plot_data_distribution(data_distribution(train_dataloader.dataset, TRAIN_DIR))
    print("\nvalidation data")
    plot_data_distribution(data_distribution(valid_dataloader.dataset, VALID_DIR))
    print(f"______________")
    start = time.time()
    val_acc = trainer.fine_tune(train_dataloader, valid_dataloader)
    model_save_path = f"./models/model_acc_{val_acc}_device_{device}_lr_{trainer.lr}_epochs_{EPOCHS}.pt"
    torch.save(model.state_dict(), model_save_path)
    print(f"Time elapsed: {time.time() - start} seconds.")


def main(aug_data=False):
    set_seed(42)
    if aug_data:
        print("Augmenting dataset...")
        augment_and_save(TRAIN_DIR)
        augment_and_save(VALID_DIR)
        print("Done Augmenting...")
    model = FireFinder(simple=True, dropout=0.5)
    optimizer = optim.Adam(model.parameters(), lr=LR)
    print(f"Finding optimum batch size...")
    # batch_size = optimum_batch_size(model, input_size)
    batch_size = 128
    train_dataloader = create_dataloader(
        TRAIN_DIR,
        batch_size=batch_size,
        shuffle=True,
        transform=img_transforms["train"],
    )
    print("Finding best init lr...")
    best_lr = find_lr(model, optimizer, train_dataloader)
    print(f"Found best learning rate: {best_lr}")
    del model, optimizer
    gc.collect()
    if device == torch.device("xpu"):
        torch.xpu.empty_cache()
    model = FireFinder(simple=True, dropout=0.5)
    trainer = Trainer(
        model=model,
        optimizer=optim.Adam,
        lr=best_lr,
        epochs=EPOCHS,
        device=device,
        use_wandb=True,
        use_ipex=True
    )
    train(model, trainer, config={"lr": best_lr, "batch_size": batch_size})


if __name__ == "__main__":
    main(aug_data=True)
