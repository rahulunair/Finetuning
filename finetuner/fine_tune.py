import os
import pathlib
import warnings
import random
import time
import gc

import torch
import numpy as np
import intel_extension_for_pytorch
import matplotlib.pyplot as plt
import wandb

from torchvision import datasets, transforms
from torch.utils.data import DataLoader

from batch_finder import optimum_batch_size
from config import device
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

# Set up environment variables and warning filters at the top
warnings.filterwarnings("ignore")
os.environ["IPEX_TILE_AS_DEVICE"] = "0"

# Constants
EPOCHS = 200
LR = 2.14e-4
SWEEP = False


def set_seed(seed_value=42):
    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    if torch.xpu.is_available():
        torch.xpu.manual_seed(seed_value)
        torch.xpu.manual_seed_all(seed_value)
        torch.backends.mkldnn.deterministic = True
        torch.backends.mkldnn.benchmark = False


def create_dataloader(directory, batch_size, shuffle=False, transform=None):
    data = datasets.ImageFolder(directory, transform=transform)
    return DataLoader(data, batch_size=batch_size, shuffle=shuffle)


def setup_dataloaders(config):
    return create_dataloader(TRAIN_DIR, config["batch_size"], shuffle=True, transform=img_transforms["train"]),\
           create_dataloader(VALID_DIR, config["batch_size"], transform=img_transforms["valid"])


def setup_device():
    return "xpu" if torch.xpu.is_available() else "cpu"


def optm_lr(model, batch_size, dataloader):
    device = setup_device()
    trainer = Trainer(model, lr=LR, epochs=EPOCHS, device=device, use_wandb=True)
    print("Attempt to find an optimal learning rate...")
    with wandb.init(project="FireFinder_finetune"):
        trainer.lr_range_test(dataloader, start_lr=1e-5, end_lr=1e-2, num_iter=100)
        optimal_lr = trainer.lr
        print(f"Identified optimal LR: {optimal_lr}")
        return optimal_lr


def sweep_config(init_lr, batch_size):
    print("Sweeping...")
    print("Fintuning the FireFinder model...")
    return {
        "method": "random",
        "metric": {"name": "Validation Acc", "goal": "maximize"},
        "parameters": {
            "lr": {"values": [init_lr * 100, init_lr * 10, init_lr, init_lr / 10, init_lr / 100]},
            "batch_size": {"value": batch_size},
        },
    }


def sweep_train(model, trainer, init_lr, batch_size):
    sweep_id = wandb.sweep(sweep_config(init_lr, batch_size), project="FireFinder_finetune")
    wandb.agent(sweep_id, function=train, args=(model, trainer))


def train(model, trainer, config):
    train_dataloader, valid_dataloader = setup_dataloaders(config)
    with wandb.init(config=config, project="FireFinder_finetune") as run:
        show_data(train_dataloader, TRAIN_DIR)
        show_data(valid_dataloader, VALID_DIR)
        plot_data_distribution(data_distribution(train_dataloader.dataset, TRAIN_DIR))
        plot_data_distribution(data_distribution(valid_dataloader.dataset, VALID_DIR))
        print(f"______________")
        print(f"Sweep: {SWEEP} with LR: {config['lr']}")
        trainer.update_learning_rate(config["lr"])
        start = time.time()
        val_acc = trainer.fine_tune(train_dataloader, valid_dataloader)
        model_save_path = f"./models/model_acc_{val_acc}_device_{device}_lr_{trainer.lr}_epochs_{EPOCHS}.pt"
        torch.save(model.state_dict(), model_save_path)
        print(f"Time elapsed: {time.time() - start} seconds.")


def main():
    set_seed(42)
    augment_data = False
    if augment_data:
        print("Augmenting dataset...")
        augment_and_save(TRAIN_DIR)
        augment_and_save(VALID_DIR)
        print("Done Augmenting...")
    model = FireFinder(simple=True, dropout=0.5)
    print("Finding optimum batch size...")
    device = setup_device()
    trainer = Trainer(model, lr=LR, epochs=EPOCHS, device=device, use_wandb=True)
    print(f"Finding optimum batch size...")
    # batch_size = optimum_batch_size(model, input_size)
    batch_size = 64
    train_dataloader = create_dataloader(TRAIN_DIR, batch_size=batch_size, shuffle=True, transform=img_transforms["train"])
    init_lr = optm_lr(model, batch_size, train_dataloader)
    if SWEEP:
        sweep_train(model, trainer, init_lr, batch_size)
    else:
        print("Running without sweep...")
        train(model, trainer, config={"lr": init_lr, "batch_size": batch_size})


if __name__ == "__main__":
    main()

