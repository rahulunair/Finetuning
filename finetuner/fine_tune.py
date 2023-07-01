import os
import time
from config import torch
from config import device
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from data_loader import data_distribution, plot_data_distribution, show_data
from data_loader import TRAIN_DIR,VALID_DIR
from data_loader import imagenet_stats
from data_loader import img_transforms
from model import FireFinder
from trainer import Trainer
from metrics import Metrics
import matplotlib.pyplot as plt

EPOCHS=0
LR=2e-2
batch_size = 384


# Data loading
train_data = datasets.ImageFolder(TRAIN_DIR, transform=img_transforms["train"])
valid_data = datasets.ImageFolder(VALID_DIR, transform=img_transforms["valid"])
train_dataloader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
valid_dataloader = DataLoader(valid_data, batch_size=batch_size)

# Training
model = FireFinder(simple=True, dropout=0.1)
trainer = Trainer(model, lr=LR, epochs=EPOCHS, device=device)
start = time.time()
val_acc = trainer.fine_tune(train_dataloader, valid_dataloader)

# Save the model
torch.save(model.state_dict(), f"./models/model_acc_{val_acc}_device_{device}_epochs_{EPOCHS}.pt")

# Display the results
print("Training completed.")
print(f"Time elapsed: {time.time() - start} seconds.")
print(f"Best validation accuracy: {val_acc}")

