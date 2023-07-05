from config import torch
from config import seed_everything

import pathlib

from torchvision import transforms
import matplotlib.pyplot as plt

imagenet_stats = [[0.485, 0.456, 0.406], [0.229, 0.224, 0.225]]
data_dir = pathlib.Path("./data/output/")
TRAIN_DIR = data_dir / "train"
VALID_DIR = data_dir / "val"

# Define transforms for training and validation data.
img_transforms = {
    "train": transforms.Compose(
        [
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.RandomRotation(45),
            transforms.ToTensor(),
            transforms.Normalize(*imagenet_stats),
        ]
    ),
    "valid": transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize(*imagenet_stats)]
    ),
}


def _denormalize(images, imagenet_stats):
    """De-normalize dataset using imagenet std and mean to show images."""
    mean = torch.tensor(imagenet_stats[0]).reshape(1, 3, 1, 1)
    std = torch.tensor(imagenet_stats[1]).reshape(1, 3, 1, 1)
    return images * std + mean

def show_data(dataloader, imagenet_stats=imagenet_stats, num_data=2):
    """Show `num_data` of images and labels from dataloader."""
    batch = next(iter(dataloader))  # batch of with images, batch of labels
    imgs, labels = batch[0][:num_data].to(device), batch[1][:num_data].tolist()  # get num_data of images, labels

    if plt.get_backend() == 'agg':
        print(f"Labels for {num_data} images: {labels}")
    else:
        _, axes = plt.subplots(1, num_data, figsize=(10, 6))
        for n in range(num_data):
            axes[n].set_title(labels[n])
            imgs[n] = _denormalize(imgs[n], imagenet_stats)
            axes[n].imshow(torch.clamp(imgs[n].cpu(), 0, 1).permute(1, 2, 0))
        plt.show()

def data_distribution(dataset, path: str) -> dict:
    """
    Returns a dictionary with the distribution of each class in the dataset.
    """
    class_counts = {cls: len(fnmatch.filter(os.listdir(f"{path}/{cls}"), "*.png")) 
                    for cls in dataset.class_to_idx.keys()}
    return class_counts

def plot_data_distribution(data_dist: dict, title: str = ""):
    """
    Plots or prints the distribution of data depending on the availability of a display.
    """
    if plt.get_backend() == 'agg':
        print(f"{title}: {data_dist}")
    else:
        classes, counts = list(data_dist.keys()), list(data_dist.values())
        sns.barplot(x=classes, y=counts).set_title(title)
        plt.show()

