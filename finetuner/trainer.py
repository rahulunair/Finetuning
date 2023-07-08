import torch
import intel_extension_for_pytorch as ipex
import os

from config import device, torch
from tqdm import tqdm
import time
import wandb
import numpy as np
from tabulate import tabulate


class Trainer:
    """Trainer class that takes care of training and validation passes."""

    def __init__(
        self,
        model,
        optimizer=torch.optim.SGD,
        epochs=10,
        lr=0.05,
        precision="fp32",
        device=device,
        use_wandb=False,
    ):
        self.use_wandb = use_wandb
        self.device = device
        self.model = model.to(self.device)
        self.loss_fn = torch.nn.CrossEntropyLoss()
        self.epochs = epochs
        self.lr = lr
        self.precision = precision
        self.optimizer = optimizer(self.model.parameters(), lr=self.lr)

        # Initialize the learning rate scheduler
        self.scheduler = torch.optim.lr_scheduler.StepLR(
            self.optimizer, step_size=30, gamma=0.1
        )

        # Check if the optimizer is an instance of Adam
        if isinstance(optimizer, torch.optim.Adam):
            self.lr = lr

    def update_learning_rate(self, lr):
        """Update learning rate of the optimizer"""
        for param_group in self.optimizer.param_groups:
            param_group["lr"] = lr

    def forward_pass(self, inputs, labels):
        """Perform forward pass of models with `inputs`,
        calculate loss and accuracy and return it.
        """
        outputs = self.model(inputs)
        loss = self.loss_fn(outputs, labels)
        preds = outputs.argmax(dim=1, keepdim=True)
        correct = preds.eq(labels.view_as(preds)).sum().item()
        total = labels.numel()
        return loss, correct, total

    def train_one_batch(self, train_dataloader, max_epoch=100):
        """Train the model using just one batch for max_epoch.
        use this function to debug the training loop"""
        self.model.train()
        inputs, labels = next(iter(train_dataloader))
        for epoch in range(max_epoch):
            inputs, labels = inputs.to(self.device), labels.to(self.device)
            self.optimizer.zero_grad()
            loss, correct, total = self.forward_pass(inputs, labels)
            loss.backward()
            self.optimizer.step()
            acc = correct / total
            print(f"[Epoch: {epoch+1}] loss: {loss.item()} acc: {acc}")

    def _to_ipex(self, dtype=torch.float32):
        """convert model memory format to channels_last to IPEX format."""
        self.model.train()
        self.model = self.model.to(memory_format=torch.channels_last)
        self.model, self.optimizer = ipex.optimize(
            self.model, optimizer=self.optimizer, dtype=torch.float32
        )

    def train(self, train_dataloader):
        """Training loop, return epoch loss and accuracy."""
        self.model.train()
        total_loss, total_correct, total_samples = 0.0, 0, 0
        for inputs, labels in tqdm(train_dataloader):
            inputs, labels = inputs.to(self.device), labels.to(self.device)
            self.optimizer.zero_grad()
            if self.precision == "bf16":
                with getattr(torch, f"{self.device.type}.amp.autocast")():
                    loss, correct, batch_size = self.forward_pass(inputs, labels)
            else:
                loss, correct, batch_size = self.forward_pass(inputs, labels)
            loss.backward()
            self.optimizer.step()
            total_loss += loss.item()
            total_correct += correct
            total_samples += batch_size
            acc = total_correct / total_samples
            if self.use_wandb:
                wandb.log(
                    {
                        "Training Loss": total_loss / len(train_dataloader),
                        "Training Acc": acc,
                    }
                )
        self.scheduler.step()
        return total_loss / len(train_dataloader), acc

    @torch.no_grad()
    def validate(self, valid_dataloader):
        """Validation loop, return validation epoch loss and accuracy."""
        self.model.eval()
        total_loss, total_correct, total_samples = 0.0, 0, 0
        for inputs, labels in tqdm(valid_dataloader):
            inputs, labels = inputs.to(self.device), labels.to(self.device)
            loss, correct, batch_size = self.forward_pass(inputs, labels)
            total_loss += loss.item()
            total_correct += correct
            total_samples += batch_size
            acc = total_correct / total_samples
            if self.use_wandb:
                wandb.log(
                    {
                        "Validation Loss": total_loss / len(valid_dataloader),
                        "Validation Acc": acc,
                    }
                )
        return total_loss / len(valid_dataloader), acc

    def lr_range_test(self, train_dataloader, start_lr=1e-7, end_lr=1e-2, num_iter=100):
        infinite_train_dataloader = train_dataloader
        orig_model_state_dict = self.model.state_dict()
        orig_opt_state_dict = self.optimizer.state_dict()
        lrs = []
        losses = []
        for param_group in self.optimizer.param_groups:
            param_group["lr"] = start_lr
        lr_lambda = lambda x: (end_lr / start_lr) ** (x / num_iter)
        lr_scheduler = torch.optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda)
        self.model.train()
        for i, (inputs, labels) in enumerate(tqdm(infinite_train_dataloader)):
            if i >= num_iter:
                break
            inputs, labels = inputs.to(self.device), labels.to(self.device)
            self.optimizer.zero_grad()
            if self.precision == "bf16":
                with getattr(torch, f"{self.device.type}.amp.autocast")():
                    loss, correct, total = self.forward_pass(inputs, labels)
            else:
                loss, correct, total = self.forward_pass(inputs, labels)
            loss.backward()
            self.optimizer.step()
            #lr_scheduler.step()

            # Log the learning rate and loss
            lrs.append(self.optimizer.param_groups[0]["lr"])
            losses.append(loss.item())
            if self.use_wandb:
                wandb.log({"lr": lrs[-1], "loss": losses[-1]})

        # Restore the original state of the model and optimizer
        self.model.load_state_dict(orig_model_state_dict)
        self.optimizer.load_state_dict(orig_opt_state_dict)
        min_loss_idx = np.argmin(losses)
        self.lr = lrs[min_loss_idx]
        table = list(zip(lrs, losses))
        print(tabulate(table, headers=["Learning Rate", "Loss"], tablefmt="pretty"))

    def fine_tune(self, train_dataloader, valid_dataloader):
        if self.use_wandb:
            wandb.init(project="cnn-training", name="cnn-model")

        for epoch in range(self.epochs):
            t_epoch_start = time.time()

            t_epoch_loss, t_epoch_acc = self.train(train_dataloader)
            v_epoch_loss, v_epoch_acc = self.validate(valid_dataloader)

            t_epoch_end = time.time()

            print(
                f"\n📅 Epoch {epoch+1}/{self.epochs}:\n"
                f"\t🏋️‍♂️ Traiing step:\n"
                f"\t - 🎯 Loss: {t_epoch_loss:.4f}"
                f", 📈 Accuracy: {t_epoch_acc:.4f}\n"
                f"\t🧪 Validation step:\n"
                f"\t - 🎯 Loss: {v_epoch_loss:.4f}"
                f", 📈 Accuracy: {v_epoch_acc:.4f}\n"
                f"⏱️ Time: {t_epoch_end - t_epoch_start:.4f} sec\n"
            )
            if self.use_wandb:
                wandb.log(
                    {
                        "Train Loss": t_epoch_loss,
                        "Train Acc": t_epoch_acc,
                        "Valid Loss": v_epoch_loss,
                        "Valid Acc": v_epoch_acc,
                        "Time": t_epoch_end - t_epoch_start,
                    }
                )

        if self.use_wandb:
            wandb.finish()
