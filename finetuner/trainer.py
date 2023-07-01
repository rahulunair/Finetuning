from config import device, torch
from tqdm import tqdm
import time

class Trainer:
    """Trainer class that takes care of training and validation passes."""

    def __init__(
        self,
        model,
        optimizer=torch.optim.SGD,
        epochs=10,
        lr=0.05,
        precision="fp32",
        device=device
    ):
        self.device = device
        self.model = model.to(self.device)
        self.loss_fn = torch.nn.CrossEntropyLoss()
        self.epochs = epochs
        self.lr = lr
        self.precision = precision
        if isinstance(optimizer, torch.optim.Adam):
            self.lr = 2e-3
        self.optimizer = optimizer(self.model.parameters(), lr=self.lr)

    def forward_pass(self, inputs, labels):
        """Perform forward pass of models with `inputs`,
        calculate loss and accuracy and return it.
        """
        outputs = self.model(inputs)
        loss = self.loss_fn(outputs, labels)
        preds = outputs.argmax(dim=1, keepdim=True)
        correct = preds.eq(labels.view_as(preds)).sum().item()
        acc = correct / labels.size(0)
        return loss, acc

    def train_one_batch(self, train_dataloader, max_epoch=100):
        """Train the model using just one batch for max_epoch.
        use this function to debug the training loop"""
        self.model.train()
        inputs, labels = next(iter(train_dataloader))
        for epoch in range(max_epoch):
            inputs, labels = inputs.to(self.device), labels.to(self.device)
            self.optimizer.zero_grad()
            loss, acc = self.forward_pass(inputs, labels)
            loss.backward()
            self.optimizer.step()
            print(f"[Epoch: {epoch+1}] loss: {loss.item()} acc: {acc}")

    def train(self, train_dataloader):
        """Training loop, return epoch loss and accuracy."""
        self.model.train()
        t_epoch_loss, t_epoch_acc = 0.0, 0.0
        for inputs, labels in tqdm(train_dataloader):
            inputs, labels = inputs.to(self.device), labels.to(self.device)
            self.optimizer.zero_grad()
            if self.precision == "bf16":
                with getattr(torch, f"{self.device.type}.amp.autocast")():
                    loss, acc = self.forward_pass(inputs, labels)
            else:
                loss, acc = self.forward_pass(inputs, labels)
            loss.backward()
            self.optimizer.step()
            t_epoch_loss += loss.item()
            t_epoch_acc += acc
        return (t_epoch_loss, t_epoch_acc)

    @torch.no_grad()
    def validate(self, valid_dataloader):
        """Validation loop, return validation epoch loss and accuracy."""
        self.model.eval()
        v_epoch_loss, v_epoch_acc = 0.0, 0.0
        for inputs, labels in tqdm(valid_dataloader):
            inputs, labels = inputs.to(self.device), labels.to(self.device)
            loss, acc = self.forward_pass(inputs, labels)
            v_epoch_loss += loss.item()
            v_epoch_acc += acc
        return (v_epoch_loss, v_epoch_acc)

    def fine_tune(self, train_dataloader, valid_dataloader):
        """Fine tune `self.model` using training set and measure perf using
        training and validation set.

        `train_dataloader`: training set
        `valid_dataloader`: validation set
        """
        print(f"Fine-tuning model for {self.epochs} epochs with lr = {self.lr}")
        for epoch in range(self.epochs):
            print(f"Epoch: [{epoch+1}]")
            t_epoch_loss, t_epoch_acc = self.train(train_dataloader)
            v_epoch_loss, v_epoch_acc = self.validate(valid_dataloader)
            print(
                f"Train Loss: {t_epoch_loss / len(train_dataloader):.4f}, "
                f"Train Acc: {t_epoch_acc / len(train_dataloader):.4f}, "
                f"Val Loss: {v_epoch_loss / len(valid_dataloader):.4f}, "
                f"Val Acc: {v_epoch_acc / len(valid_dataloader):.4f}"
            )
            return v_epoch_acc / len(valid_dataloader)

