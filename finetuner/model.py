from config import torch
import torch.nn as nn
import torchvision.models as models

class FireFinder(nn.Module):
    """
    A model to classify aerial images that could potentially Fire from satellite images
    We are using a pretrained resnet backbone model
    and images given to model are classified into one of 3 classes.
    0 - no Fire
    1 - Fire

    We currently use the resnet18 model as a backbone
    """

    def __init__(self, backbone=18, simple=True, dropout= .4):
        super(FireFinder, self).__init__()
        backbones = {
            18: models.resnet18,
            34: models.resnet34,
            50: models.resnet50,
            101: models.resnet101,
        }
        #self.network = backbones[backbone](pretrained=True)
        self.network = backbones[backbone](weights=True)
        #self.network = backbones[backbone](weights='ResNet18_Weights.DEFAULT')
        for m, p in zip(self.network.modules(), self.network.parameters()):
            if isinstance(m, nn.BatchNorm2d):
                p.requires_grad = False
        if simple:
            self.network.fc = nn.Linear(self.network.fc.in_features, 3)
            nn.init.xavier_uniform_(self.network.fc.weight)
        else:
            fc = nn.Sequential(
                nn.Linear(self.network.fc.in_features, 256),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(
                    256, 2
                ),  # here we are using 3 for `out_features` as the image given
                # to the model can be one of 3 classes (0 - no bone, 1 - bone possible, 2 - bone likely)
            )
            for layer in fc.modules():
                if isinstance(layer, nn.Linear):
                    nn.init.xavier_uniform_(layer.weight)
            self.network.fc = fc
            
            
    def forward(self, x_batch):
        return self.network(x_batch)
