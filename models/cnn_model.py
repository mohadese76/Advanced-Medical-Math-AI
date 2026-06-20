import torch
import torch.nn as nn
import torchvision.models as models

class PneumoniaCNN(nn.Module):
    def __init__(self, num_classes=2):
        super(PneumoniaCNN, self).__init__()
        self.network = models.resnet18(weights='DEFAULT')
        
       # Extract the number of input features of the last layer
        num_ftrs = self.network.fc.in_features
        
        # Last layer change: Add Dropout to calculate Uncertainty

        self.network.fc = nn.Sequential(
            nn.Dropout(p=0.2), 
            nn.Linear(num_ftrs, num_classes)
        )

    def forward(self, xb):
        return self.network(xb)

print(" Model architecture (ResNet18 + Dropout for Uncertainty) initialized.")