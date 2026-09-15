import torch
import torch.nn as nn

class TextClassifier(nn.Module):
    def __init__(self, in_features, num_classes):
        
        super().__init__()
        
        self.net = nn.Sequential(
            nn.Linear(in_features, 128)
            nn.ReLU(),
            nn.Linear(128, num_classes)
        )
    
    def forward(self, x):
        return self.net(x)