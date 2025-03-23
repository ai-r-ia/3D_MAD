import numpy as np
import torch.nn as nn
import torch.nn.functional as F


class PointNet(nn.Module):
    def __init__(self, num_classes=2, output_channels=2):
        super(PointNet, self).__init__()
        emb_dims = 512  # Increase to 1024
        self.conv1 = nn.Conv1d(3, 64, kernel_size=1, bias=False)
        self.conv2 = nn.Conv1d(64, 64, kernel_size=1, bias=False)
        self.conv3 = nn.Conv1d(64, 64, kernel_size=1, bias=False)
        self.conv4 = nn.Conv1d(64, 128, kernel_size=1, bias=False)
        self.conv5 = nn.Conv1d(128, emb_dims, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(64)
        self.bn3 = nn.BatchNorm1d(64)
        self.bn4 = nn.BatchNorm1d(128)
        self.bn5 = nn.BatchNorm1d(emb_dims)
        self.linear1 = nn.Linear(emb_dims, 1024, bias=False)  # Increase to 1024
        self.bn6 = nn.BatchNorm1d(1024)  # Match with increased feature size
        self.dp1 = nn.Dropout()
        self.linear2 = nn.Linear(1024, num_classes)  # Adjust output layer accordingly

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.relu(self.bn4(self.conv4(x)))
        x = F.relu(self.bn5(self.conv5(x)))
        x = F.adaptive_max_pool1d(x, 1)
        x = x.view(x.size(0), -1)
        features = F.relu(self.bn6(self.linear1(x)))  
        x = self.dp1(features) 
        x = self.linear2(x)
        return x, features
