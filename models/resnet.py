from typing import Tuple
import torch
from torch import nn, Tensor
from torchvision.models import resnet50, ResNet50_Weights


class resnet50Embeddings(nn.Module):
    def __init__(self, classes: int = 2, only_embeddings: bool = False) -> None:
        super(resnet50Embeddings, self).__init__()
        # Load the pre-trained resnet50 model
        self.model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
        self.only_embeddings = only_embeddings

        # Replace the classification head (fully connected layer) with a new one for our specific task
        self.model.fc = nn.Linear(self.model.fc.in_features, classes)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x: Tensor):
        # ResNet initial layers (conv1, bn1, relu, maxpool)
        x = self.model.conv1(x)
        x = self.model.bn1(x)
        x = self.model.relu(x)
        x = self.model.maxpool(x)

        # Forward pass through the 4 ResNet layers (consisting of residual blocks)
        x = self.model.layer1(x)
        x = self.model.layer2(x)
        x = self.model.layer3(x)
        x = self.model.layer4(x)

        # Global average pooling layer (before classification)
        x = self.model.avgpool(x)
        x = torch.flatten(x, 1)

        # Extract embeddings from the flattened features before the final FC layer
        embds = x

        # If we only want the embeddings, return them now
        if self.only_embeddings:
            return embds

        # Forward pass through the final classification layer (fully connected)
        x = self.model.fc(x)
        x = self.softmax(x)
        return x, embds  # Return both predictions and embeddings


class resnet50EmbeddingsAdaptive(nn.Module):
    def __init__(self, classes: int = 2, only_embeddings: bool = False) -> None:
        super(resnet50EmbeddingsAdaptive, self).__init__()
        self.model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
        self.only_embeddings = only_embeddings

        self.model.fc = nn.Linear(self.model.fc.in_features, classes)
        # self.reduce_dim = nn.Linear(768, 512)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x: Tensor, adapter_embds):
        # ResNet initial layers (conv1, bn1, relu, maxpool)
        x = self.model.conv1(x)
        x = self.model.bn1(x)
        x = self.model.relu(x)
        x = self.model.maxpool(x)

        # Forward pass through the 4 ResNet layers (consisting of residual blocks)
        x = self.model.layer1(x)
        x = self.model.layer2(x)
        x = self.model.layer3(x)
        x = self.model.layer4(x)

        # Global average pooling layer (before classification)
        x = self.model.avgpool(x)
        x = torch.flatten(x, 1)

        # Extract embeddings from the flattened features before the final FC layer
        embds = x

        # If we only want the embeddings, return them now
        if self.only_embeddings:
            return embds

        # Forward pass through the final classification layer (fully connected)
        x = self.model.fc(x)
        x = self.softmax(x)
        # adapter_embds = self.reduce_dim(adapter_embds)
        # print(embds.shape, adapter_embds.shape)
        adapter_embds = self.model.fc(adapter_embds)
        adapter_embds = self.softmax(adapter_embds)

        return x, adapter_embds, embds
