import torch
import torch.nn as nn
from torchvision import models
from options import Options


class ImageEmbeddings(nn.Module):
    def __init__(self, options: Options):
        super(ImageEmbeddings, self).__init__()
        self.conv1 = nn.Conv2d(1, 2, kernel_size=(3, 3), stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(2)
        self.relu1 = nn.ReLU()
        self.maxpool1 = nn.MaxPool2d(kernel_size=(2, 2), stride=2)

        self.conv2 = nn.Conv2d(2, 4, kernel_size=(3, 3), stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(4)
        self.relu2 = nn.ReLU()
        self.maxpool2 = nn.MaxPool2d(kernel_size=(4, 4), stride=4)

        self.lin1 = nn.Linear(900, options.embedding_size)

    def forward(self, x):
        x = x.view(-1, 1, x.shape[-1], x.shape[-1])

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)
        x = self.maxpool1(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu2(x)
        x = self.maxpool2(x)

        x = x.flatten(start_dim=1)
        x = self.lin1(x)
        return x


class OldImageEmbeddings(nn.Module):
    def __init__(self, options: Options):
        super(OldImageEmbeddings, self).__init__()
        self.resnet18 = models.resnet18(weights="ResNet18_Weights.DEFAULT")
        self.resnet18.conv1 = torch.nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        self.resnet18.fc = nn.Linear(512, options.embedding_size)

    def forward(self, x: torch.Tensor):
        # d is embedding_size
        x = x.view(-1, 1, x.shape[-1], x.shape[-1])
        d: torch.Tensor = self.resnet18(x)
        return d
