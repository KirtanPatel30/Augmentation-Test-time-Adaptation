import torch
import torch.nn as nn
import torch.nn.functional as F

MEAN = (0.4914, 0.4822, 0.4465)
STD = (0.2471, 0.2435, 0.2616)


class Block(nn.Module):
    def __init__(self, a, b, s):
        super().__init__()
        self.conv1 = nn.Conv2d(a, b, 3, s, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(b)
        self.conv2 = nn.Conv2d(b, b, 3, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(b)
        self.shortcut = nn.Sequential()
        if s != 1 or a != b:
            self.shortcut = nn.Sequential(nn.Conv2d(a, b, 1, s, bias=False), nn.BatchNorm2d(b))

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return F.relu(out + self.shortcut(x))


class ResNet18(nn.Module):
    def __init__(self, n=10):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 64, 3, 1, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = nn.Sequential(Block(64, 64, 1), Block(64, 64, 1))
        self.layer2 = nn.Sequential(Block(64, 128, 2), Block(128, 128, 1))
        self.layer3 = nn.Sequential(Block(128, 256, 2), Block(256, 256, 1))
        self.layer4 = nn.Sequential(Block(256, 512, 2), Block(512, 512, 1))
        self.linear = nn.Linear(512, n)

    def forward(self, x):
        mean = torch.tensor(MEAN, device=x.device).view(1, 3, 1, 1)
        std = torch.tensor(STD, device=x.device).view(1, 3, 1, 1)
        x = (x - mean) / std
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.layer4(self.layer3(self.layer2(self.layer1(x))))
        x = F.adaptive_avg_pool2d(x, 1).flatten(1)
        return self.linear(x)


def load(path):
    m = ResNet18()
    m.load_state_dict(torch.load(path, map_location="cpu")["model"])
    return m
