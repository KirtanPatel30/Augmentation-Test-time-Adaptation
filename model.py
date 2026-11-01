import torch
import torch.nn as nn
import torch.nn.functional as F

MEAN = (0.4914, 0.4822, 0.4465)
STD = (0.2471, 0.2435, 0.2616)


class Block(nn.Module):
    def __init__(self, a, b, s):
        super().__init__()
        self.c1 = nn.Conv2d(a, b, 3, s, 1, bias=False)
        self.b1 = nn.BatchNorm2d(b)
        self.c2 = nn.Conv2d(b, b, 3, 1, 1, bias=False)
        self.b2 = nn.BatchNorm2d(b)
        self.skip = nn.Sequential()
        if s != 1 or a != b:
            self.skip = nn.Sequential(nn.Conv2d(a, b, 1, s, bias=False), nn.BatchNorm2d(b))

    def forward(self, x):
        out = F.relu(self.b1(self.c1(x)))
        out = self.b2(self.c2(out))
        return F.relu(out + self.skip(x))


class ResNet18(nn.Module):
    def __init__(self, n=10):
        super().__init__()
        self.register_buffer("mean", torch.tensor(MEAN).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor(STD).view(1, 3, 1, 1))
        self.c1 = nn.Conv2d(3, 64, 3, 1, 1, bias=False)
        self.b1 = nn.BatchNorm2d(64)
        self.l1 = nn.Sequential(Block(64, 64, 1), Block(64, 64, 1))
        self.l2 = nn.Sequential(Block(64, 128, 2), Block(128, 128, 1))
        self.l3 = nn.Sequential(Block(128, 256, 2), Block(256, 256, 1))
        self.l4 = nn.Sequential(Block(256, 512, 2), Block(512, 512, 1))
        self.fc = nn.Linear(512, n)

    def forward(self, x):
        x = (x - self.mean) / self.std
        x = F.relu(self.b1(self.c1(x)))
        x = self.l4(self.l3(self.l2(self.l1(x))))
        x = F.adaptive_avg_pool2d(x, 1).flatten(1)
        return self.fc(x)


def load(path):
    m = ResNet18()
    m.load_state_dict(torch.load(path, map_location="cpu")["model"])
    return m
