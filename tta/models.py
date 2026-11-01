"""Models used in Stage 2.

All models take images in [0, 1] and normalize inside the network, so the
same evaluation code works for our ResNet-18 checkpoints and RobustBench models.

The ResNet-18 here is the usual CIFAR version (3x3 first conv, no max-pool).
Stage 1 checkpoints must use the same module names; see ``load_resnet18``.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD = (0.2471, 0.2435, 0.2616)

# Public models for the E1 pipeline checks (Table 2 of the proposal).
ROBUSTBENCH_MODELS = {
    "rb_standard_wrn28": "Standard",
    "rb_augmix_wrn40": "Hendrycks2020AugMix_WRN",
}


class Normalize(nn.Module):
    def __init__(self, mean=CIFAR10_MEAN, std=CIFAR10_STD):
        super().__init__()
        self.register_buffer("mean", torch.tensor(mean).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor(std).view(1, 3, 1, 1))

    def forward(self, x):
        return (x - self.mean) / self.std


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, 3, stride, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, 3, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, planes, 1, stride, bias=False),
                nn.BatchNorm2d(planes),
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return F.relu(out + self.shortcut(x))


class ResNet18(nn.Module):
    """CIFAR ResNet-18, about 11.2 M parameters. Input: [0, 1] images."""

    def __init__(self, num_classes=10, normalize=True):
        super().__init__()
        self.normalize = Normalize() if normalize else nn.Identity()
        self.conv1 = nn.Conv2d(3, 64, 3, 1, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.in_planes = 64
        self.layer1 = self._make_layer(64, 2, 1)
        self.layer2 = self._make_layer(128, 2, 2)
        self.layer3 = self._make_layer(256, 2, 2)
        self.layer4 = self._make_layer(512, 2, 2)
        self.linear = nn.Linear(512, num_classes)

    def _make_layer(self, planes, n_blocks, stride):
        layers = []
        for s in [stride] + [1] * (n_blocks - 1):
            layers.append(BasicBlock(self.in_planes, planes, s))
            self.in_planes = planes
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(self.normalize(x))))
        out = self.layer4(self.layer3(self.layer2(self.layer1(out))))
        out = F.adaptive_avg_pool2d(out, 1).flatten(1)
        return self.linear(out)


def load_resnet18(checkpoint_path, device="cpu"):
    """Load a Stage 1 checkpoint.

    Accepts a raw state_dict or a dict with a "model" / "state_dict" key, and
    strips the "module." prefix that DataParallel / torch.compile add.
    """
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if isinstance(ckpt, dict):
        for key in ("model", "state_dict", "model_state_dict"):
            if key in ckpt:
                ckpt = ckpt[key]
                break
    state = {}
    for k, v in ckpt.items():
        for prefix in ("module.", "_orig_mod."):
            if k.startswith(prefix):
                k = k[len(prefix):]
        state[k] = v
    model = ResNet18()
    # Old checkpoints without the Normalize buffers still load.
    missing, unexpected = model.load_state_dict(state, strict=False)
    missing = [k for k in missing if not k.startswith("normalize.")]
    if missing or unexpected:
        raise RuntimeError(f"checkpoint mismatch: missing={missing}, unexpected={unexpected}")
    return model.to(device)


def load_model(name, device="cpu", model_dir="models"):
    """``name`` is a RobustBench key from ROBUSTBENCH_MODELS or a checkpoint path."""
    if name in ROBUSTBENCH_MODELS:
        from robustbench.utils import load_model as rb_load

        model = rb_load(ROBUSTBENCH_MODELS[name], model_dir=model_dir,
                        dataset="cifar10", threat_model="corruptions")
        return model.to(device)
    return load_resnet18(name, device)
