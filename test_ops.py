import numpy as np
import torch
from PIL import Image
from torchvision import transforms

import augmix

BAD = ["brightness", "contrast", "color", "sharpness"]


def test_no_bad_ops():
    names = [op.__name__ for op in augmix.OPS]
    for b in BAD:
        assert b not in names, b + " is a CIFAR-10-C corruption"


def test_ops_keep_size():
    im = Image.fromarray(np.random.randint(0, 256, (32, 32, 3), dtype=np.uint8))
    for op in augmix.OPS:
        assert op(im, 3).size == (32, 32)


def test_augmix_output():
    im = Image.fromarray(np.random.randint(0, 256, (32, 32, 3), dtype=np.uint8))
    x = augmix.augmix(im, transforms.ToTensor())
    assert x.shape == (3, 32, 32)
    assert x.min() >= 0 and x.max() <= 1


def test_jsd():
    a = torch.randn(8, 10)
    assert abs(augmix.jsd(a, a, a).item()) < 1e-5
    b = (torch.randn(8, 10) * 200).requires_grad_()
    augmix.jsd(b, torch.randn(8, 10), torch.randn(8, 10)).backward()
    assert torch.isfinite(b.grad).all()


if __name__ == "__main__":
    test_no_bad_ops()
    test_ops_keep_size()
    test_augmix_output()
    test_jsd()
    print("all tests passed")
