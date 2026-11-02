# AugMix ops and JSD loss based on google-research/augmix (Apache-2.0).
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageOps
from torchvision import transforms

S = 32


def lv(k, top):
    return np.random.uniform(0.1, k) * top / 10


def flip(v):
    if np.random.rand() > 0.5:
        return -v
    return v


def move(im, t):
    return im.transform((S, S), Image.AFFINE, t, resample=Image.BILINEAR)


def autocontrast(im, k):
    return ImageOps.autocontrast(im)


def equalize(im, k):
    return ImageOps.equalize(im)


def posterize(im, k):
    return ImageOps.posterize(im, 4 - int(lv(k, 4)))


def rotate(im, k):
    return im.rotate(flip(int(lv(k, 30))), resample=Image.BILINEAR)


def solarize(im, k):
    return ImageOps.solarize(im, 256 - int(lv(k, 256)))


def shear_x(im, k):
    return move(im, (1, flip(lv(k, 0.3)), 0, 0, 1, 0))


def shear_y(im, k):
    return move(im, (1, 0, 0, flip(lv(k, 0.3)), 1, 0))


def translate_x(im, k):
    return move(im, (1, 0, flip(int(lv(k, S / 3))), 0, 1, 0))


def translate_y(im, k):
    return move(im, (1, 0, 0, 0, 1, flip(int(lv(k, S / 3)))))


OPS = [autocontrast, equalize, posterize, rotate, solarize,
       shear_x, shear_y, translate_x, translate_y]


def augmix(im, tt, k=3, sev=3):
    w = np.random.dirichlet([1.0] * k)
    m = float(np.random.beta(1.0, 1.0))
    mix = torch.zeros_like(tt(im))
    for i in range(k):
        a = im.copy()
        for _ in range(np.random.randint(1, 4)):
            op = np.random.choice(OPS)
            a = op(a, sev)
        mix += float(w[i]) * tt(a)
    return (1 - m) * tt(im) + m * mix


class AugMixData(torch.utils.data.Dataset):
    def __init__(self, ds, use_jsd):
        self.ds = ds
        self.use_jsd = use_jsd
        self.tt = transforms.ToTensor()

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, i):
        im, y = self.ds[i]
        if self.use_jsd:
            return (self.tt(im), augmix(im, self.tt), augmix(im, self.tt)), y
        return augmix(im, self.tt), y


def kl(lp, lm):
    return (lp.exp() * (lp - lm)).sum(1).mean()


def jsd(a, b, c):
    la = F.log_softmax(a, 1)
    lb = F.log_softmax(b, 1)
    lc = F.log_softmax(c, 1)
    lm = torch.clamp((la.exp() + lb.exp() + lc.exp()) / 3, 1e-7, 1).log()
    return (kl(la, lm) + kl(lb, lm) + kl(lc, lm)) / 3
