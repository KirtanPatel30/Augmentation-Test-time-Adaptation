"""CIFAR-10 / CIFAR-10-C loading.

CIFAR-10-C (Zenodo record 2535967) has one ``<corruption>.npy`` file per
corruption with shape (50000, 32, 32, 3), uint8. Rows are ordered by severity:
rows [(s-1)*10000, s*10000) are severity s. ``labels.npy`` holds the labels in
the same order. All images are returned as float tensors in [0, 1]; each model
does its own normalization.
"""
import os

import numpy as np
import torch

CORRUPTIONS = [
    # noise
    "gaussian_noise", "shot_noise", "impulse_noise",
    # blur
    "defocus_blur", "glass_blur", "motion_blur", "zoom_blur",
    # weather
    "snow", "frost", "fog", "brightness",
    # digital
    "contrast", "elastic_transform", "pixelate", "jpeg_compression",
]

# Only used to choose the Tent settings, never to report results.
EXTRA_CORRUPTIONS = ["speckle_noise", "gaussian_blur", "spatter", "saturate"]

SEVERITIES = [1, 2, 3, 4, 5]
N_PER_SEVERITY = 10000


def to_tensor(x_uint8):
    """(N, 32, 32, 3) uint8 -> (N, 3, 32, 32) float in [0, 1]."""
    x = torch.from_numpy(np.ascontiguousarray(x_uint8))
    # contiguous(): a permuted tensor breaks Tent's backward pass on Apple MPS.
    return x.permute(0, 3, 1, 2).contiguous().float().div_(255.0)


def load_cifar10c(corruption, severity, data_dir="data/CIFAR-10-C", n_examples=N_PER_SEVERITY):
    """Return (x uint8 array (n, 32, 32, 3), y int64 array (n,)) for one condition."""
    if severity not in SEVERITIES:
        raise ValueError(f"severity must be in 1..5, got {severity}")
    path = os.path.join(data_dir, f"{corruption}.npy")
    if not os.path.exists(path):
        raise FileNotFoundError(f"{path} not found. Run scripts/download_cifar10c.sh first.")
    start = (severity - 1) * N_PER_SEVERITY
    x = np.load(path, mmap_mode="r")[start:start + n_examples]
    y = np.load(os.path.join(data_dir, "labels.npy"))[start:start + n_examples]
    return np.asarray(x), y.astype(np.int64)


CIFAR10_HF_REPO = "uoft-cs/cifar10"  # official CIFAR-10 on Hugging Face, same test order
CIFAR10_HF_TEST = "plain_text/test-00000-of-00001.parquet"


def _cifar10_test_from_hf(cache_path):
    """Download the 24 MB test split from Hugging Face and cache it as .npz."""
    import io

    import pandas as pd
    from huggingface_hub import hf_hub_download
    from PIL import Image

    parquet = hf_hub_download(CIFAR10_HF_REPO, CIFAR10_HF_TEST, repo_type="dataset")
    df = pd.read_parquet(parquet)
    x = np.stack([np.asarray(Image.open(io.BytesIO(img["bytes"])).convert("RGB")) for img in df["img"]])
    y = df["label"].to_numpy(dtype=np.int64)
    np.savez(cache_path, x=x, y=y)
    return x, y


def load_cifar10_clean(data_dir="data", n_examples=N_PER_SEVERITY):
    """Clean CIFAR-10 test set, same format as load_cifar10c.

    Uses a local torchvision copy if there is one. Otherwise it gets the test split
    from Hugging Face, because the official Toronto server is often very slow
    (about 60 kB/s on Colab), and falls back to torchvision's download.
    """
    from torchvision.datasets import CIFAR10

    if os.path.isdir(os.path.join(data_dir, "cifar-10-batches-py")):
        ds = CIFAR10(root=data_dir, train=False, download=False)
        return ds.data[:n_examples], np.asarray(ds.targets[:n_examples], dtype=np.int64)
    cache = os.path.join(data_dir, "cifar10_test.npz")
    if os.path.exists(cache):
        z = np.load(cache)
        return z["x"][:n_examples], z["y"][:n_examples]
    try:
        x, y = _cifar10_test_from_hf(cache)
        return x[:n_examples], y[:n_examples]
    except Exception as e:  # no internet to HF, missing pandas, ...
        print(f"Hugging Face download of CIFAR-10 failed ({e}); using torchvision")
    ds = CIFAR10(root=data_dir, train=False, download=True)
    return ds.data[:n_examples], np.asarray(ds.targets[:n_examples], dtype=np.int64)


def load_condition(corruption, severity, data_dir="data", n_examples=N_PER_SEVERITY):
    """Load clean (corruption == "clean") or one CIFAR-10-C condition."""
    if corruption == "clean":
        return load_cifar10_clean(data_dir, n_examples)
    return load_cifar10c(corruption, severity, os.path.join(data_dir, "CIFAR-10-C"), n_examples)
