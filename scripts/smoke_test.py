"""End-to-end check of eval.py + analyze.py without downloading anything.

Makes a fake CIFAR-10-C (random images, real file layout) and a random ResNet-18
checkpoint, then runs every method and protocol on a few hundred images.
The numbers are meaningless (about 90% error); this only checks the plumbing.

    python scripts/smoke_test.py
"""
import os
import shutil
import subprocess
import sys

import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from tta.models import ResNet18  # noqa: E402

OUT = os.path.join(ROOT, "smoke")
N = 400
CORRUPTIONS = ["gaussian_noise", "fog", "contrast"]


def make_fake_data():
    data = os.path.join(OUT, "data", "CIFAR-10-C")
    os.makedirs(data, exist_ok=True)
    rng = np.random.default_rng(0)
    labels = np.lib.format.open_memmap(os.path.join(data, "labels.npy"), "w+", np.uint8, (50000,))
    for s in range(5):
        labels[s * 10000:s * 10000 + N] = rng.integers(0, 10, N)
    labels.flush()
    for c in CORRUPTIONS:
        x = np.lib.format.open_memmap(os.path.join(data, f"{c}.npy"), "w+", np.uint8, (50000, 32, 32, 3))
        for s in range(5):
            x[s * 10000:s * 10000 + N] = rng.integers(0, 256, (N, 32, 32, 3), dtype=np.uint8)
        x.flush()
    ckpt = os.path.join(OUT, "random_resnet18.pt")
    torch.save({"model": ResNet18().state_dict(), "epoch": 0}, ckpt)
    return ckpt


def run(*args):
    cmd = [sys.executable, os.path.join(ROOT, "eval.py"), *args,
           "--data-dir", os.path.join(OUT, "data"), "--out-dir", os.path.join(OUT, "results"),
           "--n-examples", str(N), "--corruptions", ",".join(CORRUPTIONS)]
    print("\n$", " ".join(os.path.relpath(a, ROOT) if a.startswith(ROOT) else a for a in cmd[1:]), flush=True)
    subprocess.run(cmd, check=True, cwd=ROOT)


def main():
    shutil.rmtree(OUT, ignore_errors=True)
    ckpt = make_fake_data()
    for method in ["none", "bn", "tent"]:
        run("--model", ckpt, "--method", method, "--protocol", "episodic")
    for method in ["bn", "tent"]:
        run("--model", ckpt, "--method", method, "--protocol", "continual")
        run("--model", ckpt, "--method", method, "--protocol", "small_batch")
    print("\n$ analyze.py", flush=True)
    subprocess.run([sys.executable, os.path.join(ROOT, "analyze.py"), "--results",
                    os.path.join(OUT, "results"), "--n-boot", "200"], check=True, cwd=ROOT)
    print(f"\nsmoke test passed; outputs in {os.path.relpath(OUT, ROOT)}/results")


if __name__ == "__main__":
    main()
