import argparse
import csv
import os
import shutil
import subprocess
import sys

import torch

METHODS = ["standard", "cutmix", "augmix_nojsd", "augmix"]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--epochs", type=int, default=2)
    p.add_argument("--workers", type=int, default=2)
    p.add_argument("--fast", type=int, default=0)
    a = p.parse_args()

    dev = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"
    out = os.path.join("runs", "timing")
    rows = []
    for name in METHODS:
        shutil.rmtree(os.path.join(out, f"{name}_seed1"), ignore_errors=True)
        cmd = [sys.executable, "train.py", "--method", name, "--stop", str(a.epochs),
               "--workers", str(a.workers), "--fast", str(a.fast), "--out", out]
        subprocess.run(cmd, check=True)
        with open(os.path.join(out, f"{name}_seed1", "log.csv")) as f:
            secs = [float(r["sec"]) for r in csv.DictReader(f)]
        sec = secs[-1]
        rows.append([name, dev, sec, round(sec * 100 / 3600, 2)])

    os.makedirs("results", exist_ok=True)
    with open(os.path.join("results", "epoch_timing.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["method", "device", "sec_per_epoch", "hours_for_100_epochs"])
        w.writerows(rows)
    for r in rows:
        print(f"{r[0]:14s} {r[2]:8.1f} s/epoch   about {r[3]} h for 100 epochs")


if __name__ == "__main__":
    main()
