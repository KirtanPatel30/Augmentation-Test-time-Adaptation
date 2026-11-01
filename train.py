import argparse
import csv
import json
import math
import os
import time

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.transforms import v2

from augmix import AugMixData, jsd
from model import ResNet18


def get_args():
    p = argparse.ArgumentParser()
    p.add_argument("--method", default="standard",
                   choices=["standard", "cutmix", "augmix", "augmix_nojsd"])
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--bs", type=int, default=128)
    p.add_argument("--lr", type=float, default=0.1)
    p.add_argument("--workers", type=int, default=2)
    p.add_argument("--data", default="data")
    p.add_argument("--out", default="runs")
    p.add_argument("--stop", type=int, default=0)
    p.add_argument("--fast", type=int, default=0)
    return p.parse_args()


def get_data(a):
    base = [transforms.RandomCrop(32, padding=4), transforms.RandomHorizontalFlip()]
    tt = transforms.ToTensor()
    if a.method.startswith("augmix"):
        tr = datasets.CIFAR10(a.data, train=True, download=True, transform=transforms.Compose(base))
        tr = AugMixData(tr, use_jsd=a.method == "augmix")
    else:
        tr = datasets.CIFAR10(a.data, train=True, download=True, transform=transforms.Compose(base + [tt]))
    te = datasets.CIFAR10(a.data, train=False, download=True, transform=tt)
    tr = DataLoader(tr, batch_size=a.bs, shuffle=True, num_workers=a.workers)
    te = DataLoader(te, batch_size=500, num_workers=a.workers)
    return tr, te


def test(m, te, dev, amp):
    m.eval()
    ok = 0
    with torch.no_grad():
        for x, y in te:
            x, y = x.to(dev), y.to(dev)
            with torch.autocast(dev, dtype=torch.float16, enabled=amp):
                ok += (m(x).argmax(1) == y).sum().item()
    return ok / len(te.dataset)


def train_epoch(m, tr, opt, sch, sc, a, dev, amp):
    m.train()
    cut = v2.CutMix(num_classes=10, alpha=1.0)
    tl, ok, n = 0, 0, 0
    for i, (x, y) in enumerate(tr):
        if a.fast and i == a.fast:
            break
        if a.method == "augmix":
            x = torch.cat(x)
        x, y = x.to(dev), y.to(dev)
        yh = y
        if a.method == "cutmix":
            x, y = cut(x, y)
            yh = y.argmax(1)
        with torch.autocast(dev, dtype=torch.float16, enabled=amp):
            out = m(x)
            if a.method == "augmix":
                o1, o2, o3 = out.split(len(y))
                loss = F.cross_entropy(o1, y) + 12 * jsd(o1.float(), o2.float(), o3.float())
                out = o1
            else:
                loss = F.cross_entropy(out, y)
        opt.zero_grad()
        sc.scale(loss).backward()
        sc.step(opt)
        sc.update()
        sch.step()
        tl += loss.item() * len(y)
        ok += (out.argmax(1) == yh).sum().item()
        n += len(y)
    return tl / n, ok / n


def save_log(path, rows):
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["epoch", "loss", "train_acc", "test_acc", "sec"])
        w.writerows(rows)


def main():
    a = get_args()
    torch.manual_seed(a.seed)
    np.random.seed(a.seed)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    amp = dev == "cuda"
    name = f"{a.method}_seed{a.seed}"
    folder = os.path.join(a.out, name)
    os.makedirs(folder, exist_ok=True)
    ck = os.path.join(folder, "last.pt")

    tr, te = get_data(a)
    steps = a.epochs * (a.fast or len(tr))
    m = ResNet18().to(dev)
    opt = torch.optim.SGD(m.parameters(), lr=a.lr, momentum=0.9, weight_decay=5e-4, nesterov=True)
    sch = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: 0.5 * (1 + math.cos(math.pi * s / steps)))
    sc = torch.amp.GradScaler(dev, enabled=amp)

    start, total, rows = 0, 0.0, []
    if os.path.exists(ck):
        s = torch.load(ck, map_location=dev)
        m.load_state_dict(s["model"])
        opt.load_state_dict(s["opt"])
        sch.load_state_dict(s["sch"])
        sc.load_state_dict(s["sc"])
        start, total, rows = s["epoch"], s["time"], s["rows"]
        print("resuming from epoch", start)

    end = a.epochs
    if a.stop:
        end = min(a.epochs, start + a.stop)
    print(f"training {name} on {dev}, epochs {start + 1} to {end}")

    for ep in range(start + 1, end + 1):
        t = time.time()
        loss, acc = train_epoch(m, tr, opt, sch, sc, a, dev, amp)
        sec = time.time() - t
        total += sec
        tacc = ""
        if ep % 5 == 0 or ep == a.epochs:
            tacc = round(test(m, te, dev, amp), 4)
        rows.append([ep, round(loss, 4), round(acc, 4), tacc, round(sec, 1)])
        save_log(os.path.join(folder, "log.csv"), rows)
        print(f"epoch {ep} | loss {loss:.4f} | train acc {acc:.4f} | test acc {tacc} | {sec:.0f}s")
        if ep % 5 == 0 or ep == end:
            s = {"model": m.state_dict(), "opt": opt.state_dict(), "sch": sch.state_dict(),
                 "sc": sc.state_dict(), "epoch": ep, "time": total, "rows": rows}
            torch.save(s, ck)

    if end < a.epochs:
        print("stopped early, run the same command again to continue")
        return

    acc = test(m, te, dev, amp)
    torch.save({"model": m.state_dict(), "method": a.method, "seed": a.seed, "acc": acc},
               os.path.join(folder, "final.pt"))
    info = {"run": name, "clean_acc": acc, "clean_err": 1 - acc, "hours": total / 3600,
            "sec_per_epoch": total / a.epochs, "device": dev}
    with open(os.path.join(folder, "summary.json"), "w") as f:
        json.dump(info, f, indent=2)
    print(f"done, clean acc {acc:.4f}, saved {folder}/final.pt")


if __name__ == "__main__":
    main()
