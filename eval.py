"""Stage 2 evaluation: run one model x one method x one protocol and save every prediction.

Examples:
    # E1 check: public Standard WRN-28-10 with Tent, severity 5 only
    python eval.py --model rb_standard_wrn28 --method tent --protocol episodic --severities 5

    # E3: our ResNet-18 checkpoint, all 75 conditions
    python eval.py --model checkpoints/augmix_jsd_s1.pt --method bn --protocol episodic

    # E4: difficult streams
    python eval.py --model checkpoints/standard_s1.pt --method tent --protocol continual
    python eval.py --model checkpoints/standard_s1.pt --method tent --protocol small_batch

    # Choose the Tent learning rate on the 4 extra corruptions only
    python eval.py --model rb_standard_wrn28 --method tent --corruptions extra --severities 5 --lr 1e-4

Each run writes results/preds/<run>.npz (logits for every image, in stream
order) and appends one row per condition to results/summary.csv. analyze.py
computes every metric from the .npz files, so it needs no GPU.
"""
import argparse
import csv
import json
import os
import time

import numpy as np
import torch

from tta.data import CORRUPTIONS, EXTRA_CORRUPTIONS, load_condition
from tta.methods import build_method
from tta.metrics import ece, error
from tta.models import load_model
from tta.streams import PROTOCOLS, REFERENCE_BATCH_SIZE, build_conditions, run_stream


def pick_device(name):
    if name != "auto":
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def model_tag(name):
    return os.path.splitext(os.path.basename(name))[0]


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", required=True, help="RobustBench key (rb_standard_wrn28, rb_augmix_wrn40) or checkpoint path")
    p.add_argument("--method", required=True, choices=["none", "bn", "tent"])
    p.add_argument("--protocol", default="episodic", choices=list(PROTOCOLS))
    p.add_argument("--corruptions", default="test",
                   help="'test' (15), 'extra' (4, for tuning), or a comma-separated list")
    p.add_argument("--severities", default=None, help="comma-separated, default from the protocol")
    p.add_argument("--clean", action="store_true", help="also evaluate the clean test set first")
    p.add_argument("--batch-size", type=int, default=None, help="default from the protocol")
    p.add_argument("--lr", type=float, default=1e-3, help="Tent learning rate at batch 200")
    p.add_argument("--no-lr-scaling", action="store_true",
                   help="do not scale the Tent lr by batch_size/200")
    p.add_argument("--steps", type=int, default=1)
    p.add_argument("--shuffle-seed", type=int, default=None, help="shuffle image order inside each condition")
    p.add_argument("--reset-every", type=int, default=None, help="reset every N batches (RDumb, E6)")
    p.add_argument("--n-examples", type=int, default=10000, help="images per condition (lower for quick trials)")
    p.add_argument("--data-dir", default="data")
    p.add_argument("--model-dir", default="models", help="where RobustBench caches its models")
    p.add_argument("--out-dir", default="results")
    p.add_argument("--tag", default="", help="extra text for the run name")
    p.add_argument("--device", default="auto")
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    torch.manual_seed(args.seed)
    device = pick_device(args.device)
    proto = PROTOCOLS[args.protocol]

    if args.corruptions == "test":
        corruptions = CORRUPTIONS
    elif args.corruptions == "extra":
        corruptions = EXTRA_CORRUPTIONS
    else:
        corruptions = args.corruptions.split(",")
    severities = [int(s) for s in args.severities.split(",")] if args.severities else None
    conditions = build_conditions(args.protocol, corruptions, severities)
    if args.clean:
        conditions = [("clean", 0)] + conditions

    batch_size = args.batch_size or proto["batch_size"]
    lr = args.lr
    if args.method == "tent" and not args.no_lr_scaling:
        lr = args.lr * batch_size / REFERENCE_BATCH_SIZE

    run = "_".join(x for x in [model_tag(args.model), args.method, args.protocol, f"bs{batch_size}",
                               f"lr{lr:g}" if args.method == "tent" else "",
                               f"shuf{args.shuffle_seed}" if args.shuffle_seed is not None else "",
                               f"reset{args.reset_every}" if args.reset_every else "",
                               f"n{args.n_examples}" if args.n_examples != 10000 else "",
                               args.tag] if x)
    print(f"run {run} on {device}: {len(conditions)} conditions, batch {batch_size}")

    model = load_model(args.model, device, args.model_dir)
    method = build_method(args.method, model, lr=lr, steps=args.steps).to(device)

    def load_fn(c, s):
        return load_condition(c, s, args.data_dir, args.n_examples)

    t0 = time.time()
    results = run_stream(method, conditions, load_fn, batch_size, proto["reset_each"], device,
                         shuffle_seed=args.shuffle_seed, reset_every=args.reset_every)
    total = time.time() - t0

    meta = dict(run=run, model=args.model, method=args.method, protocol=args.protocol,
                batch_size=batch_size, lr=lr if args.method == "tent" else None, steps=args.steps,
                shuffle_seed=args.shuffle_seed, reset_every=args.reset_every,
                reset_each=proto["reset_each"], n_examples=args.n_examples, device=str(device),
                seconds=total, conditions=[f"{c}:{s}" for c, s in conditions])

    pred_dir = os.path.join(args.out_dir, "preds")
    os.makedirs(pred_dir, exist_ok=True)
    np.savez_compressed(
        os.path.join(pred_dir, f"{run}.npz"),
        meta=json.dumps(meta),
        logits=np.concatenate([r["logits"] for r in results]).astype(np.float16),
        labels=np.concatenate([r["labels"] for r in results]).astype(np.int16),
        index=np.concatenate([r["index"] for r in results]).astype(np.int32),
        cond=np.concatenate([np.full(len(r["labels"]), i, np.int16) for i, r in enumerate(results)]),
    )

    summary = os.path.join(args.out_dir, "summary.csv")
    fields = ["run", "model", "method", "protocol", "batch_size", "lr", "shuffle_seed",
              "corruption", "severity", "error", "ece", "seconds", "ms_per_batch"]
    new_file = not os.path.exists(summary)
    with open(summary, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        if new_file:
            w.writeheader()
        for r in results:
            n_b = int(np.ceil(len(r["labels"]) / batch_size))
            w.writerow(dict(run=run, model=model_tag(args.model), method=args.method,
                            protocol=args.protocol, batch_size=batch_size, lr=meta["lr"],
                            shuffle_seed=args.shuffle_seed, corruption=r["corruption"],
                            severity=r["severity"], error=round(100 * error(r["logits"], r["labels"]), 3),
                            ece=round(100 * ece(r["logits"], r["labels"]), 3),
                            seconds=round(r["seconds"], 3), ms_per_batch=round(1000 * r["seconds"] / n_b, 2)))

    corr = [r for r in results if r["corruption"] != "clean"]
    mean_err = 100 * np.mean([error(r["logits"], r["labels"]) for r in corr])
    print(f"mean corruption error {mean_err:.2f}% over {len(corr)} conditions, {total:.0f}s total")
    return meta


if __name__ == "__main__":
    main()
