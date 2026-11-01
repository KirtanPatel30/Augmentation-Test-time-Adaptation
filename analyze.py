"""Compute every Stage 2 metric from saved predictions. CPU only.

    python analyze.py                      # reads results/preds/*.npz
    python analyze.py --n-boot 200         # faster bootstrap for a quick look

Writes:
    results/tables/runs.csv             one row per run: clean / corruption / sev-5 error,
                                        ECE, harm and help rate vs. no adaptation, collapse score
    results/tables/conditions.csv       one row per run x condition: error, ECE, effect (+1 harm,
                                        -1 help, 0 none) with its bootstrap interval
    results/figures/map_<run>.png       75-condition help/harm map (E5)
    results/figures/stream_<run>.png    accuracy and collapse over the stream (continual runs, E4)
"""
import argparse
import glob
import json
import os
from collections import defaultdict

import numpy as np

from tta.metrics import collapse_score, ece, error, paired_bootstrap


def load_run(path):
    z = np.load(path, allow_pickle=False)
    meta = json.loads(str(z["meta"]))
    logits = z["logits"].astype(np.float32)
    labels, index, cond = z["labels"].astype(np.int64), z["index"], z["cond"]
    conds = {}
    for i, name in enumerate(meta["conditions"]):
        m = cond == i
        c, s = name.split(":")
        conds[(c, int(s))] = dict(logits=logits[m], labels=labels[m], index=index[m])
    meta["stream_preds"] = logits.argmax(1)
    meta["stream_labels"] = labels
    return meta, conds


def aligned_correct(entry):
    """Per-image correctness sorted by original test index, so runs can be paired."""
    order = np.argsort(entry["index"], kind="stable")
    return (entry["logits"].argmax(1) == entry["labels"])[order], entry["index"][order]


def find_reference(runs, meta):
    """The no-adaptation run of the same model (any protocol: source predictions do not depend on order)."""
    best = None
    for other_meta, other_conds in runs:
        if other_meta["model"] != meta["model"] or other_meta["method"] != "none":
            continue
        if other_meta["n_examples"] != meta["n_examples"]:
            continue
        if best is None or len(other_conds) > len(best[1]):
            best = (other_meta, other_conds)
    return best


def plot_condition_map(run, effects, out_path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap

    from tta.data import CORRUPTIONS, SEVERITIES

    grid = np.full((len(CORRUPTIONS), len(SEVERITIES)), np.nan)
    for (c, s), e in effects.items():
        if c in CORRUPTIONS:
            grid[CORRUPTIONS.index(c), s - 1] = e
    fig, ax = plt.subplots(figsize=(4, 6))
    cmap = ListedColormap(["#3b82f6", "#e5e7eb", "#ef4444"])  # help, none, harm
    ax.imshow(grid, cmap=cmap, vmin=-1, vmax=1, aspect="auto")
    ax.set_xticks(range(5), [str(s) for s in SEVERITIES])
    ax.set_yticks(range(len(CORRUPTIONS)), CORRUPTIONS, fontsize=8)
    ax.set_xlabel("severity")
    ax.set_title(f"{run}\nblue: helps, red: harms", fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_stream(meta, conds, out_path, window=1000):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    preds, labels = meta["stream_preds"], meta["stream_labels"]
    correct = (preds == labels).astype(float)
    kernel = np.ones(window) / window
    acc = np.convolve(correct, kernel, mode="valid")
    col = collapse_score(preds, window)
    fig, ax = plt.subplots(figsize=(9, 3.5))
    ax.plot(acc, label="accuracy (window 1000)")
    ax.plot(col, label="collapse score")
    start = 0
    for (c, _), entry in conds.items():
        ax.axvline(start, color="gray", lw=0.5)
        ax.text(start + 100, 0.02, c.replace("_", " "), rotation=90, fontsize=6, va="bottom")
        start += len(entry["labels"])
    ax.set_ylim(0, 1)
    ax.set_xlabel("test images seen")
    ax.set_title(meta["run"], fontsize=9)
    ax.legend(loc="upper right", fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--results", default="results")
    p.add_argument("--n-boot", type=int, default=1000)
    p.add_argument("--no-figures", action="store_true")
    args = p.parse_args(argv)

    paths = sorted(glob.glob(os.path.join(args.results, "preds", "*.npz")))
    if not paths:
        raise SystemExit(f"no predictions in {args.results}/preds; run eval.py first")
    runs = [load_run(path) for path in paths]
    os.makedirs(os.path.join(args.results, "tables"), exist_ok=True)
    os.makedirs(os.path.join(args.results, "figures"), exist_ok=True)

    run_rows, cond_rows = [], []
    for meta, conds in runs:
        ref = find_reference(runs, meta) if meta["method"] != "none" else None
        effects = {}
        for (c, s), entry in conds.items():
            row = dict(run=meta["run"], corruption=c, severity=s,
                       error=100 * error(entry["logits"], entry["labels"]),
                       ece=100 * ece(entry["logits"], entry["labels"]))
            if ref is not None and (c, s) in ref[1]:
                new_correct, new_idx = aligned_correct(entry)
                ref_correct, ref_idx = aligned_correct(ref[1][(c, s)])
                assert np.array_equal(new_idx, ref_idx), "runs do not cover the same images"
                diff, low, high = paired_bootstrap(ref_correct, new_correct, n_boot=args.n_boot)
                effects[(c, s)] = 1 if low > 0 else (-1 if high < 0 else 0)
                row.update(diff_vs_none=100 * diff, ci_low=100 * low, ci_high=100 * high,
                           effect=effects[(c, s)])
            cond_rows.append(row)

        rows = [r for r in cond_rows if r["run"] == meta["run"]]
        corr = [r for r in rows if r["corruption"] != "clean"]
        sev5 = [r for r in corr if r["severity"] == 5]
        clean = [r for r in rows if r["corruption"] == "clean"]
        col = collapse_score(meta["stream_preds"])
        run_rows.append(dict(
            run=meta["run"], model=os.path.basename(meta["model"]), method=meta["method"],
            protocol=meta["protocol"], batch_size=meta["batch_size"], lr=meta["lr"],
            shuffle_seed=meta["shuffle_seed"], n_conditions=len(corr),
            clean_error=clean[0]["error"] if clean else None,
            corruption_error=np.mean([r["error"] for r in corr]) if corr else None,
            sev5_error=np.mean([r["error"] for r in sev5]) if sev5 else None,
            ece=np.mean([r["ece"] for r in corr]) if corr else None,
            harm_rate=np.mean([e == 1 for e in effects.values()]) if effects else None,
            help_rate=np.mean([e == -1 for e in effects.values()]) if effects else None,
            collapse_max=float(col.max()), collapse_mean=float(col.mean()),
            seconds=meta["seconds"],
        ))

        if not args.no_figures:
            if len(effects) >= 15:
                plot_condition_map(meta["run"], effects,
                                   os.path.join(args.results, "figures", f"map_{meta['run']}.png"))
            if meta["protocol"] == "continual":
                plot_stream(meta, conds, os.path.join(args.results, "figures", f"stream_{meta['run']}.png"))

    import csv

    def write(rows, path):
        keys = list(dict.fromkeys(k for r in rows for k in r))
        with open(path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=keys)
            w.writeheader()
            for r in rows:
                w.writerow({k: (round(v, 3) if isinstance(v, float) else v) for k, v in r.items()})

    write(run_rows, os.path.join(args.results, "tables", "runs.csv"))
    write(cond_rows, os.path.join(args.results, "tables", "conditions.csv"))

    print(f"{'run':60s} {'corr err':>8s} {'sev5':>6s} {'ECE':>6s} {'harm':>5s} {'collapse':>8s}")
    fmt = lambda v, f: (f % v) if v is not None else "-"
    for r in run_rows:
        print(f"{r['run'][:60]:60s} {fmt(r['corruption_error'], '%8.2f')} {fmt(r['sev5_error'], '%6.2f')} "
              f"{fmt(r['ece'], '%6.2f')} {fmt(r['harm_rate'], '%5.2f')} {r['collapse_max']:8.3f}")


if __name__ == "__main__":
    main()
