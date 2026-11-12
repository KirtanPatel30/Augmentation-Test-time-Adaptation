"""Help/harm maps of BN-adapt and Tent (episodic) for all 6 models in one figure (final report, RQ5).

    python scripts/harm_maps_figure.py --results results/all_models

Reads <results>/tables/conditions.csv (from analyze.py) and writes <results>/figures/harm_maps.png.
Blue: significantly better than no adaptation, red: significantly worse, grey: no significant change.
"""
import argparse
import csv
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.colors import ListedColormap  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tta.data import CORRUPTIONS, SEVERITIES  # noqa: E402

MODELS = [("standard_seed1", "Standard s1"), ("standard_seed2", "Standard s2"), ("cutmix_seed1", "CutMix"),
          ("augmix_nojsd_seed1", "AugMix no JSD"), ("augmix_seed1", "AugMix+JSD s1"),
          ("augmix_seed2", "AugMix+JSD s2")]
METHODS = [("bn_episodic_bs200", "BN-adapt"), ("tent_episodic_bs200_lr0.001", "Tent")]
COLORS = ["#3b82f6", "#e5e7eb", "#ef4444"]  # help, none, harm (same as analyze.py)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--results", default="results")
    args = p.parse_args(argv)

    effects = {}
    with open(os.path.join(args.results, "tables", "conditions.csv"), newline="") as f:
        for r in csv.DictReader(f):
            if r["corruption"] in CORRUPTIONS and r["effect"] != "":
                effects[(r["run"], r["corruption"], int(r["severity"]))] = int(float(r["effect"]))

    fig, axes = plt.subplots(len(METHODS), len(MODELS), figsize=(13, 8.5), sharex=True, sharey=True)
    cmap = ListedColormap(COLORS)
    for i, (suffix, method) in enumerate(METHODS):
        for j, (model, name) in enumerate(MODELS):
            run = f"{model}_{suffix}"
            grid = np.array([[effects[(run, c, s)] for s in SEVERITIES] for c in CORRUPTIONS])
            ax = axes[i, j]
            ax.imshow(grid, cmap=cmap, vmin=-1, vmax=1, aspect="auto")
            ax.set_xticks(range(len(SEVERITIES)), [str(s) for s in SEVERITIES], fontsize=8)
            ax.set_yticks(range(len(CORRUPTIONS)), [c.replace("_", " ") for c in CORRUPTIONS], fontsize=8)
            ax.set_xticks(np.arange(-0.5, len(SEVERITIES)), minor=True)
            ax.set_yticks(np.arange(-0.5, len(CORRUPTIONS)), minor=True)
            ax.grid(which="minor", color="white", linewidth=1)
            ax.tick_params(which="minor", length=0)
            ax.set_title(f"{name}\n{method}: {np.sum(grid == 1)} harm, {np.sum(grid == -1)} help", fontsize=9)
            if i == len(METHODS) - 1:
                ax.set_xlabel("severity", fontsize=8)
    fig.legend(handles=[Patch(color=COLORS[0], label="helps"), Patch(color=COLORS[1], label="no significant change"),
                        Patch(color=COLORS[2], label="harms")],
               loc="lower center", ncol=3, frameon=False, fontsize=9)
    fig.suptitle("Episodic test: adaptation vs. no adaptation (95% paired bootstrap)", fontsize=11)
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    out = os.path.join(args.results, "figures", "harm_maps.png")
    fig.savefig(out, dpi=150)
    print("wrote", out)


if __name__ == "__main__":
    main()
