import csv

import matplotlib.pyplot as plt
import numpy as np

T = "results/all_models/tables/"
NAMES = {"standard": "Standard", "cutmix": "CutMix",
         "augmix_nojsd": "AugMix without JSD", "augmix": "AugMix + JSD"}
COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]
MARKS = ["o", "s", "^", "D"]
ADAPT = {"none": "No adaptation", "bn": "BN-adapt", "tent": "Tent"}
COLS = ["clean_error", "corruption_error", "sev5_error", "ece", "harm_rate", "help_rate"]


def read(name):
    with open(T + name, newline="") as f:
        return list(csv.DictReader(f))


def use(run):
    return "episodic" in run and "shuf" not in run


def mean(v):
    return round(float(np.mean(v)), 3) if v else ""


runs = read("runs.csv")
conds = read("conditions.csv")
info = {r["run"]: r for r in runs}

table = []
for t in NAMES:
    for m in ADAPT:
        g = [r for r in runs if use(r["run"]) and r["model"].split("_seed")[0] == t and r["method"] == m]
        row = {"training": t, "adaptation": m, "seeds": len(g)}
        for c in COLS:
            row[c] = mean([float(r[c]) for r in g if r[c] != ""])
        table.append(row)

with open(T + "seed_means.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=["training", "adaptation", "seeds"] + COLS)
    w.writeheader()
    w.writerows(table)

fig, ax = plt.subplots(1, 3, figsize=(11, 3.9), sharey=True)
for a, m in zip(ax, ADAPT):
    for i, t in enumerate(NAMES):
        y = []
        for s in range(1, 6):
            v = []
            for r in conds:
                k = info[r["run"]]
                if use(r["run"]) and r["corruption"] != "clean" and int(r["severity"]) == s \
                        and k["model"].split("_seed")[0] == t and k["method"] == m:
                    v.append(float(r["error"]))
            y.append(np.mean(v))
        a.plot(range(1, 6), y, color=COLORS[i], marker=MARKS[i], linewidth=2, label=NAMES[t])
    a.set_title(ADAPT[m], loc="left")
    a.set_xlabel("Severity")
    a.set_xticks(range(1, 6))
    a.set_ylim(0, 50)
    a.grid(axis="y", color="#e6e5e0")
    a.spines[["top", "right", "left"]].set_visible(False)

ax[0].set_ylabel("Mean error over 15 corruptions (%)")
h, l = ax[0].get_legend_handles_labels()
fig.legend(h, l, loc="lower center", ncol=4, frameon=False)
fig.tight_layout(rect=(0, 0.09, 1, 1))
fig.savefig("results/all_models/figures/severity_curves.png", dpi=200)
print("saved seed_means.csv and severity_curves.png")
