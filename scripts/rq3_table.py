"""RQ3 / E3: gain of BN-adapt over no adaptation, and extra gain of Tent over BN-adapt, per model.

Episodic protocol, from the saved predictions. CPU only.

    python scripts/rq3_table.py --results results/all_models

Writes <results>/tables/rq3.csv: one row per model with the errors and gains (points) over all
75 conditions, severity 1 and severity 5, the gains per corruption family, and how many of the 75
conditions Tent is significantly better / worse than BN-adapt (95% paired bootstrap).
"""
import argparse
import csv
import glob
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from analyze import aligned_correct, load_run  # noqa: E402
from tta.metrics import paired_bootstrap  # noqa: E402

FAMILIES = {
    "noise": ["gaussian_noise", "shot_noise", "impulse_noise"],
    "blur": ["defocus_blur", "glass_blur", "motion_blur", "zoom_blur"],
    "weather": ["snow", "frost", "fog", "brightness"],
    "digital": ["contrast", "elastic_transform", "pixelate", "jpeg_compression"],
}


def err(conds, key):
    return 100 * (1 - aligned_correct(conds[key])[0].mean())


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--results", default="results")
    p.add_argument("--n-boot", type=int, default=1000)
    args = p.parse_args(argv)

    preds = os.path.join(args.results, "preds")
    rows = []
    for none_path in sorted(glob.glob(os.path.join(preds, "*_none_episodic_bs200.npz"))):
        model = os.path.basename(none_path)[: -len("_none_episodic_bs200.npz")]
        bn_path = os.path.join(preds, f"{model}_bn_episodic_bs200.npz")
        tent_path = os.path.join(preds, f"{model}_tent_episodic_bs200_lr0.001.npz")
        if not (os.path.exists(bn_path) and os.path.exists(tent_path)):
            continue
        none, bn, tent = (load_run(path)[1] for path in (none_path, bn_path, tent_path))
        keys = [k for k in bn if k[0] != "clean"]
        e = {name: np.array([err(c, k) for k in keys]) for name, c in [("none", none), ("bn", bn), ("tent", tent)]}
        sev = np.array([s for _, s in keys])
        corr = np.array([c for c, _ in keys])

        row = dict(model=model)
        for name, mask in [("all", sev > 0), ("sev1", sev == 1), ("sev5", sev == 5)]:
            n, b, t = (e[m][mask].mean() for m in ("none", "bn", "tent"))
            row.update({f"{name}_none": n, f"{name}_bn": b, f"{name}_tent": t,
                        f"{name}_bn_gain": n - b, f"{name}_tent_extra": b - t})
        for fam, names in FAMILIES.items():
            mask = np.isin(corr, names)
            row[f"tent_extra_{fam}"] = (e["bn"][mask] - e["tent"][mask]).mean()

        better = worse = 0
        for k in keys:
            _, low, high = paired_bootstrap(aligned_correct(bn[k])[0], aligned_correct(tent[k])[0], n_boot=args.n_boot)
            better += high < 0
            worse += low > 0
        row.update(tent_better_than_bn=better, tent_worse_than_bn=worse,
                   clean_none=err(none, ("clean", 0)), clean_bn=err(bn, ("clean", 0)),
                   clean_tent=err(tent, ("clean", 0)))
        rows.append(row)

    out = os.path.join(args.results, "tables", "rq3.csv")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        for r in rows:
            w.writerow({k: (round(v, 3) if isinstance(v, (float, np.floating)) else v) for k, v in r.items()})
    for r in rows:
        print(f"{r['model']:20s} all: BN gain {r['all_bn_gain']:6.2f}  Tent extra {r['all_tent_extra']:5.2f} | "
              f"sev5: BN gain {r['sev5_bn_gain']:6.2f}  Tent extra {r['sev5_tent_extra']:5.2f} | "
              f"Tent better/worse than BN on {r['tent_better_than_bn']}/{r['tent_worse_than_bn']} of 75")
    print("wrote", out)


if __name__ == "__main__":
    main()
