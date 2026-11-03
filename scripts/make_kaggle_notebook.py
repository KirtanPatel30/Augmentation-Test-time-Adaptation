import json
import sys

cells = [
    ("md", """# Stage 2 on Kaggle: test-time adaptation on CIFAR-10-C

Runs every Stage 2 experiment (E2 + E3, E4, Tent shuffle orders) on the Stage 1 models in the background.

**Setup (once):**
1. Upload the Stage 1 models as a Kaggle dataset: a zip with one folder per run, e.g.
   `standard_seed1/final.pt`, `standard_seed2/final.pt`, ... (Datasets → New Dataset).
2. In this notebook: **Settings → Accelerator → GPU T4 x2** (or P100), **Internet → On**,
   and **Add Input** → your models dataset.
3. **Save Version → Save & Run All (Commit)**. You can close the browser; Kaggle runs it to the end.
4. When it finishes, open the version's **Output** tab and download `stage2_results.zip`.

Every `final.pt` under `/kaggle/input` is evaluated; the run is named after its folder (`standard_seed1`, ...)."""),
    ("code", """BRANCH = 'member2/stage2-adaptation-eval'   # change to 'main' after the merge
HF_REPO = 'YashP54/cifar10-c'               # Hugging Face mirror of CIFAR-10-C

!nvidia-smi --query-gpu=name,memory.total --format=csv
%cd /tmp
!rm -rf repo && git clone -q -b $BRANCH https://github.com/KirtanPatel30/Augmentation-Test-time-Adaptation.git repo
%cd /tmp/repo
!git log --oneline -1
# Data stays in /tmp (not saved). Results are written straight to /kaggle/working/results,
# which Kaggle saves as the notebook's output. (Not a symlink: the repo already has a results/ folder.)
OUT = '/kaggle/working/results'
!mkdir -p /tmp/data $OUT && ln -sfn /tmp/data data"""),
    ("code", """!CIFAR10C_HF_REPO=$HF_REPO bash scripts/download_cifar10c.sh data
!python -m pytest -q tests"""),
    ("code", """import glob, os
# Models whose full set of runs is already done: they only get the extra Tent continual runs.
DONE = ['standard_seed1', 'standard_seed2']
CKPTS = sorted(glob.glob('/kaggle/input/**/final.pt', recursive=True))
NEW = [c for c in CKPTS if os.path.basename(os.path.dirname(c)) not in DONE]
OLD = [c for c in CKPTS if c not in NEW]
print('full runs:', *NEW, sep='\\n  ')
print('extra Tent continual runs only:', *OLD, sep='\\n  ')
assert CKPTS, 'No final.pt found: add the Stage 1 models dataset with Add Input'"""),
    ("code", """# About 45 minutes per new model on a T4, 5 minutes per done model. The log is also saved to the output.
import subprocess
def run_stage2(ckpts, **env):
    if not ckpts:
        return
    with open(f'{OUT}/run_stage2.log', 'a') as log:
        p = subprocess.Popen(['bash', 'scripts/run_stage2.sh', *ckpts], stdout=subprocess.PIPE,
                             stderr=subprocess.STDOUT, text=True, env={**os.environ, 'OUT': OUT, **env})
        for line in p.stdout:
            log.write(line); log.flush()
            if 'Warning' not in line and 'pkg_resources' not in line:
                print(line, end='')
    print('exit code', p.wait())

run_stage2(NEW)
run_stage2(OLD, REPEATS_ONLY='1')"""),
    ("code", """import pandas as pd
pd.set_option('display.width', 200)
pd.read_csv(f'{OUT}/tables/runs.csv')"""),
    ("code", """!cd /kaggle/working && zip -qr stage2_results.zip results && ls -lh stage2_results.zip && unzip -l stage2_results.zip | tail -n 1"""),
]


def to_cell(kind, src):
    lines = src.splitlines(keepends=True)
    if kind == "md":
        return {"cell_type": "markdown", "metadata": {}, "source": lines}
    return {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": lines}


nb = {
    "nbformat": 4, "nbformat_minor": 5,
    "metadata": {"kernelspec": {"name": "python3", "display_name": "Python 3", "language": "python"},
                 "language_info": {"name": "python"},
                 "kaggle": {"accelerator": "nvidiaTeslaT4", "isInternetEnabled": True, "isGpuEnabled": True}},
    "cells": [to_cell(k, s) for k, s in cells],
}
with open(sys.argv[1], "w") as f:
    json.dump(nb, f, indent=1)
