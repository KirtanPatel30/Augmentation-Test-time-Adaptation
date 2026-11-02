import json
import sys

cells = [
    ("md", """# Stage 2 on Colab: test-time adaptation on CIFAR-10-C

Run this on a free **T4 GPU** (Runtime → Change runtime type → T4 GPU).

* CIFAR-10-C comes from our Hugging Face mirror (`HF_REPO` below): fast, no login. If the mirror is not there yet, it falls back to Zenodo (about 17 minutes).
* By default everything lives on the Colab disk and is lost when the session ends: download `results/` at the end (last cell).
  Set `USE_DRIVE = True` to keep data and results in Google Drive instead.
* Stage 1 checkpoints go in `checkpoints/` (upload them, or put them in `MyDrive/cmpe252/checkpoints/` with Drive on)."""),
    ("code", """USE_DRIVE = False
HF_REPO = 'YashP54/cifar10-c'   # Hugging Face mirror of CIFAR-10-C

if USE_DRIVE:
    from google.colab import drive
    drive.mount('/content/drive')
    DRIVE = '/content/drive/MyDrive/cmpe252'
else:
    DRIVE = '/content/cmpe252'
!mkdir -p $DRIVE/data $DRIVE/models $DRIVE/results $DRIVE/checkpoints
!nvidia-smi --query-gpu=name,memory.total --format=csv"""),
    ("code", """BRANCH = 'member2/stage2-adaptation-eval'   # change to 'main' after the merge
%cd /content
!rm -rf Augmentation-Test-time-Adaptation
!git clone -q -b $BRANCH https://github.com/KirtanPatel30/Augmentation-Test-time-Adaptation.git
%cd /content/Augmentation-Test-time-Adaptation
!pip -q install git+https://github.com/RobustBench/robustbench.git
# Point data/, models/, results/, checkpoints/ at $DRIVE (Google Drive or the Colab disk).
# The repo has its own results/ folder, so move it aside first (a symlink would land inside it).
!mv results results_repo && ln -sfn $DRIVE/data data && ln -sfn $DRIVE/models models && ln -sfn $DRIVE/results results && ln -sfn $DRIVE/checkpoints checkpoints"""),
    ("code", """# Hugging Face mirror first, Zenodo as fallback.
!CIFAR10C_HF_REPO=$HF_REPO bash scripts/download_cifar10c.sh data
!python -m pytest -q tests"""),
    ("md", """### One time only: upload CIFAR-10-C to Hugging Face

Skip this once the mirror exists. Needs a Hugging Face **write** token (huggingface.co → Settings → Access Tokens); paste it when asked."""),
    ("code", """from huggingface_hub import login
login()   # paste your write token in the box
!python scripts/upload_cifar10c_to_hf.py --repo $HF_REPO --data-dir data/CIFAR-10-C"""),
    ("md", """## E1: pipeline checks against published numbers

Target severity-5 error (Tent repo): Standard WRN-28-10 **43.5 / 20.4 / 18.6**, AugMix WRN-40-2 **18.3 / 14.5 / 12.1** (none / BN / Tent)."""),
    ("code", """for model in ['rb_standard_wrn28', 'rb_augmix_wrn40']:
    for method in ['none', 'bn', 'tent']:
        !python eval.py --model $model --method $method --protocol episodic --severities 5"""),
    ("code", """!python analyze.py --no-figures"""),
    ("md", """## Tent learning rate (4 extra corruptions only)"""),
    ("code", """for lr in ['1e-4', '3e-4', '1e-3', '3e-3']:
    !python eval.py --model rb_standard_wrn28 --method tent --corruptions extra --severities 5 --lr $lr --tag tune --out-dir results/tune
!python analyze.py --results results/tune --no-figures"""),
    ("md", """## E2 + E3: no adaptation, BN-adapt and Tent on Stage 1 models (all 75 conditions)

* **E2 (RQ1, RQ2):** the `none` runs give each model's clean error and corruption error, for example AugMix with vs. without JSD. Training cost comes from Stage 1.
* **E3 (RQ3):** the `bn` and `tent` runs against those `none` runs."""),
    ("code", """# Stage 1 models: runs/<method>_seed<N>/final.pt from train.py.
# Either upload the run folders into checkpoints/ (Files pane), or set RUNS_FOLDER to a
# Google Drive folder link shared as "Anyone with the link" and it downloads them (no login).
RUNS_FOLDER = ''
if RUNS_FOLDER:
    !gdown -q --folder "$RUNS_FOLDER" -O checkpoints
    !find checkpoints -name last.pt -delete
import glob
CKPTS = sorted(glob.glob('checkpoints/*.pt') + glob.glob('checkpoints/**/final.pt', recursive=True))
print(CKPTS)
for ckpt in CKPTS:
    for method in ['none', 'bn', 'tent']:
        !python eval.py --model $ckpt --method $method --protocol episodic --clean"""),
    ("md", """## E4: difficult streams (main experiment)"""),
    ("code", """for ckpt in CKPTS:
    for method in ['bn', 'tent']:
        !python eval.py --model $ckpt --method $method --protocol continual
        !python eval.py --model $ckpt --method $method --protocol small_batch
    # 3 shuffle orders for the severity-5 Tent runs
    for seed in [1, 2, 3]:
        !python eval.py --model $ckpt --method tent --protocol episodic --severities 5 --shuffle-seed $seed"""),
    ("code", """!python analyze.py
import pandas as pd
pd.read_csv('results/tables/runs.csv')"""),
    ("code", """# Without Drive, download the results before the session ends.
!zip -qr /content/results.zip results/summary.csv results/tables results/figures results/tune
from google.colab import files
files.download('/content/results.zip')"""),
]


def to_cell(kind, src):
    lines = src.splitlines(keepends=True)
    if kind == "md":
        return {"cell_type": "markdown", "metadata": {}, "source": lines}
    return {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": lines}


nb = {
    "nbformat": 4, "nbformat_minor": 5,
    "metadata": {"accelerator": "GPU", "colab": {"provenance": []},
                 "kernelspec": {"name": "python3", "display_name": "Python 3"}},
    "cells": [to_cell(k, s) for k, s in cells],
}
with open(sys.argv[1], "w") as f:
    json.dump(nb, f, indent=1)
