# Does Augmentation Make Test-Time Adaptation Safer?

CMPE 252: Artificial Intelligence, San José State University, Fall 2026

**Team:** Kirtan Patel and Yash Patel

> Status: all 6 models are trained and tested. The final report and slides are in progress.

## What is this project?

Image models are good at clean pictures. They make many more mistakes on blurry, noisy, or foggy pictures.

There are two ways to help:

1. **Train better.** Show the model edited pictures while training (CutMix, AugMix).
2. **Adjust while testing.** Let the model update itself a little on the new pictures (BN-adapt, Tent).

Adjusting while testing can sometimes go wrong. For example, it can fail when the pictures keep changing or come in small groups.

**Our question:** if a model was trained with edited pictures, is adjusting it while testing safer?

## What we do

1. Train a ResNet-18 model in 4 ways: Standard, CutMix, AugMix, and AugMix without its extra loss.
2. Test each model with: no change, BN-adapt, and Tent.
3. Test in normal conditions and in 2 hard conditions (changing pictures and small groups).
4. Compare the results.

## Data

- **CIFAR-10:** 60,000 small pictures in 10 classes. Used for training.
- **CIFAR-10-C:** the CIFAR-10 test pictures with 15 kinds of damage. Used for testing only.

Both are free and public.

## Repository layout

```
model.py        ResNet-18 for 32x32 images (takes images in [0, 1])
augmix.py       AugMix ops (no brightness/contrast/color/sharpness) and JSD loss
train.py        Train one model: standard, cutmix, augmix, or augmix_nojsd
time_epochs.py  Time 2 epochs of each method -> results/epoch_timing.csv
test_ops.py     Checks AugMix never uses brightness/contrast/color/sharpness
colab_train.ipynb  Train one model on Colab (set METHOD and SEED)
kaggle_train.ipynb Train one model on Kaggle (set METHOD and SEED)
results/        Small result files (timing, summaries)
```

## Stage 1 progress

| Run | Clean acc. | Hours (T4) |
|---|---|---|
| Standard, seed 1 | 95.08% | 0.65 |
| Standard, seed 2 | 94.97% | 0.58 |
| AugMix + JSD, seed 1 | 95.21% | 3.93 |
| AugMix + JSD, seed 2 | 95.29% | 2.24 (Kaggle) |
| AugMix without JSD, seed 1 | 95.31% | 1.13 (Kaggle) |
| CutMix, seed 1 | 96.18% | 0.52 (Kaggle) |

Time per epoch on a T4 (2 CPU workers): standard 21 s, cutmix 23 s, augmix_nojsd 91 s, augmix 181 s.

## Quick start (training)

```bash
pip install -r requirements.txt
python train.py --method standard --seed 1
python train.py --method augmix --seed 1
python train.py --method augmix_nojsd --seed 1
python train.py --method cutmix --seed 1
```

Quick test on a laptop (5 batches per epoch, 1 epoch):

```bash
python train.py --method standard --fast 5 --stop 1 --out runs/test
```

If training stops, run the same command again. It continues from `runs/<method>_seed<N>/last.pt`.

See [docs/member1_training.md](docs/member1_training.md) for Colab / Kaggle steps.

Full Stage 1 progress report: [docs/progress_report.md](docs/progress_report.md).

## Stage 2: test-time adaptation and evaluation (Member 2)

Progress, results and next steps: [docs/stage2_report.md](docs/stage2_report.md).

```
tta/data.py      CIFAR-10 / CIFAR-10-C loading (images in [0, 1])
tta/models.py    CIFAR ResNet-18 (normalizes inside), checkpoint loader, RobustBench models
tta/methods.py   none / BN-adapt / Tent (Tent adapted from DequanWang/tent, MIT)
tta/streams.py   episodic, continual, small-batch protocols
tta/metrics.py   error, ECE, paired bootstrap (harm rate), collapse score
eval.py          one model x one method x one protocol; saves all predictions
analyze.py       all metrics, condition maps, and stream plots from saved predictions (CPU)
notebooks/stage2_colab.ipynb   full runs on a free Colab T4
```

Quick checks on a laptop (no download needed):

```bash
python -m pytest -q tests
python scripts/smoke_test.py
```

Full runs on a GPU:

```bash
CIFAR10C_HF_REPO=YashP54/cifar10-c bash scripts/download_cifar10c.sh   # HF mirror, Zenodo fallback
python eval.py --model rb_standard_wrn28 --method tent --severities 5   # E1 check
python eval.py --model runs/standard_seed1/final.pt --method tent --protocol continual
python analyze.py
```

Stage 1 checkpoints (`runs/<method>_seed<N>/final.pt` from `train.py`) load directly:
`model.py` and `tta/models.py` use the same layer names and both normalize inside the model.

## Code we build on

- [AugMix](https://github.com/google-research/augmix)
- [Tent](https://github.com/DequanWang/tent)
- [RobustBench](https://github.com/RobustBench/robustbench)
