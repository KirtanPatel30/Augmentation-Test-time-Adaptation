# Does Augmentation Make Test-Time Adaptation Safer?

CMPE 252: Artificial Intelligence, San José State University, Fall 2026

**Team:** Kirtan Patel and Yash Patel

> 🚧 Work in progress. Code and results coming soon.

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
results/        Small result files (timing, summaries)
```

## Stage 1 progress

| Run | Clean acc. | Hours (T4) |
|---|---|---|
| Standard, seed 1 | 95.08% | 0.65 |
| Standard, seed 2 | 94.97% | 0.58 |
| AugMix + JSD, seed 1 | 95.21% | 3.93 |
| AugMix + JSD, seed 2 | 95.29% | 2.24 (Kaggle) |

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

## Code we build on

- [AugMix](https://github.com/google-research/augmix)
- [Tent](https://github.com/DequanWang/tent)
- [RobustBench](https://github.com/RobustBench/robustbench)
