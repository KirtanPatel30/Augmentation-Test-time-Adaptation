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

## Stage 2: test-time adaptation and evaluation (Member 2)

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
python eval.py --model checkpoints/standard_s1.pt --method tent --protocol continual
python analyze.py
```

**Checkpoint format for Stage 1:** save `{"model": model.state_dict(), ...}` from
`tta.models.ResNet18` (or the same module names). The model takes images in [0, 1]
and normalizes inside, so training must not normalize in the data loader.

## Code we build on

- [AugMix](https://github.com/google-research/augmix)
- [Tent](https://github.com/DequanWang/tent)
- [RobustBench](https://github.com/RobustBench/robustbench)
