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

## Code we build on

- [AugMix](https://github.com/google-research/augmix)
- [Tent](https://github.com/DequanWang/tent)
- [RobustBench](https://github.com/RobustBench/robustbench)
