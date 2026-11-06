# Progress Report: Member 1 (Stage 1, Training)

**Project:** Does Augmentation Make Test-Time Adaptation Safer?
**Course:** CMPE 252, San José State University, Fall 2026
**Member 1:** Kirtan Patel (training the models)
**Member 2:** Yash Patel (testing the models with BN-adapt and Tent on CIFAR-10-C)
**Date:** November 5, 2026

---

## 1. The big picture

We want to know one thing: **if a model is trained with edited pictures (augmentation), is it safer to let it adjust itself at test time?**

The work has two stages:

- **Stage 1 (me):** train the same image model in 4 different ways and save the trained models.
- **Stage 2 (Yash):** take those saved models and test them on damaged pictures (CIFAR-10-C), with and without test-time adjusting.

This report covers Stage 1. Stage 2 is in [stage2_report.md](stage2_report.md).

---

## 2. What Member 1 has to do

| Task | Status |
|---|---|
| Write the model (ResNet-18 for small 32x32 pictures) | Done |
| Write the 4 training methods | Done |
| Test that AugMix never uses the "forbidden" edits | Done |
| Time 2 epochs of each method | Done |
| Train Standard, 2 seeds | Done |
| Train AugMix + JSD, 2 seeds | Done |
| Train AugMix without JSD, 1 seed | Done |
| Train CutMix, 1 seed | Done |
| Record how much compute each run cost | Done |
| Weeks 5–6: help with hard test conditions, confidence intervals and condition maps | Done in Stage 2 (Yash's runs and `analyze.py`) |

### Where we are in the 8-week plan

Stage 1 is **on time**. The plan says all models should be ready at the end of week 4, and they are.

| Week | Member 1 task (from the proposal) | Status |
|---|---|---|
| 1 (Oct 12) | Make the repo, write `train.py` with mixed precision and resume, time 2 epochs per setting, start Standard seed 1 | Done |
| 2 (Oct 19) | Standard seed 2, start AugMix + JSD seed 1, unit test for forbidden edits | Done |
| 3 (Oct 26) | AugMix + JSD seed 2, AugMix without JSD, start CutMix | Done |
| 4 (Nov 2) | Finish CutMix, record the cost of each run | Done (all models ready) |
| 5 (Nov 9) | Help with the difficult-stream runs | Done early (Yash ran them on all 6 models) |
| 6 (Nov 16) | Bootstrap intervals and condition maps | Done early (`analyze.py`) |
| 7 (Nov 23) | Together: figures and report draft | Next |
| 8 (Nov 30) | Together: code clean-up, README, final report, slides | To do |

---

## 3. The 4 training methods (in simple words)

1. **Standard:** normal training. Pictures are only randomly cropped and flipped.
2. **CutMix:** cut a square from one picture and paste it onto another. The label is mixed in the same amount (for example 70% cat, 30% dog).
3. **AugMix (with JSD):** make 2 extra edited copies of each picture (rotated, sheared, posterized and so on) and blend them with the original. The model is also asked to give **the same answer for all 3 copies**. This extra rule is called the JSD loss.
4. **AugMix without JSD:** the same edited pictures, but without the "same answer" rule. This lets us see if the extra rule is what helps.

**Forbidden edits:** CIFAR-10-C (the test set) includes damage like brightness, contrast, color and sharpness changes. If we trained with those same edits, the test would be unfair. So AugMix here **never** uses brightness, contrast, color or sharpness, and a test checks this.

---

## 4. The code (all in the repo)

| File | What it does |
|---|---|
| `model.py` | The ResNet-18 model. It takes pictures with values from 0 to 1 and normalizes them inside, so Yash's code can use it directly. |
| `augmix.py` | The AugMix edits and the JSD "same answer" loss. |
| `train.py` | Trains one model. You choose the method and the seed. It saves progress every 5 epochs and can continue if it stops. |
| `test_ops.py` | Tests: no forbidden edits, edits keep the picture size, AugMix output is valid, JSD loss is safe. |
| `time_epochs.py` | Times 2 epochs of each method. |
| `colab_train.ipynb` | Runs training on Google Colab. |
| `kaggle_train.ipynb` | Runs training on Kaggle. |

**Training settings (same for every run):** 100 epochs, batch size 128, SGD with learning rate 0.1, momentum 0.9 (Nesterov), weight decay 0.0005 and cosine learning-rate decay.

---

## 5. Results so far

**Clean accuracy** = how often the model is right on the normal CIFAR-10 test pictures (10,000 pictures).

| Run | Clean accuracy | GPU hours | Where |
|---|---|---|---|
| Standard, seed 1 | 95.08% | 0.65 | Colab T4 |
| Standard, seed 2 | 94.97% | 0.58 | Colab T4 |
| AugMix + JSD, seed 1 | 95.21% | 3.93 | Colab T4 (epochs 1–85) + Kaggle T4 (epochs 86–100) |
| AugMix + JSD, seed 2 | 95.29% | 2.24 | Kaggle T4 |
| AugMix without JSD, seed 1 | 95.31% | 1.13 | Kaggle T4 |
| CutMix, seed 1 | 96.18% | 0.52 | Kaggle T4 |

**What this shows so far:**

- All models are about **95–96% accurate** on clean pictures, which is normal for ResNet-18 on CIFAR-10.
- CutMix is the best on clean pictures (96.18%).
- Seeds of the same method are very close (within about 0.1%), so the results are stable.
- AugMix is slightly better on clean pictures than Standard here. The real question, how they behave on damaged pictures with Tent and BN-adapt, is answered in Stage 2.

Every AugMix model was re-checked on the laptop after downloading, and it gave the exact same accuracy.

---

## 6. Compute cost

**Time for one epoch (on a T4 GPU):**

| Method | Colab (2 CPU workers) | 100 epochs on Colab |
|---|---|---|
| Standard | 21 s | 0.58 h |
| CutMix | 23 s | 0.64 h |
| AugMix without JSD | 91 s | 2.53 h |
| AugMix + JSD | 181 s | 5.02 h |

AugMix is slow because the edits are done on the CPU, not the GPU. Kaggle gives 4 CPU workers instead of 2, so AugMix + JSD took **81 s per epoch on Kaggle** and AugMix without JSD took **41 s per epoch**, both more than 2x faster than Colab.

**Total GPU hours:** about **9.0 hours** for all 6 training runs, plus about 0.2 hours for the timing runs. This is inside the 6–11 hours we planned in the proposal.

---

## 7. Problems we hit and how we fixed them

| Problem | Fix |
|---|---|
| My laptop has no NVIDIA GPU, so training would take days. | Train on free cloud GPUs (Colab and Kaggle). The laptop is used for writing code and quick tests. |
| Downloading CIFAR-10 from its official site was very slow (about 80 KB/s). | Downloaded it once, put the file in Google Drive and as a Kaggle dataset, and copy it from there. |
| AugMix + JSD training crashed with `nan` (not a number) after a few steps. | The JSD loss took the log of 0 in some cases. I rewrote it in a safe way and added a test for it. |
| The model's layer names did not match Yash's loading code. | Renamed the layers to match. Checked that his code loads my model and gives the same outputs. |
| Colab disconnected during AugMix seed 1 at epoch 86, then hit its free GPU limit. | Training saves a checkpoint every 5 epochs. I moved the epoch-85 checkpoint to Kaggle and finished it there. |
| Re-running the Colab notebook gave a "cannot access current directory" error. | Added a step that moves to a safe folder before deleting and re-downloading the code. |

---

## 8. What is left

All 6 Stage 1 models are trained and re-checked on the laptop. Each one gives the same accuracy as in training.

Yash has all 6 `final.pt` files and has tested every model. Stage 1 and Stage 2 are both merged into `main`.

1. Weeks 7–8: write the Stage 1 part of the final report (methods, training settings, cost and problems).
2. Weeks 7–8: figures, final report, code clean-up and slides together.

---

## 9. How to use the trained models (for Yash)

Each finished run has a folder `cmpe252/runs/<method>_seed<N>/` on Drive with:

- `final.pt`: the trained model
- `summary.json`: accuracy, hours and device

Load one with:

```python
from model import load
m = load("runs/augmix_seed1/final.pt")
```

The model takes pictures with values from 0 to 1, the same as RobustBench and our Stage 2 code.
