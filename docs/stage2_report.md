# Stage 2 progress report (Member 2: Yash Patel)

**Project:** Does Augmentation Make Test-Time Adaptation Safer? (CMPE 252, Fall 2026)
**Branch:** `member2/stage2-adaptation-eval` (pull request #3 into `main`)
**Last updated:** November 2, 2026 (week 4 of 8)

---

## 1. Status at a glance

| Item | Status |
|---|---|
| Stage 2 code: BN-adapt, Tent, 3 test streams, metrics, eval and analysis scripts | ✅ Done. 15 Stage 2 tests + Kirtan's 4 = 19 pass |
| CIFAR-10-C download and hosting | ✅ Done, mirrored on Hugging Face: [`YashP54/cifar10-c`](https://huggingface.co/datasets/YashP54/cifar10-c) |
| GPU notebooks | ✅ Kaggle (main, runs in the background): [`notebooks/stage2_kaggle.ipynb`](../notebooks/stage2_kaggle.ipynb). Colab: [`notebooks/stage2_colab.ipynb`](../notebooks/stage2_colab.ipynb) |
| Merge with Kirtan's Stage 1 code, checkpoint handoff | ✅ Done. Our loader gives exactly Kirtan's clean accuracy (95.08%, 94.97%) |
| **E1:** reproduce the published RobustBench + Tent numbers | ✅ Done, all 6 numbers within 0.05 points |
| Tent learning rate chosen on the 4 extra corruptions | ✅ Done, 1e-3 (the official value) |
| **E2 + E3:** none / BN-adapt / Tent, 75 conditions + clean | ✅ All 6 models |
| **E4:** continual and small-batch streams (RQ4) | ✅ All 6 models, Tent continual 3 times each |
| **E5:** help/harm maps and calibration (RQ5) | ✅ Maps for every run, comparison across augmentations (section 5.5) |
| **E6:** optional extensions | Periodic reset (`--reset-every`) implemented, not run yet |

---

## 2. What we built

| File | What it does |
|---|---|
| `tta/data.py` | Loads CIFAR-10 and CIFAR-10-C. Images are float tensors in [0, 1]. The clean test set comes from Hugging Face (`uoft-cs/cifar10`), cached as `data/cifar10_test.npz`. |
| `tta/models.py` | CIFAR ResNet-18 (normalizes inside the model, same layer names as Kirtan's `model.py`), Stage 1 checkpoint loader, RobustBench models for E1. `model_tag` names a run after the checkpoint's folder (`runs/standard_seed1/final.pt` → `standard_seed1`). |
| `tta/methods.py` | No adaptation, BN-adapt and Tent. Tent is adapted from the official code (DequanWang/tent, MIT license). Each method predicts on a batch before it updates on that batch. |
| `tta/streams.py` | Episodic (75 conditions, batch 200, reset each condition), continual (severity 5, no reset), small batch (severity 5, batch 16, Tent learning rate × 16/200). Optional shuffle seed and periodic reset. |
| `tta/metrics.py` | Error, ECE (15 bins), paired bootstrap for the harm rate, collapse score (sliding window of 1,000). |
| `eval.py` | Runs one model × method × protocol, saves every prediction (`.npz`) and appends to `summary.csv`. |
| `analyze.py` | CPU only. Builds `runs.csv` and `conditions.csv`, the 75-condition help/harm maps and the continual-stream plots. |
| `tests/test_stage2.py` | 15 unit tests: Tent updates only BN scale and shift, predicts before updating, reset works, streams, metrics, checkpoint loading, run names. |
| `scripts/run_stage2.sh` | All Stage 2 runs for a list of checkpoints (E2 + E3, E4, 3 shuffle orders), then `analyze.py`. Takes `OUT` and `EVAL_ARGS`. |
| `scripts/make_kaggle_notebook.py`, `scripts/make_colab_notebook.py` | Generate the two notebooks. |
| `scripts/smoke_test.py` | End-to-end check on fake data, no download needed. |
| `scripts/download_cifar10c.sh` | Gets CIFAR-10-C from Hugging Face, with Zenodo as the fallback. |
| `scripts/upload_cifar10c_to_hf.py` | One-time upload of the Hugging Face mirror, with an attribution page for the dataset. |

---

## 3. What we did, in order

**Weeks 1–3 (to Nov 1)**
1. Read the proposal and split out Member 2's part: Stage 2, adaptation and evaluation.
2. Cloned the team repo, made the branch, wrote the Stage 2 code and tests.
3. Smoke trial on the Mac (Apple M1, MPS). It found a real bug (section 4, change 1).
4. Ran in Colab on a T4: downloaded CIFAR-10-C, mirrored it to Hugging Face.
5. Ran E1 (6 runs) and the Tent learning-rate check (4 runs). Tables are in `results/e1_colab/`.

**Nov 2**
6. Merged Kirtan's Stage 1 work from `main` (training code, `model.py`, timing). Checked the handoff: the two Standard checkpoints load in our ResNet-18 and give exactly Kirtan's clean accuracy (95.08% and 94.97%).
7. Started the full Stage 2 run in Colab. The free runtime disconnected after 34 minutes because nobody was using the page, and the run was lost. Moved the full runs to **Kaggle**, which runs a notebook in the background ("Save & Run All") with no browser open.
8. Kaggle Version 1 finished, but its output was empty (section 4, change 9). Fixed and reran as Version 2, which finished in about 1 h 20 min on a T4 (22 runs).
9. Downloaded the Version 2 results, re-ran `analyze.py` on the Mac (its tables match Kaggle's exactly) and saved them in `results/kaggle_standard/`.
10. Kirtan uploaded the other 4 models. We added them to the Kaggle dataset and changed the notebook so models that are already done get only the 2 extra Tent continual runs (`REPEATS_ONLY`). Version 3 ran 52 runs in about 5 h 15 min with no failures.
11. Combined both Kaggle runs (72 runs, 6 models) in `results/all_models/` and analyzed them together on the Mac (tables match Kaggle's exactly).

---

## 4. What we changed along the way, and why

| # | First plan | Problem | Change |
|---|---|---|---|
| 1 | Images converted with a plain `permute`. | Tent's backward pass crashed on Apple MPS: `view size is not compatible with input tensor's size and stride`. | `to_tensor` now calls `.contiguous()`. Works on MPS and CUDA. |
| 2 | Keep CIFAR-10-C and results in Google Drive. | Mounting Drive in Colab failed (`credential propagation was unsuccessful`). | Drive is optional (`USE_DRIVE = False`). |
| 3 | Download CIFAR-10-C from Zenodo each session. | About 14–17 minutes per session. | Mirrored to Hugging Face, [`YashP54/cifar10-c`](https://huggingface.co/datasets/YashP54/cifar10-c), credited under CC BY 4.0. Zenodo is the fallback. |
| 4 | Hugging Face repo name `yashP54/...`. | The account is `YashP54`. | Fixed. |
| 5 | E2 and E3 as separate sections. | E2 has no separate evaluation step; its numbers are the "none" runs of E3. | One "E2 + E3" section. Kirtan's logs give the training cost. |
| 6 | Python 3.11. | Colab runs Python 3.13. | Works on both, no change. |
| 7 | Checkpoints named `standard_s1.pt` etc. | Stage 1 saves every model as `runs/<method>_seed<N>/final.pt`, so all runs had the same name. | `model_tag` uses the folder name. New test for it. |
| 8 | Clean CIFAR-10 through torchvision. | The Toronto server ran at about 60 kB/s on Colab. | Clean test set from Hugging Face `uoft-cs/cifar10`. All 10,000 labels checked against CIFAR-10-C. |
| 9 | Colab for the full run; results linked into the repo's `results/` folder. | Colab disconnects idle sessions. On Kaggle, `ln -sfn /kaggle/working/results results` made the link *inside* the existing `results/` folder, so Version 1 saved nothing (166 B output). | Kaggle notebook passes `OUT=/kaggle/working/results` to `run_stage2.sh`; Colab notebook moves the repo folder aside first. |
| 10 | One Tent continual run per model. | The same run gave different numbers on two Kaggle runs (up to 1 point), and the two Standard seeds differed by 5 points. | 3 Tent continual runs per model (`REPEATS`, runs named `..._rep2`, `..._rep3`). |
| 11 | Harm and help rates over every episodic condition, clean included (76). | The proposal defines the harm rate over the 75 CIFAR-10-C conditions; counting clean mixed two questions. Found in the pull request review. | `analyze.py` counts only corrupted conditions and reports clean separately (`clean_effect` in `runs.csv`). Episodic harm rates went down by about 0.01; no conclusion changed. |

---

## 5. Results

### 5.1 E1: reproduction of the published numbers

Severity-5 error (%), mean over 15 corruptions, episodic, batch 200.

| Model | Method | Ours | Published (Tent repo) | Difference |
|---|---|---|---|---|
| Standard WRN-28-10 | None | 43.51 | 43.5 | +0.01 |
| | BN-adapt | 20.44 | 20.4 | +0.04 |
| | Tent | 18.58 | 18.6 | −0.02 |
| AugMix WRN-40-2 | None | 18.27 | 18.3 | −0.03 |
| | BN-adapt | 14.49 | 14.5 | −0.01 |
| | Tent | 12.08 | 12.1 | −0.02 |

The pipeline check in the proposal is passed.

### 5.2 Tent learning rate (4 extra corruptions only)

| Learning rate | 1e-4 | 3e-4 | **1e-3** | 3e-3 |
|---|---|---|---|---|
| Mean error (%) | 15.80 | 15.34 | **14.59** | 16.11 |

We keep 1e-3 (8e-5 for batch 16).

### 5.3 E1 public models: gain, calibration, harm, collapse

| Model | Method | Error | ECE (%) | Harm rate | Collapse (max) |
|---|---|---|---|---|---|
| Standard WRN-28-10 | None | 43.51 | 34.00 | | 0.579 |
| | BN-adapt | 20.44 | 13.77 | 0 / 15 | 0.143 |
| | Tent | 18.58 | 13.34 | 0 / 15 | 0.153 |
| AugMix WRN-40-2 | None | 18.27 | 10.18 | | 0.334 |
| | BN-adapt | 14.49 | 4.87 | **3 / 15** | 0.144 |
| | Tent | 12.08 | 5.41 | 0 / 15 | 0.148 |

BN-adapt harms AugMix on JPEG (+2.79), elastic (+1.60) and glass blur (+1.26).

### 5.4 E2 + E3 + E4 on our ResNet-18, Standard training (seeds 1 and 2)

All numbers are errors in %. "Corr." is the mean over all 75 conditions (episodic only). Harm rate: share of conditions where the method is worse than no adaptation across the whole 95% paired bootstrap interval (the 75 corrupted conditions for episodic, 15 for the severity-5 streams; clean is reported on its own). Collapse: largest share of one class in any window of 1,000 predictions (0.1 is balanced). Seed 1 / seed 2.

| Protocol | Method | Clean | Corr. | Sev. 5 | ECE | Harm rate | Collapse |
|---|---|---|---|---|---|---|---|
| Episodic, batch 200 | None | 4.92 / 5.03 | 25.90 / 25.87 | 43.64 / 43.84 | 17.90 / 17.22 | | 0.851 / 0.856 |
| | BN-adapt | 6.69 / 6.86 | 15.82 / 15.88 | 22.01 / 22.14 | 8.91 / 9.11 | 0.25 / 0.23 | 0.145 / 0.145 |
| | Tent | 5.93 / 5.69 | 13.27 / 13.60 | 18.39 / 18.84 | 8.51 / 8.96 | 0.09 / 0.07 | 0.173 / 0.176 |
| Continual, batch 200 | BN-adapt | | | 22.01 / 22.14 | 12.91 / 13.19 | 1/15 / 1/15 | 0.145 / 0.141 |
| | Tent | | | **18.38 / 23.38** | 14.88 / **20.09** | 1/15 / **3/15** | 0.174 / **0.280** |
| Small batch 16 | BN-adapt | | | 24.80 / 24.74 | 14.52 / 14.70 | 3/15 / 3/15 | 0.141 / 0.140 |
| | Tent | | | 23.06 / 23.16 | 14.44 / 14.75 | 3/15 / 3/15 | 0.155 / 0.161 |
| Episodic sev. 5, 3 shuffle orders | Tent | | | 18.08–18.31 / 18.72–18.98 | 11.9–12.1 / 12.6–12.8 | 0 / 0 | ≤ 0.174 |

(Episodic ECE is over all 75 conditions, so it is lower than the severity-5-only ECE of the other rows.)

**Where adaptation harms (episodic, conditions harmed for both seeds):**
- BN-adapt: clean, brightness 1–4, JPEG 1–4, fog 1–2, defocus 1–2, contrast 1, elastic 1, pixelate 1, snow 1. Largest: +2.86 points.
- Tent: clean, brightness 1–2, contrast 1, defocus 1, fog 1. Largest: +1.16 points.
- Severity-5 streams (continual and small batch): brightness, elastic and JPEG, the three corruptions the model already handles well. Small batch harms both methods on all three (largest +4.91 for BN-adapt); Tent continual harms brightness by up to +7.4 points.

**Tent continual, per corruption (sev. 5, seed 1 → seed 2):** the two seeds start the same (Gaussian noise 26.7 vs 26.9) and drift apart along the stream: contrast 12.9 → 22.2, pixelate 16.7 → 25.2, JPEG 21.2 → 32.1. Seed 2's error grows while the stream goes on, the sign of accumulating errors.

**Run-to-run variation:** Kaggle Version 1 and Version 2 ran the same code on the same checkpoints. Every episodic, BN-adapt and small-batch number agrees to within 0.02 points. Tent continual does not: 18.91 → 18.38 (seed 1) and 24.37 → 23.38 (seed 2). Small GPU rounding differences (cuDNN kernels are not deterministic) grow over a long stream with no reset.

**Cost on a Kaggle T4 (ResNet-18, ms per batch of 200):** none 93, BN-adapt 96, Tent 201. Batch 16: BN-adapt 6.4, Tent 13.4. One model's full Stage 2 set takes about 40 minutes.

Files: `results/kaggle_standard/tables/runs.csv`, `tables/conditions.csv`, `summary.csv`, `run_stage2.log`, help/harm maps `figures/map_*.png` and stream plots `figures/stream_*.png`. Per-image predictions (`preds/`, 125 MB) are kept locally, out of git.

### 5.5 All 6 models (E2–E5)

Our ResNet-18, one row per Stage 1 model. Errors in %. Tent continual is the mean of 3 runs (range in brackets).

**Severity-5 error**

| Model | None | BN episodic | Tent episodic | BN small batch | Tent small batch | BN continual | Tent continual (3 runs) |
|---|---|---|---|---|---|---|---|
| Standard s1 | 43.64 | 22.01 | 18.39 | 24.80 | 23.06 | 22.01 | 19.33 (18.4–20.4) |
| Standard s2 | 43.84 | 22.14 | 18.84 | 24.74 | 23.16 | 22.14 | **25.97 (23.4–28.7)** |
| CutMix s1 | 46.07 | 22.21 | 17.48 | 24.95 | 22.61 | 22.21 | **24.18 (21.3–25.7)** |
| AugMix no JSD s1 | 21.89 | 16.27 | 13.66 | 18.81 | 17.20 | 16.27 | 16.11 (15.2–18.0) |
| AugMix s1 | 17.39 | 13.89 | 11.74 | 16.10 | 14.80 | 13.89 | 13.45 (13.0–14.1) |
| AugMix s2 | 17.89 | 13.60 | 11.62 | 15.90 | 14.54 | 13.60 | 12.85 (12.5–13.4) |

**Episodic: clean, and the 75 corrupted conditions** (BN-adapt and Tent are significantly worse than no adaptation on clean data for every model)

| Model | Clean: none / BN / Tent | Corr. error: none / BN / Tent | ECE: none / BN / Tent | Harm rate: BN / Tent |
|---|---|---|---|---|
| Standard s1 | 4.92 / 6.69 / 5.93 | 25.90 / 15.82 / 13.27 | 17.9 / 8.9 / 8.5 | 0.25 / 0.09 |
| Standard s2 | 5.03 / 6.86 / 5.69 | 25.87 / 15.88 / 13.60 | 17.2 / 9.1 / 9.0 | 0.23 / 0.07 |
| CutMix s1 | 3.82 / 5.04 / 4.70 | 28.50 / 15.33 / 12.22 | 14.9 / 3.5 / 6.5 | **0.19 / 0.05** |
| AugMix no JSD s1 | 4.69 / 6.41 / 5.45 | 13.19 / 11.95 / 10.19 | 6.2 / 5.6 / 5.7 | 0.49 / 0.24 |
| AugMix s1 | 4.79 / 6.33 / 5.42 | 10.84 / 10.52 / 9.05 | 5.1 / 3.5 / 4.4 | **0.63 / 0.27** |
| AugMix s2 | 4.72 / 6.12 / 5.32 | 11.17 / 10.37 / 8.97 | 5.0 / 3.3 / 4.2 | 0.53 / 0.21 |

**Severity-5 streams: harm rate (of 15) and collapse (max)**

| Model | BN small batch | Tent small batch | Tent continual harm (mean of 3) | Tent continual collapse (3 runs) |
|---|---|---|---|---|
| Standard s1 | 0.20 | 0.20 | 0.07 | 0.17 / 0.23 / 0.20 |
| Standard s2 | 0.20 | 0.20 | 0.20 | 0.28 / 0.35 / **0.40** |
| CutMix s1 | 0.07 | 0.07 | 0.20 | 0.33 / **0.40** / 0.28 |
| AugMix no JSD s1 | 0.47 | 0.33 | 0.36 | 0.20 / 0.32 / 0.20 |
| AugMix s1 | 0.47 | 0.40 | 0.36 | 0.20 / 0.20 / 0.23 |
| AugMix s2 | 0.47 | 0.40 | 0.33 | 0.18 / 0.18 / 0.17 |

ECE of Tent continual (mean of 3): Standard 15.8 / 22.7, CutMix 20.9, AugMix no JSD 13.0, AugMix 10.5 / 9.8. Shuffle orders change Tent's severity-5 error by at most 0.25 points for every model.

**Where BN-adapt harms AugMix (episodic, both seeds):** clean and the low severities of almost every corruption (brightness, fog, snow, defocus, motion and zoom blur 1–4, contrast and pixelate 1–3), plus JPEG and elastic at all 5 severities. Largest +2.42 points. Tent harms the same kind of conditions but fewer of them (largest +1.15).

Files: `results/all_models/` (`tables/runs.csv`, `tables/conditions.csv`, `summary.csv`, `figures/`, `logs/`). Per-image predictions (`preds/`, about 400 MB) are kept locally, out of git.

### 5.6 RQ3: how much Tent adds after BN-adapt, per model

Episodic protocol, batch 200. "BN gain" = error(none) − error(BN-adapt); "Tent extra" = error(BN-adapt) − error(Tent); both in points. "BN share" = BN gain / total gain. Tent vs. BN-adapt counts the conditions where the 95% paired bootstrap interval of the difference is entirely on one side. Made by `scripts/rq3_table.py` → `results/all_models/tables/rq3.csv`.

| Model | None (75) | BN gain | Tent extra | BN share | None (sev. 5) | BN gain | Tent extra | BN share | Tent better / worse than BN (of 75) |
|---|---|---|---|---|---|---|---|---|---|
| Standard s1 | 25.90 | 10.08 | 2.55 | 80% | 43.64 | 21.63 | 3.62 | 86% | 74 / 0 |
| Standard s2 | 25.87 | 9.99 | 2.28 | 81% | 43.84 | 21.70 | 3.31 | 87% | 73 / 0 |
| CutMix s1 | 28.50 | 13.17 | 3.11 | 81% | 46.07 | 23.86 | 4.73 | 83% | 72 / 0 |
| AugMix no JSD s1 | 13.19 | 1.24 | 1.76 | 41% | 21.89 | 5.62 | 2.62 | 68% | 72 / 0 |
| AugMix s1 | 10.84 | 0.32 | 1.47 | 18% | 17.38 | 3.49 | 2.16 | 62% | 75 / 0 |
| AugMix s2 | 11.17 | 0.80 | 1.40 | 36% | 17.89 | 4.29 | 1.98 | 68% | 75 / 0 |

**Tent extra by corruption family (all severities, points):**

| Model | Noise | Blur | Weather | Digital |
|---|---|---|---|---|
| Standard s1 / s2 | 5.16 / 4.74 | 1.71 / 1.68 | 1.80 / 1.53 | 2.16 / 1.80 |
| CutMix s1 | 6.17 | 1.84 | 1.83 | 3.36 |
| AugMix no JSD s1 | 2.80 | 1.49 | 1.50 | 1.50 |
| AugMix s1 / s2 | 1.87 / 1.71 | 1.39 / 1.31 | 1.32 / 1.22 | 1.40 / 1.42 |

**At severity 1** BN-adapt is worse than no adaptation for every AugMix model (−0.99 to −1.32 points) and Tent more than makes up for it (+0.98 to +1.23). For Standard and CutMix, BN-adapt still helps at severity 1 (+1.2 to +3.3).

**Clean test set (error %, none / BN-adapt / Tent):** Standard 4.92 / 6.69 / 5.93 and 5.03 / 6.86 / 5.69, CutMix 3.82 / 5.04 / 4.70, AugMix no JSD 4.69 / 6.41 / 5.45, AugMix 4.79 / 6.33 / 5.42 and 4.72 / 6.12 / 5.32. Tent is significantly better than BN-adapt on clean data for every model, but still worse than no adaptation.

**Answer to RQ3:**
1. **Tent always adds something on top of BN-adapt.** For every model it is significantly better on 72–75 of the 75 conditions and never significantly worse, and it is better on clean data too.
2. **How much: 1.4–3.1 points over all 75 conditions, 2.0–4.7 at severity 5.** That fits H3's "about 2 points": about 2.4 for Standard, 3.1 for CutMix, 1.4–1.8 for AugMix. The public WideResNets in E1 gave 1.9 (Standard) and 2.4 (AugMix) at severity 5.
3. **"Most of the gain comes from BN-adapt" holds only for models without corruption-like augmentation.** For Standard and CutMix, BN-adapt gives 80–87% of the total gain. For AugMix, BN-adapt gives only 18–41% over all severities, because it hurts at low severities, and 62–68% at severity 5. On AugMix, Tent's extra gain is larger than BN-adapt's own gain over all severities.
4. **The total gain is much smaller for AugMix, as H3 predicted:** 1.8–3.0 points over all conditions (5.7–8.2 at severity 5) against 12.3–16.3 (25.0–28.6) for Standard and CutMix. AugMix is still the most accurate after adaptation (Tent: 8.97–9.05, against 12.22–13.60).
5. **Where Tent adds most: noise.** Tent's extra gain on noise is 4.7–6.2 points for Standard and CutMix, 2–3 times its gain on the other families. AugMix already handles noise better, so Tent adds only 1.7–2.8 there.
6. **JSD matters a little here too:** AugMix without JSD gets more from adaptation (3.0 points over all conditions) than AugMix with JSD (1.8–2.2), because it starts worse (13.2 vs 10.8–11.2), but it still ends worse after Tent (10.2 vs 9.0).

---

## 6. What the results tell us

Points 1–6 come from the Standard models; point 7 compares all 6 models. RQ3 for all 6 models is answered in section 5.6.

1. **H3 (most of the gain is BN-adapt; Tent adds about 2 points): supported on our ResNet-18.** At severity 5, BN-adapt takes error from 43.7 to 22.1 (−21.6); Tent adds −3.5 (18.6). The same pattern as the public WRN-28-10 in E1. For AugMix models the first half of H3 does not hold (section 5.6).
2. **Adaptation is not free, even episodically.** BN-adapt is worse than no adaptation on about a quarter of the 75 corrupted conditions and on clean data (+1.8 points); Tent on less than a tenth, and on clean data too. The harm sits on clean data, mild severities and the easy corruptions (brightness, JPEG, fog), where the model was already good. This matches the AugMix WRN result in E1: adaptation hurts where the source model is already right.
3. **RQ4, the difficult streams:**
   - **Small batch hurts both methods** at severity 5: BN-adapt 22.1 → 24.8 (+2.7), Tent 18.6 → 23.1 (+4.5). With batch 16, Tent keeps only 1.7 points of its 3.5-point lead over BN-adapt.
   - **Continual Tent is unstable.** For seed 1 it matches episodic Tent (18.4). For seed 2 it is worse than plain BN-adapt (23.4 vs 22.1), with the highest ECE (20.1) and the highest collapse score (0.28) of any adapted run. Same training recipe, different seed, very different result: the continual stream is where Tent can fail.
   - **BN-adapt has no state**, so its continual and episodic severity-5 errors are identical (22.01 / 22.14), a useful sanity check.
4. **Calibration (RQ5):** adaptation halves ECE episodically (17.6 → 8.7–9.0), with Tent slightly better than BN-adapt here. In the hard streams Tent loses that edge: continual Tent seed 2 has ECE 20.1, worse than no adaptation's ECE over all conditions.
5. **Order inside a condition does not matter much:** 3 shuffle orders change Tent's severity-5 error by at most 0.25 points, and none is harmful.
6. **For the next runs:** continual Tent needs repeated runs (at least 3 per model), because a single run moves by up to 1 point between GPU runs and by 5 points between seeds. (Done for all 6 models: the 3 runs differ by up to 5 points.)
7. **All 6 models: does augmentation make adaptation safer (H4)? It depends on what "safer" means.**
   - **Lowest error and most stable: AugMix (with JSD).** It has the lowest error in every protocol, and continual Tent works on it: all 3 runs are better than BN-adapt (12.5–14.1 vs 13.6–13.9), with low collapse (≤ 0.23) and the lowest ECE. Without the JSD loss AugMix is worse (16.1) and one of the 3 runs collapses more (0.32) and is 2.8 points worse.
   - **Most often harmed: also AugMix.** BN-adapt is worse than no adaptation on 49–63% of AugMix conditions, against 19–25% for Standard and CutMix, and Tent on 21–27% against 5–9%. The harm-rate tables say the opposite of H4: the better the model is without adaptation, the more often adaptation hurts it (clean and mild corruptions, where the source BN statistics are already right). The harm is small, at most +2.4 points.
   - **CutMix does not protect the hard streams.** It has the best clean error (3.82) but the worst corruption error without adaptation (46.1 at severity 5). It gains the most from adaptation and is rarely harmed in episodic tests, but continual Tent is as unstable as for Standard seed 2: worse than BN-adapt in 2 of 3 runs, collapse up to 0.40, ECE 20.9.
   - **Small batches hurt every model** by 2–3 points for BN-adapt and 2–4 for Tent, and harm 47% of the severity-5 conditions for AugMix.
   - **Summary for H4:** augmentation that covers the corruptions (AugMix + JSD) makes adaptation *stable* (no collapse, no long-stream failure), but not *harmless*: adaptation hurts AugMix more often, in small amounts, because there is less to gain. Mixing images (CutMix) gives neither.

---

## 7. Progress against the plan (proposal Table 7, Member 2)

| Week | Planned (Member 2) | Status |
|---|---|---|
| 1 (Oct 12) | CIFAR-10-C, `eval.py`, reproduce Standard WRN-28-10 | ✅ Done |
| 2 (Oct 19) | Reproduce AugMix WRN-40-2, continual and small-batch streams | ✅ Done |
| 3 (Oct 26) | Tent learning rate, episodic tests on finished models | ✅ Done (Standard seeds 1–2) |
| 4 (Nov 2) | Episodic tests on all models, ECE from saved predictions | ✅ Done (all 6 models) |
| 5 (Nov 9) | Continual stream on all models | ✅ Done (3 runs per model) |
| 6 (Nov 16) | Small-batch stream, harm rate, collapse; optional E6 | ✅ Done; ⏳ optional E6 not run |
| 7 to 8 | Together: figures, report, slides, code clean-up | Not started |

---

## 8. What is next

### Yash (Stage 2)
1. **Optional E6:** periodic reset (`--reset-every`) on continual Tent for Standard s2 and CutMix, the two models where it drifts. About 15 minutes per run on Kaggle.
2. **Figures for the final report:** one figure with the 6 help/harm maps side by side, and one with the continual-stream plots of the 3 Tent runs per model.
3. Get pull request #3 reviewed and merged by Kirtan.

### Together (weeks 7–8)
- Final report: Stage 1 training cost (Kirtan's logs, E2) next to the Stage 2 tables of section 5.5.
- Slides and code clean-up.

---

## 9. How to reproduce

```bash
# Mac or any machine, no download
python -m pytest -q tests
python scripts/smoke_test.py

# All Stage 2 runs for some checkpoints (GPU; Kaggle: notebooks/stage2_kaggle.ipynb)
CIFAR10C_HF_REPO=YashP54/cifar10-c bash scripts/download_cifar10c.sh
OUT=results bash scripts/run_stage2.sh runs/*/final.pt

# Tables and figures from saved predictions (CPU)
python analyze.py --results results/kaggle_standard
```

E1 and learning-rate tables: `results/e1_colab/`. Standard ResNet-18 results (first Kaggle run): `results/kaggle_standard/`. All 6 models: `results/all_models/`. Per-image predictions are kept out of git and can be regenerated with the commands above.
