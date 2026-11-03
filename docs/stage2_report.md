# Stage 2 progress report (Member 2: Yash Patel)

**Project:** Does Augmentation Make Test-Time Adaptation Safer? (CMPE 252, Fall 2026)
**Branch:** `member2/stage2-adaptation-eval` (pull request #3 into `main`)
**Last updated:** November 2, 2026 (start of week 4 of 8)

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
| **E2 + E3:** none / BN-adapt / Tent, 75 conditions + clean | ✅ Standard seeds 1–2. ⏳ 4 more models (waiting for Kirtan's `final.pt` files) |
| **E4:** continual and small-batch streams (RQ4) | ✅ Standard seeds 1–2. ⏳ 4 more models |
| **E5:** help/harm maps and calibration (RQ5) | ✅ Maps for every Standard run; ⏳ comparison across augmentations |
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

All numbers are errors in %. "Corr." is the mean over all 75 conditions (episodic only). Harm rate: share of conditions where the method is worse than no adaptation across the whole 95% paired bootstrap interval (76 conditions with clean for episodic, 15 for the severity-5 streams). Collapse: largest share of one class in any window of 1,000 predictions (0.1 is balanced). Seed 1 / seed 2.

| Protocol | Method | Clean | Corr. | Sev. 5 | ECE | Harm rate | Collapse |
|---|---|---|---|---|---|---|---|
| Episodic, batch 200 | None | 4.92 / 5.03 | 25.90 / 25.87 | 43.64 / 43.84 | 17.90 / 17.22 | | 0.851 / 0.856 |
| | BN-adapt | 6.69 / 6.86 | 15.82 / 15.88 | 22.01 / 22.14 | 8.91 / 9.11 | 0.26 / 0.24 | 0.145 / 0.145 |
| | Tent | 5.93 / 5.69 | 13.27 / 13.60 | 18.39 / 18.84 | 8.51 / 8.96 | 0.11 / 0.08 | 0.173 / 0.176 |
| Continual, batch 200 | BN-adapt | | | 22.01 / 22.14 | 12.91 / 13.19 | 1/15 / 1/15 | 0.145 / 0.141 |
| | Tent | | | **18.38 / 23.38** | 14.88 / **20.09** | 1/15 / **3/15** | 0.174 / **0.280** |
| Small batch 16 | BN-adapt | | | 24.80 / 24.74 | 14.52 / 14.70 | 3/15 / 3/15 | 0.141 / 0.140 |
| | Tent | | | 23.06 / 23.16 | 14.44 / 14.75 | 3/15 / 3/15 | 0.155 / 0.161 |
| Episodic sev. 5, 3 shuffle orders | Tent | | | 18.08–18.31 / 18.72–18.98 | 11.9–12.1 / 12.6–12.8 | 0 / 0 | ≤ 0.174 |

(Episodic ECE is over 76 conditions, so it is lower than the severity-5-only ECE of the other rows.)

**Where adaptation harms (episodic, conditions harmed for both seeds):**
- BN-adapt: clean, brightness 1–4, JPEG 1–4, fog 1–2, defocus 1–2, contrast 1, elastic 1, pixelate 1, snow 1. Largest: +2.86 points.
- Tent: clean, brightness 1–2, contrast 1, defocus 1, fog 1. Largest: +1.16 points.
- Severity-5 streams (continual and small batch): brightness, elastic and JPEG, the three corruptions the model already handles well. Small batch harms both methods on all three (largest +4.91 for BN-adapt); Tent continual harms brightness by up to +7.4 points.

**Tent continual, per corruption (sev. 5, seed 1 → seed 2):** the two seeds start the same (Gaussian noise 26.7 vs 26.9) and drift apart along the stream: contrast 12.9 → 22.2, pixelate 16.7 → 25.2, JPEG 21.2 → 32.1. Seed 2's error grows while the stream goes on, the sign of accumulating errors.

**Run-to-run variation:** Kaggle Version 1 and Version 2 ran the same code on the same checkpoints. Every episodic, BN-adapt and small-batch number agrees to within 0.02 points. Tent continual does not: 18.91 → 18.38 (seed 1) and 24.37 → 23.38 (seed 2). Small GPU rounding differences (cuDNN kernels are not deterministic) grow over a long stream with no reset.

**Cost on a Kaggle T4 (ResNet-18, ms per batch of 200):** none 93, BN-adapt 96, Tent 201. Batch 16: BN-adapt 6.4, Tent 13.4. One model's full Stage 2 set takes about 40 minutes.

Files: `results/kaggle_standard/tables/runs.csv`, `tables/conditions.csv`, `summary.csv`, `run_stage2.log`, help/harm maps `figures/map_*.png` and stream plots `figures/stream_*.png`. Per-image predictions (`preds/`, 125 MB) are kept locally, out of git.

---

## 6. What the results tell us so far

Only the Standard models are done, so these are answers for the baseline. The augmentation comparison needs the other 4 models.

1. **H3 (most of the gain is BN-adapt; Tent adds about 2 points): supported on our ResNet-18.** At severity 5, BN-adapt takes error from 43.7 to 22.1 (−21.6); Tent adds −3.5 (18.6). The same pattern as the public WRN-28-10 in E1.
2. **Adaptation is not free, even episodically.** BN-adapt is worse than no adaptation on about a quarter of all conditions, including clean data (+1.8 points), and Tent on about a tenth. The harm sits on clean data, mild severities and the easy corruptions (brightness, JPEG, fog), where the model was already good. This matches the AugMix WRN result in E1: adaptation hurts where the source model is already right.
3. **RQ4, the difficult streams:**
   - **Small batch hurts both methods** at severity 5: BN-adapt 22.1 → 24.8 (+2.7), Tent 18.6 → 23.1 (+4.5). With batch 16, Tent keeps only 1.7 points of its 3.5-point lead over BN-adapt.
   - **Continual Tent is unstable.** For seed 1 it matches episodic Tent (18.4). For seed 2 it is worse than plain BN-adapt (23.4 vs 22.1), with the highest ECE (20.1) and the highest collapse score (0.28) of any adapted run. Same training recipe, different seed, very different result: the continual stream is where Tent can fail.
   - **BN-adapt has no state**, so its continual and episodic severity-5 errors are identical (22.01 / 22.14), a useful sanity check.
4. **Calibration (RQ5):** adaptation halves ECE episodically (17.6 → 8.7–9.0), with Tent slightly better than BN-adapt here. In the hard streams Tent loses that edge: continual Tent seed 2 has ECE 20.1, worse than no adaptation's ECE over all conditions.
5. **Order inside a condition does not matter much:** 3 shuffle orders change Tent's severity-5 error by at most 0.25 points, and none is harmful.
6. **For the next runs:** continual Tent needs repeated runs (at least 3 per model), because a single run moves by up to 1 point between GPU runs and by 5 points between seeds.

---

## 7. Progress against the plan (proposal Table 7, Member 2)

| Week | Planned (Member 2) | Status |
|---|---|---|
| 1 (Oct 12) | CIFAR-10-C, `eval.py`, reproduce Standard WRN-28-10 | ✅ Done |
| 2 (Oct 19) | Reproduce AugMix WRN-40-2, continual and small-batch streams | ✅ Done |
| 3 (Oct 26) | Tent learning rate, episodic tests on finished models | ✅ Done (Standard seeds 1–2) |
| 4 (Nov 2) | Episodic tests on all models, ECE from saved predictions | ✅ Standard done; ⏳ 4 models waiting for checkpoints |
| 5 (Nov 9) | Continual stream on all models | ✅ Standard done; ⏳ 4 models |
| 6 (Nov 16) | Small-batch stream, harm rate, collapse; optional E6 | ✅ Standard done; ⏳ 4 models, E6 |
| 7 to 8 | Together: figures, report, slides, code clean-up | Not started |

---

## 8. What is next

### Yash (Stage 2)
1. **When Kirtan uploads the 4 remaining models** (`augmix_seed1`, `augmix_seed2`, `augmix_nojsd_seed1`, `cutmix_seed1`, each `runs/<name>/final.pt`): add them to the Kaggle dataset `cmpe252-stage1-models` as a new version, and run the Kaggle notebook only on those 4 (about 2 h 40 min on a T4, within the 30 h/week quota).
2. **Repeat Tent continual** 3 times per model (finding 6).
3. **E5:** put the help/harm maps of all 6 models side by side, and compare harm rate, ECE and collapse across augmentations. This answers H4: does augmentation make adaptation safer?
4. **Optional E6:** periodic reset (`--reset-every`) on continual Tent, which should fix the seed-2 drift.
5. Get pull request #3 reviewed and merged by Kirtan.

### Kirtan (Stage 1): what Stage 2 needs
- The 4 remaining `final.pt` files. `augmix_seed1` has only `last.pt` on Drive so far.
- Same format as the Standard ones, which work: `{"model": state_dict, ...}` from `model.py`, no normalization in the data loader.
- No CIFAR-10-C needed for training.

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

E1 and learning-rate tables: `results/e1_colab/`. Standard ResNet-18 results: `results/kaggle_standard/`. Per-image predictions are kept out of git and can be regenerated with the commands above.
