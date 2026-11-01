# Stage 2 progress report (Member 2: Yash Patel)

**Project:** Does Augmentation Make Test-Time Adaptation Safer? (CMPE 252, Fall 2026)
**Branch:** `member2/stage2-adaptation-eval`
**Last updated:** November 1, 2026 (end of week 3 of 8)

---

## 1. Status at a glance

| Item | Status |
|---|---|
| Stage 2 code: BN-adapt, Tent, 3 test streams, metrics, eval and analysis scripts | ✅ Done, 14 unit tests pass (Mac and Colab) |
| CIFAR-10-C download and hosting | ✅ Done, mirrored on Hugging Face: [`YashP54/cifar10-c`](https://huggingface.co/datasets/YashP54/cifar10-c) |
| Colab notebook for the GPU runs | ✅ Done, [`notebooks/stage2_colab.ipynb`](../notebooks/stage2_colab.ipynb) |
| **E1:** reproduce the published RobustBench + Tent numbers | ✅ Done, all 6 numbers within 0.05 points |
| Tent learning rate chosen on the 4 extra corruptions | ✅ Done, 1e-3 (the official value) |
| **E2 + E3:** no adaptation / BN-adapt / Tent on our ResNet-18 models | ⏳ Waiting for Stage 1 checkpoints (Kirtan) |
| **E4:** continual and small-batch streams (main experiment, RQ4) | ⏳ Code ready; waiting for checkpoints |
| **E5:** help/harm maps and calibration (RQ5) | ⏳ Code ready; first results from E1 below |
| **E6:** optional extensions | Partly ready: periodic reset (`--reset-every`) is implemented |

Member 2 is on schedule. Every Member 2 task for weeks 1 to 3 in the proposal (Table 7) is done, except "run episodic tests on finished models", which needs the first trained model.

---

## 2. What we built

All Stage 2 code is in this repository:

| File | What it does |
|---|---|
| `tta/data.py` | Loads CIFAR-10 and CIFAR-10-C. Images are float tensors in [0, 1]. |
| `tta/models.py` | CIFAR ResNet-18 (normalizes inside the model), Stage 1 checkpoint loader, RobustBench models for E1. |
| `tta/methods.py` | No adaptation, BN-adapt and Tent. Tent is adapted from the official code (DequanWang/tent, MIT license). Each method predicts on a batch before it updates on that batch. |
| `tta/streams.py` | Episodic (75 conditions, batch 200, reset each condition), continual (severity 5, no reset), small batch (severity 5, batch 16, Tent learning rate × 16/200). Optional shuffle seed and periodic reset. |
| `tta/metrics.py` | Error, ECE (15 bins), paired bootstrap for the harm rate, collapse score (sliding window of 1,000). |
| `eval.py` | Runs one model × method × protocol, saves every prediction (`.npz`) and appends to `summary.csv`. |
| `analyze.py` | CPU only. Builds `runs.csv` and `conditions.csv`, the 75-condition help/harm maps and the continual-stream plots. |
| `tests/test_stage2.py` | 14 unit tests: Tent updates only BN scale and shift, predicts before updating, reset works, streams, metrics, checkpoint loading. |
| `scripts/smoke_test.py` | End-to-end check on fake data, no download needed. |
| `scripts/download_cifar10c.sh` | Gets CIFAR-10-C from Hugging Face, with Zenodo as the fallback. |
| `scripts/upload_cifar10c_to_hf.py` | One-time upload of the Hugging Face mirror, with an attribution page for the dataset. |
| `notebooks/stage2_colab.ipynb` | The full Stage 2 run on a free Colab T4. |

---

## 3. What we did, in order

1. **Read the proposal** and split out Member 2's part: Stage 2, adaptation and evaluation.
2. **Cloned the team repo** and made the branch `member2/stage2-adaptation-eval`.
3. **Wrote the Stage 2 code** (section 2) and the unit tests. All 14 tests passed on the Mac.
4. **Smoke trial on the Mac (Apple M1, MPS):** this found a real bug (section 4, change 1), which we fixed. After that, every method and stream ran end to end.
5. **Pushed the branch** and opened the notebook in Colab on a T4 GPU.
6. **Downloaded CIFAR-10-C** in Colab from Zenodo (2.78 GB, about 14 minutes). All 14 tests passed on Colab too.
7. **Mirrored CIFAR-10-C to Hugging Face**, so later sessions skip Zenodo.
8. **Ran E1** (6 runs) and the **Tent learning-rate check** (4 runs). Downloaded the results and saved the tables in `results/e1_colab/`.

---

## 4. What we changed along the way, and why

| # | First plan | Problem | Change |
|---|---|---|---|
| 1 | Images converted with a plain `permute`. | Tent's backward pass crashed on Apple MPS: `view size is not compatible with input tensor's size and stride`. The smoke trial found it. | `to_tensor` in `tta/data.py` now calls `.contiguous()`. All runs work on MPS and CUDA. |
| 2 | Keep CIFAR-10-C and results in Google Drive, as the proposal suggested. | Mounting Drive in Colab failed with `credential propagation was unsuccessful`. We also preferred not to give the notebook Drive access. | Drive is now optional (`USE_DRIVE = False`). By default everything is on the Colab disk, and the last cell downloads the results as a zip. |
| 3 | Download CIFAR-10-C from Zenodo each session. | Zenodo ran at about 3 MB/s, so about 14 to 17 minutes per session. Colab disk is cleared after each session. | Mirrored to Hugging Face: [`YashP54/cifar10-c`](https://huggingface.co/datasets/YashP54/cifar10-c). Public, so no login is needed to download, and it gives credit under CC BY 4.0. `download_cifar10c.sh` tries Hugging Face first, then Zenodo. |
| 4 | Hugging Face repo name `yashP54/...`. | The account name is `YashP54`, with a capital Y. | Fixed in the notebook and the README. |
| 5 | E2 and E3 as separate notebook sections. | E2 (training comparison) has no separate evaluation step. Its numbers are the "no adaptation" runs of E3. | The notebook section is now "E2 + E3". Kirtan's training logs give the training cost for E2. |
| 6 | Proposal lists Python 3.11. | Colab now runs Python 3.13. | The code works on both: Python 3.11 on the Mac, 3.13 on Colab. No change needed. |

---

## 5. Results so far

### 5.1 E1: reproduction of the published numbers

Severity-5 error (%), mean over 15 corruptions, episodic protocol, batch 200.

| Model | Method | Ours | Published (Tent repo) | Difference |
|---|---|---|---|---|
| Standard WRN-28-10 | None | 43.51 | 43.5 | +0.01 |
| | BN-adapt | 20.44 | 20.4 | +0.04 |
| | Tent | 18.58 | 18.6 | −0.02 |
| AugMix WRN-40-2 | None | 18.27 | 18.3 | −0.03 |
| | BN-adapt | 14.49 | 14.5 | −0.01 |
| | Tent | 12.08 | 12.1 | −0.02 |

**What this tells us:** data loading, BN handling, Tent and the error metric are correct for both model types. The pipeline check in the proposal ("our numbers must match the published numbers in Table 2") is passed.

### 5.2 Tent learning rate (4 extra corruptions only)

Standard WRN-28-10, severity 5, on speckle noise, Gaussian blur, spatter and saturate. None of these is among the 15 test corruptions.

| Learning rate | Mean error (%) |
|---|---|
| 1e-4 | 15.80 |
| 3e-4 | 15.34 |
| **1e-3** | **14.59** |
| 3e-3 | 16.11 |

**Decision:** keep **1e-3**. For the small-batch stream the code scales it to 1e-3 × 16/200 = 8e-5.

### 5.3 Adaptation gain, calibration, harm and collapse

| Model | Method | Error | Gain over previous | ECE (%) | Harm rate vs. none | Collapse (max) |
|---|---|---|---|---|---|---|
| Standard WRN-28-10 | None | 43.51 | | 34.00 | | 0.579 |
| | BN-adapt | 20.44 | −23.07 | 13.77 | 0 / 15 | 0.143 |
| | Tent | 18.58 | −1.86 | 13.34 | 0 / 15 | 0.153 |
| AugMix WRN-40-2 | None | 18.27 | | 10.18 | | 0.334 |
| | BN-adapt | 14.49 | −3.78 | 4.87 | **3 / 15** | 0.144 |
| | Tent | 12.08 | −2.41 | 5.41 | 0 / 15 | 0.148 |

The harm rate counts a condition as harm only when the error difference is above 0 for the whole 95% paired bootstrap interval over the 10,000 test images. ECE is the mean over the 15 conditions. The collapse score is the largest share of one class in any window of 1,000 predictions; 0.1 is balanced.

**Where BN-adapt harms the AugMix model** (error change against no adaptation, 95% interval):

| Corruption | Change | 95% interval |
|---|---|---|
| JPEG compression | +2.79 | [2.26, 3.41] |
| Elastic transform | +1.60 | [1.03, 2.23] |
| Glass blur | +1.26 | [0.55, 1.96] |

### 5.4 Error per corruption (severity 5, %)

| Corruption | Std none | Std BN | Std Tent | AugMix none | AugMix BN | AugMix Tent |
|---|---|---|---|---|---|---|
| gaussian_noise | 72.3 | 28.1 | 24.8 | 28.8 | 18.5 | 15.6 |
| shot_noise | 65.7 | 26.1 | 23.5 | 23.0 | 16.2 | 13.2 |
| impulse_noise | 72.9 | 36.3 | 33.0 | 26.2 | 22.3 | 18.8 |
| defocus_blur | 46.9 | 12.8 | 12.0 | 9.5 | 8.9 | 7.9 |
| glass_blur | 54.3 | 35.3 | 31.9 | 20.6 | 21.9 | 18.1 |
| motion_blur | 34.8 | 14.2 | 13.7 | 10.6 | 10.5 | 9.0 |
| zoom_blur | 42.0 | 12.1 | 10.8 | 9.3 | 9.7 | 8.0 |
| snow | 25.1 | 17.3 | 15.9 | 14.2 | 12.8 | 10.4 |
| frost | 41.3 | 17.4 | 16.2 | 15.3 | 13.3 | 10.9 |
| fog | 26.0 | 15.3 | 13.7 | 17.5 | 15.0 | 12.3 |
| brightness | 9.3 | 8.4 | 7.9 | 7.6 | 7.6 | 6.7 |
| contrast | 46.7 | 12.6 | 12.1 | 20.9 | 11.9 | 10.0 |
| elastic_transform | 26.6 | 23.8 | 22.0 | 14.7 | 16.3 | 14.0 |
| pixelate | 58.5 | 19.7 | 17.3 | 41.3 | 15.0 | 11.4 |
| jpeg_compression | 30.3 | 27.3 | 24.2 | 14.7 | 17.5 | 14.8 |

### 5.5 Cost on a Colab T4 (ms per batch of 200)

| Model | None | BN-adapt | Tent |
|---|---|---|---|
| Standard WRN-28-10 | 373 | 422 | 881 |
| AugMix WRN-40-2 | 129 | 133 | 248 |

Tent costs about 2× BN-adapt because of the extra backward pass. Our ResNet-18 is much smaller than WRN-28-10, so the E3/E4 runs should be faster than these.

---

## 6. What the results tell us so far

These results use the two public WideResNets at severity 5, not our ResNet-18 models, so they are early hints, not final answers.

1. **H3 (most of the gain is BN-adapt; Tent adds about 2 points; smaller gain for AugMix): supported.** BN-adapt gives −23.1 points on Standard and −3.8 on AugMix. Tent adds −1.9 and −2.4 points. The total gain is 24.9 points on Standard and only 6.2 on AugMix.
2. **H4 (augmentation makes adaptation safer): a first counter-example.** BN-adapt never hurts the Standard model, but it **significantly hurts the AugMix model on 3 of 15 corruptions** (JPEG, elastic, glass blur). Those are corruptions where the AugMix model was already good without adaptation. Tent removes that harm in the episodic setting. The real test is E4 (continual and small batch) on our own models.
3. **H5 (noise and glass blur stay hardest; contrast and fog get much easier with BN-adapt): supported on Standard.** After Tent, the hardest are impulse noise (33.0), glass blur (31.9) and Gaussian noise (24.8). With BN-adapt, contrast drops 46.7 → 12.6 and fog 26.0 → 15.3.
4. **Calibration (RQ5):** adaptation makes Standard much better calibrated (ECE 34.0 → 13.8). On AugMix, Tent's ECE (5.41) is slightly worse than BN-adapt's (4.87) even though its error is lower. That fits the worry that entropy minimization makes a model over-confident. We will check this on our models.
5. **Collapse:** there is no collapse in the episodic protocol with resets (all adapted runs ≈ 0.14 to 0.15). The high score for "none" on Standard (0.58) comes from the unadapted model putting most noisy images into one class, not from adaptation. Collapse matters for the continual stream (E4).

---

## 7. Progress against the plan (proposal Table 7, Member 2)

| Week | Planned (Member 2) | Status |
|---|---|---|
| 1 (Oct 12) | Download CIFAR-10-C, write `eval.py` that saves predictions, reproduce Standard WRN-28-10 | ✅ Done |
| 2 (Oct 19) | Reproduce AugMix WRN-40-2, write the continual and small-batch streams | ✅ Done |
| 3 (Oct 26) | Choose the Tent learning rate on the 4 extra corruptions, run episodic tests on finished models | ✅ Learning rate done; ⏳ episodic tests need checkpoints |
| 4 (Nov 2) | Episodic tests on all models, ECE from saved predictions | ⏳ Next: needs all 6 checkpoints |
| 5 (Nov 9) | Continual stream on all models | Code ready |
| 6 (Nov 16) | Small-batch stream, harm rate, collapse score; optional E6 | Code ready |
| 7 to 8 | Together: figures, report, slides, code clean-up | Not started |

---

## 8. What is next

### Yash (Stage 2)
1. **As soon as the first checkpoint is ready** (even a half-trained one), run the E2 + E3 cell on it to test the handoff format.
2. **Week 4:** E2 + E3 on all 6 models (`--clean`, all 75 conditions, none / BN-adapt / Tent). Make the first results table for the mid-project check-in.
3. **Week 5:** E4 continual stream on all models. Plot accuracy and collapse over the stream (`analyze.py` makes these plots).
4. **Week 6:** E4 small-batch stream, 3 shuffle orders for the severity-5 Tent runs, harm rate and collapse for RQ4. Optional E6: periodic reset (`--reset-every`) and the mixed stream.
5. **Housekeeping:**
   - Open a pull request to merge this branch into `main` after Kirtan reviews it.
   - Delete the Hugging Face write token; downloads do not need it.
   - Generate the help/harm map figures. `analyze.py` makes them when figures are not turned off; the E1 run used `--no-figures`.

### Kirtan (Stage 1): what Stage 2 needs
- **Checkpoint format:** `torch.save({"model": model.state_dict(), ...}, path)` using `tta.models.ResNet18`, or a model with the same layer names.
- **No normalization in the data loader.** The model normalizes images itself and takes inputs in [0, 1]. Use only random crop, flip and `ToTensor`.
- **File names:** `standard_s1.pt`, `standard_s2.pt`, `cutmix_s1.pt`, `augmix_nojsd_s1.pt`, `augmix_jsd_s1.pt`, `augmix_jsd_s2.pt`.
- **For E2:** record the training time per epoch for each setting.
- **No CIFAR-10-C needed:** training uses clean CIFAR-10 only. If it is ever needed, it is at [`YashP54/cifar10-c`](https://huggingface.co/datasets/YashP54/cifar10-c).

---

## 9. How to reproduce

```bash
# Mac or any machine, no download
python -m pytest -q tests
python scripts/smoke_test.py

# GPU (Colab: open notebooks/stage2_colab.ipynb, Runtime → T4 GPU, run the cells)
CIFAR10C_HF_REPO=YashP54/cifar10-c bash scripts/download_cifar10c.sh
python eval.py --model rb_standard_wrn28 --method tent --severities 5
python analyze.py
```

The E1 and learning-rate tables in this report are in `results/e1_colab/` (`summary.csv`, `e1_table.txt`, `tune_table.txt`, `tables/`). The raw per-image predictions (`preds/`, about 20 MB) are kept locally and out of git. They can be regenerated with the commands above in about 40 minutes on a T4.
