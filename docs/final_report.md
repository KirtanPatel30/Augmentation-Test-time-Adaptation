# Does Augmentation Make Test-Time Adaptation Safer?

**Authors:** Kirtan Patel and Yash Patel
**Course:** CMPE 252, Fall 2026
**Status:** Draft. Sections marked **[Yash]** are still to be filled in.

---

## Abstract

Image models do well on clean test pictures but make many more mistakes on damaged pictures, such as noisy, blurry or foggy ones. Two ways to fix this are training with augmentation and adapting the model at test time. We study how these two work together. We train a ResNet-18 on CIFAR-10 with Standard training, CutMix, AugMix without JSD and AugMix with JSD, and test each model on CIFAR-10-C with no adaptation, BN-adapt and Tent. AugMix + JSD has the lowest error, 11.01% without adaptation and 9.01% with Tent. But adaptation hurts AugMix models more often: in the episodic test, BN-adapt makes 49–63% of the conditions worse for AugMix models and 19–25% for Standard and CutMix. In a continual stream, Tent drifts on Standard and CutMix, and resetting it every 100 batches fixes this. So whether augmentation makes adaptation "safer" depends on what safer means.

## 1 Introduction

Deep image models such as ResNets [1] get more than 95% accuracy on CIFAR-10 [18]. But this only holds when the test pictures look like the training pictures. In real use, this is often not true. A car camera sees rain and fog, a scanner adds noise, and a web picture gets compressed. A person can still recognize these pictures, but a model can make many more mistakes [2].

CIFAR-10-C [2] measures this problem. It applies 15 kinds of damage at 5 severity levels to the CIFAR-10 test pictures, which gives 75 test conditions. The main rule is that a model must not train on these same kinds of damage.

There are two main ways to make a model stronger:

1. **Training-time defenses** change the training pictures. Examples are CutMix [4] and AugMix [5].
2. **Test-time defenses** change the model during use, with unlabeled test pictures. Examples are BN-adapt [6] and Tent [7].

Earlier work already shows that test-time adaptation still helps an augmentation-trained model, but the gain is smaller [6, 8]. The open question is about safety. Test-time adaptation can make a model worse. It can fail when the pictures keep changing [9, 10] or when batches are small [8, 11]. It can also hurt in some CIFAR-10-C conditions [12]. But these failure studies mostly use standard-trained models.

**Our question is: if a model was trained with augmentation, is test-time adaptation safer?**

To answer this, we train a ResNet-18 in 4 ways: Standard, CutMix, AugMix without its JSD loss, and AugMix with JSD. We then test all 6 trained models with no adaptation, BN-adapt and Tent. We use the standard episodic test and two harder tests: a continual stream and small batches. We measure error, harm rate, collapse and calibration.

**Our contributions are:**

- A fair comparison of 4 training methods, with one network, one set of training settings and one evaluation code.
- A check on ResNet-18 of how much Tent adds after BN-adapt for each model.
- A test of whether augmentation makes adaptation safer in continual and small-batch streams.
- A map of where adaptation helps or harms each model, and how it changes calibration.

**Our main finding is that the answer depends on what "safer" means.** AugMix + JSD has the lowest error, and Tent stays stable on it in the continual stream. But adaptation hurts AugMix models more often, usually by small amounts, because they are already good on clean pictures and mild damage. In the episodic test, BN-adapt harms 49–63% of the conditions for AugMix models and 19–25% for Standard and CutMix. CutMix has the best clean accuracy, but it does not protect the model in the continual stream.

## 2 Related Work

### 2.1 Benchmarks and Why Models Fail

Hendrycks and Dietterich [2] introduced CIFAR-10-C and ImageNet-C. RobustBench [3] gives standard leaderboards and pretrained models. Yin et al. [13] showed that augmentations that help with noise can hurt on fog and contrast. Mintun et al. [14] showed that an augmentation helps most on damage that looks like it. So a fair test must not train on edits that copy the test damage.

### 2.2 Training-Time Defenses

Mixup [15] blends two pictures and their labels. CutMix [4] pastes a box from one picture onto another and mixes the labels by area. AutoAugment [16] learns a set of picture edits, but its CIFAR-10 policy uses brightness and contrast, which are also CIFAR-10-C corruptions, so we do not use it. AugMix [5] mixes several chains of simple edits and adds a JSD loss, so the model gives similar answers for different versions of the same picture.

### 2.3 Test-Time Defenses

BN-adapt [6] replaces the stored batch normalization statistics with statistics from the test batch. Tent [7] also updates the BN scale and shift parameters to lower prediction entropy. CoTTA [9] studies long streams that keep changing. SAR [11] studies small batches and mixed shifts. RDumb [10] shows that regularly resetting the model is a strong baseline for long streams. Lowering entropy can also make a model over-confident, so we also measure calibration [17].

### 2.4 Closest Work

Three studies are closest to ours. Schneider et al. [6] applied BN-adapt to AugMix models and found a smaller gain for stronger models. TTAB [8] trained CIFAR-10 models with MixUp, AugMix and PixMix, and found that adaptation gives them only a small extra gain. Guha Majumdar and Saha [12] measured whether BN-adapt, Tent and EATA help or harm in each of the 75 CIFAR-10-C conditions, but only on a standard-trained model.

None of these studies tests whether augmentation-trained models make adaptation safer in continual or small-batch streams. Our project fills this gap.

## 3 Method

### 3.1 Stage 1: Training the Models

In Stage 1 we train the models that Stage 2 tests. Every model uses the same network and the same training settings. Only the training pictures and the loss change, so any difference comes from the training method.

**Data.** We train on the 50,000 CIFAR-10 training pictures [18]. Each picture is 32x32 and belongs to 1 of 10 classes. We check clean accuracy on the 10,000 CIFAR-10 test pictures. We only use the test accuracy to report results, not to pick a model. Each saved model is the model from the last epoch. We do not use CIFAR-10-C [2] in Stage 1.

**Model.** We use ResNet-18 [1], the version made for 32x32 pictures. It has about 11 million parameters and a batch normalization (BN) layer after each convolution. BN-adapt and Tent both need these BN layers. The model takes pictures with values from 0 to 1 and normalizes them inside, so Stage 2 can use it directly.

**Training methods.** All methods use random crop (padding 4) and random horizontal flip.

| Method | What it does | Loss |
|---|---|---|
| Standard | Crop and flip only | Cross-entropy |
| CutMix [4] | Pastes a box from one picture onto another and mixes the labels by area | Cross-entropy |
| AugMix without JSD [5] | Trains on one AugMix picture for each image | Cross-entropy |
| AugMix + JSD [5] | Uses the clean picture and 2 AugMix pictures, and asks the model to give the same answer for all 3 | Cross-entropy + 12 x JSD |

To make an AugMix picture, we make 3 edited copies. Each copy gets 1 to 3 random edits at severity 3. We mix the copies together, then mix the result with the original picture.

**Keeping the test fair.** The model must not train on the same changes it is tested on. So we use only 9 edits: autocontrast, equalize, posterize, rotate, solarize, shear x, shear y, translate x and translate y. We do not use brightness, contrast, color or sharpness, and a unit test checks this.

**Training settings.**

| Setting | Value |
|---|---|
| Epochs | 100 |
| Batch size | 128 |
| Optimizer | SGD, momentum 0.9, Nesterov |
| Learning rate | 0.1 with cosine decay |
| Weight decay | 0.0005 |
| Precision | Mixed precision (FP16) |
| CutMix | alpha = 1.0 |
| AugMix | 3 copies, 1–3 edits, severity 3, JSD weight 12 |

These are the AugMix default settings. We did not tune them for each method. We trained 6 models: 2 seeds for Standard, 2 seeds for AugMix + JSD, and 1 seed each for CutMix and AugMix without JSD.

### 3.2 Stage 2: Test-Time Adaptation **[Yash]**

**[Yash: BN-adapt and Tent setup, Tent learning rates, the 3 protocols (episodic, continual, small batch), and the metrics (error, harm rate, help rate, paired bootstrap, collapse score, ECE). See sections 3–4 of `docs/stage2_report.md`.]**

## 4 Results

### 4.1 Clean Results

| Model | Seed | Clean accuracy | Clean error | Seconds per epoch | GPU hours | Where |
|---|---|---|---|---|---|---|
| Standard | 1 | 95.08% | 4.92% | 23.37 | 0.649 | Colab T4 |
| Standard | 2 | 94.97% | 5.03% | 20.84 | 0.579 | Colab T4 |
| AugMix + JSD | 1 | 95.21% | 4.79% | 141.58 | 3.933 | Colab T4, then Kaggle T4 |
| AugMix + JSD | 2 | 95.29% | 4.71% | 80.65 | 2.240 | Kaggle T4 |
| AugMix without JSD | 1 | 95.31% | 4.69% | 40.63 | 1.129 | Kaggle T4 |
| CutMix | 1 | 96.18% | 3.82% | 18.66 | 0.518 | Kaggle T4 |

All models reach about 95–96% clean accuracy. The two seeds are close: they differ by 0.11 points for Standard and 0.08 points for AugMix + JSD. CutMix is the best on clean pictures.

Stage 2 got the same clean error for 5 of the 6 models. AugMix + JSD seed 2 differs by one test picture (4.72% vs 4.71%). This shows the models were passed to Stage 2 correctly.

### 4.2 Corruption Results (RQ1 and RQ2)

This table shows the mean error over the 75 CIFAR-10-C conditions, using the episodic test with batch size 200. Standard and AugMix + JSD are the mean of 2 seeds.

| Model | Clean error, no adaptation | No adaptation | BN-adapt | Tent |
|---|---|---|---|---|
| Standard | 4.98% | 25.88% | 15.85% | 13.43% |
| CutMix | 3.82% | 28.50% | 15.33% | 12.22% |
| AugMix without JSD | 4.69% | 13.19% | 11.95% | 10.19% |
| AugMix + JSD | 4.76% | 11.01% | 10.45% | 9.01% |

**RQ1 and H1: which training method is best on corruptions?** AugMix + JSD has the lowest error, so H1 is supported. Both of its seeds (10.84% and 11.17%) beat every other model. CutMix is the best on clean pictures, but it is 2.61 points worse than Standard on corruptions.

**RQ2 and H2: how much does JSD help?** Removing JSD adds 2.18 points of error, which is close to the 2 points we expected. So H2 is supported, but AugMix without JSD has only 1 seed. Compared with Standard, AugMix + JSD lowers the error by 14.88 points. About 12.70 points come from the augmentation and 2.18 points come from JSD.

![Mean error by severity](../results/all_models/figures/severity_curves.png)

**Severity curves.** Without adaptation, the error of Standard and CutMix grows quickly as severity goes up. At severity 5, Standard has 43.74% error and AugMix + JSD has 17.64%. Adaptation makes the curves much flatter. At severity 1, adaptation makes AugMix + JSD a little worse (6.78% with no adaptation, 7.95% with BN-adapt and 6.96% with Tent). Tent helps it from severity 2, and BN-adapt helps it from severity 4.

**H5: which corruptions stay hard?** H5 is partly supported. Glass blur and noise stay among the hardest, but JPEG compression also joins the 5 hardest after adaptation. Contrast gets much easier for Standard (23.5% to 9.9% with BN-adapt). Fog improves less (11.4% to 9.4%), and BN-adapt makes fog slightly worse for AugMix + JSD (8.4% to 8.8%).

### 4.3 How Much Tent Adds After BN-adapt (RQ3) **[Yash]**

**[Yash: from section 5.6 of `docs/stage2_report.md`, with the two fixes: at severity 1 Tent fully recovers only for AugMix without JSD, and "5.7–8.2" should be "5.6–8.2".]**

### 4.4 Continual and Small-Batch Streams (RQ4) **[Yash]**

**[Yash: continual stream and small batches for all 6 models, including the 3 Tent continual runs per model.]**

### 4.5 Where Adaptation Helps or Harms, and Calibration (RQ5) **[Yash]**

**[Yash: harm and help rates over the 75 corrupted conditions, the help/harm maps, collapse and ECE.]**

### 4.6 Tent with a Periodic Reset (E6) **[Yash]**

**[Yash: short version of section 5.7 of `docs/stage2_report.md`. Resetting every 100 batches takes continual Tent from 26.0 to 18.5 (Standard seed 2) and from 24.2 to 17.5 (CutMix).]**

### 4.7 Training Cost

All 6 runs took about 9.05 GPU hours on T4 GPUs, plus about 0.2 hours for timing runs. This is inside the 6–11 hours we planned. The two AugMix + JSD runs used 6.17 hours, which is 68% of the total.

| Method | Colab, 2 CPU workers (s/epoch) | Kaggle, 4 CPU workers (s/epoch) |
|---|---|---|
| Standard | 21.0 | Not run |
| CutMix | 23.0 | 18.66 |
| AugMix without JSD | 91.0 | 40.63 |
| AugMix + JSD | 180.6 | 80.65 |

On Colab, AugMix + JSD took about 8.6 times longer per epoch than Standard. AugMix is slow because the pictures are edited on the CPU. With 4 CPU workers instead of 2, both AugMix versions ran about 2.2 times faster. Because of this, GPU hours from Colab and Kaggle should not be compared directly.

## 5 Discussion **[Together]**

**[Together: answer the main question using RQ1–RQ5 and E6. Is adaptation safer for augmentation-trained models? Lower error, but more often harmed in the episodic test; more stable in the continual stream; a periodic reset fixes the drift for Standard and CutMix.]**

## 6 Problems and Limitations

**Problems we fixed in Stage 1.**

- **JSD loss gave `nan`.** It took the log of 0 in some cases. We rewrote it in a safe way and added a test.
- **Training stopped.** Free GPU sessions can stop at any time, so `train.py` saves every 5 epochs and continues on its own. AugMix + JSD seed 1 stopped on Colab at epoch 86, so we continued it on Kaggle from epoch 85.
- **Layer names did not match Stage 2.** We renamed them, so Stage 2 can load the models directly.

**Limitations.**

- CutMix and AugMix without JSD have only 1 seed, so those results are less certain.
- We used one network, one dataset and one set of settings, and we did not tune them for each method.
- CutMix can need more epochs, so its result at 100 epochs may be a little low.
- The training runs cannot be repeated exactly. Checkpoints do not save the random state, and we used two platforms.
- **[Yash: Stage 2 limitations.]**

## 7 Conclusion **[Together]**

**[Together: 3–4 sentences with the main answer and what we would do next.]**

## References

[1] K. He, X. Zhang, S. Ren, and J. Sun, "Deep Residual Learning for Image Recognition," CVPR, 2016.

[2] D. Hendrycks and T. Dietterich, "Benchmarking Neural Network Robustness to Common Corruptions and Perturbations," ICLR, 2019.

[3] F. Croce, M. Andriushchenko, V. Sehwag, E. Debenedetti, N. Flammarion, M. Chiang, P. Mittal, and M. Hein, "RobustBench: A Standardized Adversarial Robustness Benchmark," NeurIPS Datasets and Benchmarks Track, 2021.

[4] S. Yun, D. Han, S. J. Oh, S. Chun, J. Choe, and Y. Yoo, "CutMix: Regularization Strategy to Train Strong Classifiers with Localizable Features," ICCV, 2019.

[5] D. Hendrycks, N. Mu, E. D. Cubuk, B. Zoph, J. Gilmer, and B. Lakshminarayanan, "AugMix: A Simple Data Processing Method to Improve Robustness and Uncertainty," ICLR, 2020.

[6] S. Schneider, E. Rusak, L. Eck, O. Bringmann, W. Brendel, and M. Bethge, "Improving Robustness Against Common Corruptions by Covariate Shift Adaptation," NeurIPS, 2020.

[7] D. Wang, E. Shelhamer, S. Liu, B. Olshausen, and T. Darrell, "Tent: Fully Test-Time Adaptation by Entropy Minimization," ICLR, 2021.

[8] H. Zhao, Y. Liu, A. Alahi, and T. Lin, "On Pitfalls of Test-Time Adaptation," ICML, 2023.

[9] Q. Wang, O. Fink, L. Van Gool, and D. Dai, "Continual Test-Time Domain Adaptation," CVPR, 2022.

[10] O. Press, S. Schneider, M. Kümmerer, and M. Bethge, "RDumb: A Simple Approach That Questions Our Progress in Continual Test-Time Adaptation," NeurIPS, 2023.

[11] S. Niu, J. Wu, Y. Zhang, Z. Wen, Y. Chen, P. Zhao, and M. Tan, "Towards Stable Test-Time Adaptation in Dynamic Wild World," ICLR, 2023.

[12] S. Guha Majumdar and A. Saha, "When Test-Time Adaptation Helps, Harms, or Becomes Inactive: A Condition-Level Study on CIFAR-10-C," arXiv:2608.22233, 2026.

[13] D. Yin, R. Gontijo Lopes, J. Shlens, E. D. Cubuk, and J. Gilmer, "A Fourier Perspective on Model Robustness in Computer Vision," NeurIPS, 2019.

[14] E. Mintun, A. Kirillov, and S. Xie, "On Interaction Between Augmentations and Corruptions in Natural Corruption Robustness," NeurIPS, 2021.

[15] H. Zhang, M. Cisse, Y. N. Dauphin, and D. Lopez-Paz, "mixup: Beyond Empirical Risk Minimization," ICLR, 2018.

[16] E. D. Cubuk, B. Zoph, D. Mane, V. Vasudevan, and Q. V. Le, "AutoAugment: Learning Augmentation Strategies from Data," CVPR, 2019.

[17] C. Guo, G. Pleiss, Y. Sun, and K. Q. Weinberger, "On Calibration of Modern Neural Networks," ICML, 2017.

[18] A. Krizhevsky, "Learning Multiple Layers of Features from Tiny Images," Technical Report, University of Toronto, 2009.
