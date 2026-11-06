# Final Report Draft: Introduction and Related Work

**Author:** Kirtan Patel (Member 1)
**Status:** Draft. Reference numbers follow the project proposal.

---

## 1 Introduction

Deep image models such as ResNets [1] get more than 95% accuracy on CIFAR-10. But this only holds when the test pictures look like the training pictures. In real use, this is often not true. A car camera sees rain and fog, a scanner adds noise, and a web picture gets compressed. A person can still recognize these pictures, but a model can make many more mistakes [2].

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
