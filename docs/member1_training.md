# Member 1: how to train

## Runs to make

| Run | Command | Week |
|---|---|---|
| Standard, seed 1 | `python train.py --method standard --seed 1` | 1 |
| Standard, seed 2 | `python train.py --method standard --seed 2` | 2 |
| AugMix + JSD, seed 1 | `python train.py --method augmix --seed 1` | 2 |
| AugMix + JSD, seed 2 | `python train.py --method augmix --seed 2` | 3 |
| AugMix, no JSD | `python train.py --method augmix_nojsd --seed 1` | 3 |
| CutMix | `python train.py --method cutmix --seed 1` | 3-4 |

Each run makes a folder `runs/<method>_seed<N>/` with:

- `log.csv`: loss, train accuracy, test accuracy (every 5 epochs), seconds per epoch
- `last.pt`: checkpoint, saved every 5 epochs, used to continue
- `final.pt`: the finished model for Member 2 (`model.load("runs/.../final.pt")`)
- `summary.json`: clean accuracy and total hours

## Google Colab

```python
from google.colab import drive
drive.mount('/content/drive')
!git clone https://github.com/KirtanPatel30/Augmentation-Test-time-Adaptation.git
%cd Augmentation-Test-time-Adaptation
!python train.py --method standard --seed 1 --out /content/drive/MyDrive/cmpe252/runs
```

The checkpoints go to Google Drive. If the session stops, run the same cells again.

## Kaggle

Use `kaggle_train.ipynb`. Settings: GPU T4 x2, Internet on, and add the `cifar10-python` dataset (your uploaded `cifar-10-python.tar.gz`) as input.
Set METHOD and SEED, then Save Version > Save & Run All. Download `final.pt` and `summary.json` from the Output tab and put them in `cmpe252/runs/<method>_seed<N>/` on Drive.

## Week 1: time 2 epochs of each method

```python
!python time_epochs.py
```

This writes `results/epoch_timing.csv`.
