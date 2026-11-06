"""One-time upload of CIFAR-10-C to a Hugging Face dataset repo.

CIFAR-10-C is CC BY 4.0, so a mirror is allowed with attribution; the dataset
card below gives it. After this, download_cifar10c.sh pulls from the mirror:

    CIFAR10C_HF_REPO=<user>/cifar10-c bash scripts/download_cifar10c.sh

Usage (needs a Hugging Face *write* token, from `huggingface-cli login` or HF_TOKEN):
    python scripts/upload_cifar10c_to_hf.py --repo <user>/cifar10-c [--data-dir data/CIFAR-10-C]
"""
import argparse
import glob
import os

from huggingface_hub import HfApi

CARD = """---
license: cc-by-4.0
pretty_name: CIFAR-10-C (mirror)
task_categories:
- image-classification
size_categories:
- 100K<n<1M
---

# CIFAR-10-C (mirror)

Unchanged copy of **CIFAR-10-C** by Dan Hendrycks and Thomas Dietterich, from
[Zenodo record 2535967](https://zenodo.org/records/2535967), licensed under
[CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). Mirrored for the
CMPE 252 project [Augmentation-Test-time-Adaptation](https://github.com/KirtanPatel30/Augmentation-Test-time-Adaptation)
so that Colab sessions can download it quickly.

Each `<corruption>.npy` has shape (50000, 32, 32, 3), uint8: rows
`[(s-1)*10000, s*10000)` are severity `s`. `labels.npy` holds the labels.

Please cite the original paper:

> D. Hendrycks and T. Dietterich. Benchmarking Neural Network Robustness to
> Common Corruptions and Perturbations. ICLR 2019. https://arxiv.org/abs/1903.12261
"""


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--repo", required=True, help="<user>/<name> on Hugging Face")
    p.add_argument("--data-dir", default="data/CIFAR-10-C")
    p.add_argument("--private", action="store_true")
    args = p.parse_args()

    files = sorted(glob.glob(os.path.join(args.data_dir, "*.npy")))
    if len(files) < 20:
        raise SystemExit(f"expected 20 .npy files in {args.data_dir}, found {len(files)}")

    api = HfApi()
    api.create_repo(args.repo, repo_type="dataset", private=args.private, exist_ok=True)
    api.upload_file(path_or_fileobj=CARD.encode(), path_in_repo="README.md",
                    repo_id=args.repo, repo_type="dataset")
    api.upload_folder(folder_path=args.data_dir, repo_id=args.repo, repo_type="dataset",
                      allow_patterns=["*.npy"], commit_message="Add CIFAR-10-C (Zenodo 2535967)")
    print(f"done: https://huggingface.co/datasets/{args.repo}")


if __name__ == "__main__":
    main()
