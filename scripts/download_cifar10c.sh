#!/usr/bin/env bash
# Get CIFAR-10-C into <data_dir>/CIFAR-10-C.
#
# Tries our Hugging Face mirror first (fast, no login for a public repo), then
# falls back to the original Zenodo record 2535967 (CC BY 4.0, about 2.9 GB).
#
# Usage: bash scripts/download_cifar10c.sh [data_dir]
#        CIFAR10C_HF_REPO=user/cifar10-c bash scripts/download_cifar10c.sh
set -euo pipefail
DATA_DIR="${1:-data}"
HF_REPO="${CIFAR10C_HF_REPO:-}"
mkdir -p "$DATA_DIR"
if [ -f "$DATA_DIR/CIFAR-10-C/labels.npy" ]; then
  echo "CIFAR-10-C already in $DATA_DIR/CIFAR-10-C"
  exit 0
fi

if [ -n "$HF_REPO" ]; then
  echo "Downloading CIFAR-10-C from Hugging Face ($HF_REPO)"
  if python -c "
import sys
from huggingface_hub import snapshot_download
snapshot_download(repo_id=sys.argv[1], repo_type='dataset', local_dir=sys.argv[2], allow_patterns=['*.npy'])
" "$HF_REPO" "$DATA_DIR/CIFAR-10-C"; then
    ls "$DATA_DIR/CIFAR-10-C"
    exit 0
  fi
  echo "Hugging Face download failed; falling back to Zenodo"
fi

curl -L --fail -o "$DATA_DIR/CIFAR-10-C.tar" "https://zenodo.org/records/2535967/files/CIFAR-10-C.tar?download=1"
tar -xf "$DATA_DIR/CIFAR-10-C.tar" -C "$DATA_DIR"
rm "$DATA_DIR/CIFAR-10-C.tar"
ls "$DATA_DIR/CIFAR-10-C"
