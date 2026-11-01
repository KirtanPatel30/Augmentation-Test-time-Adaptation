#!/usr/bin/env bash
# Download CIFAR-10-C (Zenodo record 2535967, CC BY 4.0, about 2.9 GB) into data/CIFAR-10-C.
# Usage: bash scripts/download_cifar10c.sh [data_dir]
set -euo pipefail
DATA_DIR="${1:-data}"
mkdir -p "$DATA_DIR"
if [ -f "$DATA_DIR/CIFAR-10-C/labels.npy" ]; then
  echo "CIFAR-10-C already in $DATA_DIR/CIFAR-10-C"
  exit 0
fi
curl -L --fail -o "$DATA_DIR/CIFAR-10-C.tar" "https://zenodo.org/records/2535967/files/CIFAR-10-C.tar?download=1"
tar -xf "$DATA_DIR/CIFAR-10-C.tar" -C "$DATA_DIR"
rm "$DATA_DIR/CIFAR-10-C.tar"
ls "$DATA_DIR/CIFAR-10-C"
