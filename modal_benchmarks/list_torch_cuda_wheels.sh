#!/usr/bin/env bash
# List the torch versions published for a given CUDA wheel tag on download.pytorch.org.
#
# Use this before adding/trusting a (torch, cuda) pin in torch_cuda_benchmark.py's
# MATRIX — PyTorch's CUDA-tag support changes with every release, so a pin that was
# correct when this script was written can later 404 at image-build time.
#
# Usage:
#   ./list_torch_cuda_wheels.sh cu121
#   ./list_torch_cuda_wheels.sh cu130

set -euo pipefail

if [[ $# -ne 1 ]]; then
  echo "Usage: $0 <cuda-tag>  (e.g. cu121, cu124, cu126, cu128, cu130)" >&2
  exit 1
fi

tag="$1"
url="https://download.pytorch.org/whl/${tag}/torch/"

curl -fsSL "$url" \
  | grep -oE "torch-[0-9]+\.[0-9]+\.[0-9]+(\+${tag})?-" \
  | sed -E "s/^torch-([0-9]+\.[0-9]+\.[0-9]+).*/\1/" \
  | sort -Vu
