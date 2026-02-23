#!/usr/bin/env bash
#
# For licensing see accompanying LICENSE_MODEL file.
# Copyright (C) 2025 Apple Inc. All Rights Reserved.
#
# This script mirrors Apple's ml-fastvlm Model Zoo download helper.

set -euo pipefail

CHECKPOINT_DIR="${1:-checkpoints}"

mkdir -p "${CHECKPOINT_DIR}"

base_url="https://ml-site.cdn-apple.com/datasets/fastvlm"
files=(
  # "llava-fastvithd_0.5b_stage2.zip"
  "llava-fastvithd_0.5b_stage3.zip"
  # "llava-fastvithd_1.5b_stage2.zip"
  # "llava-fastvithd_1.5b_stage3.zip"
  # "llava-fastvithd_7b_stage2.zip"
  # "llava-fastvithd_7b_stage3.zip"
)

echo "Downloading FastVLM checkpoints into '${CHECKPOINT_DIR}'..."
for file in "${files[@]}"; do
  wget "${base_url}/${file}" -P "${CHECKPOINT_DIR}"
done

pushd "${CHECKPOINT_DIR}" >/dev/null
echo "Extracting archives..."
for file in "${files[@]}"; do
  unzip -qq "${file}"
  rm "${file}"
done
popd >/dev/null

echo "All checkpoints are downloaded and extracted under '${CHECKPOINT_DIR}'."
