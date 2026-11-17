#!/usr/bin/env bash
# Downloads the QuALITY v1.0.1 dataset (Pang et al., 2022) into ./data
set -euo pipefail

mkdir -p data
cd data
curl -sSL -o quality.zip \
  https://raw.githubusercontent.com/nyu-mll/quality/main/data/v1.0.1/QuALITY.v1.0.1.zip
unzip -o -q quality.zip
rm -rf __MACOSX quality.zip
echo "QuALITY downloaded to ./data:"
ls -1
