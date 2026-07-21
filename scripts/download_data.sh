#!/usr/bin/env bash
# Fetches the tiny shakespeare dataset used in the video, into data/input.txt
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p data
curl -sL -o data/input.txt \
  https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt
echo "Saved to data/input.txt ($(wc -l < data/input.txt) lines)"
