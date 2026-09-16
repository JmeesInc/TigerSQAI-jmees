#!/usr/bin/env bash
# 学習に必要な最小データを tar.zst に固める → tiger_train_data.tar.zst (~450MB)
# 含むもの: 1024x576 キャッシュ一式 / class_weights / LUT / fold v1 / labelmap.csv
# 含まないもの: 元解像度 data/ (OOF 推論・公式評価はローカルで行う)
set -euo pipefail
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "${DIR}/../.." && pwd)"
OUT="${DIR}/tiger_train_data.tar.zst"
cd "${REPO}"
tar --use-compress-program 'zstd -3 -T8' -cf "${OUT}" \
  workspace/data_proc/images_1024 \
  workspace/data_proc/labels_fine_1024 \
  workspace/data_proc/labels_coarse_1024 \
  workspace/data_proc/class_weights.json \
  workspace/fold/v1/folds.csv \
  data/labelmap.csv
ls -la --block-size=M "${OUT}"
