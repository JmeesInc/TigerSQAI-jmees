#!/usr/bin/env bash
# tar.gz でのファイル提出が必要な場合用 (Synapse Docker repo への push が本線)
set -euo pipefail
TEAM="${TEAM:-shunsuke}"; VER="${VER:-v5}"
cd "$(dirname "$0")"
OUT="tigersqai_${TEAM}_task123.tar.gz"
# 13GB のイメージを単スレッド gzip で固めると数時間かかる。pigz があれば並列で使う
if command -v pigz >/dev/null; then
  docker save "tigersqai26_${TEAM}:${VER}" | pigz -p 16 > "${OUT}"
else
  docker save "tigersqai26_${TEAM}:${VER}" | gzip > "${OUT}"
fi
ls -la --block-size=M "${OUT}"
