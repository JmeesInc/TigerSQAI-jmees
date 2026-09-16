#!/usr/bin/env bash
# tar.gz でのファイル提出が必要な場合用 (Synapse Docker repo への push が本線)
set -euo pipefail
TEAM="${TEAM:-shunsuke}"; VER="${VER:-v4}"
cd "$(dirname "$0")"
OUT="tigersqai_${TEAM}_task123_${VER}.tar.gz"
docker save "tigersqai26_${TEAM}:${VER}" | gzip > "${OUT}"
ls -la --block-size=M "${OUT}"
