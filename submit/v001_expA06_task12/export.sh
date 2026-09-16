#!/usr/bin/env bash
# 提出物 tar.gz を作成: tigersqai_<team>_task12.tar.gz (命名は wiki 639935 準拠)
set -euo pipefail
TEAM="${TEAM:-shunsuke}"; VER="${VER:-v1}"
cd "$(dirname "$0")"
OUT="tigersqai_${TEAM}_task12.tar.gz"
docker save "tigersqai26_${TEAM}:${VER}" | gzip > "${OUT}"
ls -la --block-size=M "${OUT}"
echo "アップロードは手動で: Synapse evaluation queue へ (${OUT})"
