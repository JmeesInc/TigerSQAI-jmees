#!/usr/bin/env bash
# docker build。TEAM は提出チーム名 (Synapse 登録名) に合わせること。
set -euo pipefail
TEAM="${TEAM:-shunsuke}"
VER="${VER:-v1}"
cd "$(dirname "$0")"
docker build --platform linux/amd64 -t "tigersqai26_${TEAM}:${VER}" .
echo "built tigersqai26_${TEAM}:${VER}"
