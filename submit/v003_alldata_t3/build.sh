#!/usr/bin/env bash
# docker build。TEAM は提出チーム名 (Synapse 登録名) に合わせること。
set -euo pipefail
TEAM="${TEAM:-shunsuke}"
VER="${VER:-v3}"
cd "$(dirname "$0")"
docker build --platform linux/amd64 -t "tigersqai26_${TEAM}:${VER:-v2}" .
echo "built tigersqai26_${TEAM}:${VER:-v2}"
