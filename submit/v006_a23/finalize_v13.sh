#!/usr/bin/env bash
# v13: model_v13 を組む → Dockerfile を差し替える → dl2 で build & 回帰テスト
set -euo pipefail
cd "$(dirname "$0")"
PY=../../.venv/bin/python3

echo "=== 5. model_v13 を組む ==="
$PY make_model_v13.py

echo "=== 6. Dockerfile / .dockerignore ==="
[ -f Dockerfile.v12.bak ] || cp Dockerfile Dockerfile.v12.bak
sed -i 's#^COPY model_v11/ /app/model/#COPY model_v13/ /app/model/#' Dockerfile
sed -i 's#^\# v011:.*#\# v013: v12 の seg 7 枠 25 本 + 6 枠に全データ seed 違いを 1 本ずつ（枠内平均、priority 末尾）#' Dockerfile
grep -q "^model_v11/" .dockerignore || echo "model_v11/" >> .dockerignore
grep -q "^model_v13_new/" .dockerignore || echo "model_v13_new/" >> .dockerignore
grep -n "COPY model\|^# v01" Dockerfile

echo "=== 7. dl2 で build ==="
ssh dl2 "cd ${REPO_DL2:-<repo path on the build host>}/submit/v006_a23 && VER=v13 ./build.sh" 2>&1 | tail -4
echo "=== 7. dl2 で回帰テスト (4090) ==="
ssh dl2 "cd ${REPO_DL2:-<repo path on the build host>}/submit/v006_a23 && VER=v13 GPU_DEV=2 ./test_v12.sh 6" 2>&1 | tail -25
ssh dl2 "docker images --format '{{.Repository}}:{{.Tag}} {{.Size}}' | grep tigersqai26_shunsuke:v13"
