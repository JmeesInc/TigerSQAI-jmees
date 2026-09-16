#!/usr/bin/env bash
# 指定 GPU が空いたら 5fold 展開のジョブ列を流す。
#   usage: ./chain5.sh <gpu> <fold>:<config> [<fold>:<config> ...]
# GPU メモリで空きを判定する（pgrep 方式は自分のコマンドラインにマッチして壊れる）
cd "$(dirname "$0")"
PY="$(cd ../.. && pwd)/.venv/bin/python3"
gpu="$1"; shift
while true; do
  used=$(nvidia-smi -i "$gpu" --query-gpu=memory.used --format=csv,noheader,nounits)
  [ "${used:-9999}" -lt 1000 ] && break
  sleep 60
done
for job in "$@"; do
  fold="${job%%:*}"; cfg="${job#*:}"
  echo "[gpu$gpu] fold$fold $cfg  $(date +%H:%M)"
  CUDA_VISIBLE_DEVICES="$gpu" "$PY" train.py --fold "$fold" --config "configs/${cfg}.yaml" \
    >> "logs/${cfg}_fold${fold}.log" 2>&1
done
echo "[gpu$gpu] all done $(date +%H:%M)"
