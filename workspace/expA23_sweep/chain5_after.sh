#!/usr/bin/env bash
# 指定 PID の終了を待ってから 5fold ジョブ列を流す
cd "$(dirname "$0")"
PY="$(cd ../.. && pwd)/.venv/bin/python3"
gpu="$1"; wait_pid="$2"; shift 2
while kill -0 "$wait_pid" 2>/dev/null; do sleep 60; done
for job in "$@"; do
  fold="${job%%:*}"; cfg="${job#*:}"
  echo "[gpu$gpu] fold$fold $cfg $(date +%H:%M)"
  CUDA_VISIBLE_DEVICES="$gpu" "$PY" train.py --fold "$fold" --config "configs/${cfg}.yaml" \
    >> "logs/${cfg}_fold${fold}.log" 2>&1
done
echo "[gpu$gpu] all done $(date +%H:%M)"
