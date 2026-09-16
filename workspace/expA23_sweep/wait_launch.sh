#!/usr/bin/env bash
# GPU が空く (<1GB) まで待ってから全データ学習を 1 本投入する
cd "$(dirname "$0")"
g="$1"; n="$2"
while true; do
  used=$(nvidia-smi -i "$g" --query-gpu=memory.used --format=csv,noheader,nounits)
  [ "${used:-9999}" -lt 1000 ] && break
  sleep 60
done
echo "$(date +%H:%M) GPU$g <- $n"
CUDA_VISIBLE_DEVICES=$g exec ../../.venv/bin/python3 train.py --fold 0 --config "configs/expA23_$n.yaml"
