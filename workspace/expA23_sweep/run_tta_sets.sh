#!/usr/bin/env bash
cd "$(dirname "$0")"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 HD_LONG=480
mapfile -t S < tta_sets.txt
exec ../../.venv/bin/python3 ens_greedy_mgpu.py --arms x --gpus 0 2 3 --per-gpu 3 --max-size 0 --out ens_tta_mgpu.json --sets "${S[@]}"
