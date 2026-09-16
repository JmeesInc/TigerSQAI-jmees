#!/usr/bin/env bash
cd "$(dirname "$0")"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 HD_LONG=480
exec ../../.venv/bin/python3 ens_greedy_mgpu.py \
  --gpus 0 1 2 3 --per-gpu 2 --max-size 10 --out ens_greedy_mgpu.json \
  --arms q_endovis18_dlv3 r_xl q_both_dlv3 q_cholec_dlv3 q_evshallow_dlv3 r_toolpaste \
         r_ft_fine r_detbd k_dicedet_anat3d k_dicedet_rules s_kdr_seed43 e_convnext_xl_384 l_dicedet
