#!/usr/bin/env bash
cd "$(dirname "$0")"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=2
exec ../../.venv/bin/python3 ens_greedy_gpu.py \
  --max-size 10 --jobs 96 --out ens_greedy_gpu.json \
  --arms q_endovis18_dlv3 r_xl q_both_dlv3 q_cholec_dlv3 q_evshallow_dlv3 r_toolpaste \
         r_ft_fine r_detbd k_dicedet_anat3d ft_t2_fine k_dicedet_rules s_kdr_seed43 \
         e_convnext_xl_384 l_dicedet
