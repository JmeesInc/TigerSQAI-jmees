#!/usr/bin/env bash
# 採用構成の最終値を **公式 CPU 実装**で出す（write-up 用）
cd "$(dirname "$0")"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
exec ../../.venv/bin/python3 ens_greedy_fast.py --jobs 96 --out ens_sets_final_official.json \
  --sets "q_endovis18_dlv3,r_xl,r_ft_fine,q_both_dlv3,s_kdr_seed43,l_dicedet,r_detbd"
