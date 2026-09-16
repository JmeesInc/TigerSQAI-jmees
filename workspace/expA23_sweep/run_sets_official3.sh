#!/usr/bin/env bash
cd "$(dirname "$0")"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
exec ../../.venv/bin/python3 ens_greedy_fast.py --jobs 96 --out ens_sets_official3.json \
  --sets "q_endovis18_dlv3_tta,r_xl,r_ft_fine_tta,q_both_dlv3_tta,s_kdr_seed43_tta,l_dicedet_tta,r_nohflip" \
         "q_endovis18_dlv3_tta,r_xl_tta,r_ft_fine_tta,q_both_dlv3_tta,s_kdr_seed43_tta,l_dicedet_tta,r_nohflip"
