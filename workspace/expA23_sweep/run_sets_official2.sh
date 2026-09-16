#!/usr/bin/env bash
cd "$(dirname "$0")"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
exec ../../.venv/bin/python3 ens_greedy_fast.py --jobs 96 --out ens_sets_official2.json \
  --sets "q_endovis18_dlv3,r_xl,r_ft_fine,q_both_dlv3,s_kdr_seed43,l_dicedet,r_detbd" \
         "q_endovis18_dlv3,r_xl,r_ft_fine,q_both_dlv3,s_kdr_seed43,l_dicedet,r_nohflip"
