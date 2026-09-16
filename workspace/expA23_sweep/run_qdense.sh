#!/usr/bin/env bash
cd "$(dirname "$0")"
until [ "$(ls results/expA23_q_dense_dlv3/probs/coarse 2>/dev/null | wc -l)" -ge 526 ]; do sleep 15; done
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 HD_LONG=480
B=(q_endovis18_dlv3 r_xl r_ft_fine q_both_dlv3 s_kdr_seed43 l_dicedet r_detbd)
S=("$(IFS=,; echo "${B[*]}"),q_dense_dlv3")
for i in "${!B[@]}"; do R=("${B[@]}"); unset 'R[i]'; S+=("$(IFS=,; echo "${R[*]}"),q_dense_dlv3"); done
exec ../../.venv/bin/python3 ens_greedy_mgpu.py --arms x --gpus 2 --per-gpu 3 --max-size 0 --out ens_qdense_mgpu.json --sets "${S[@]}"
