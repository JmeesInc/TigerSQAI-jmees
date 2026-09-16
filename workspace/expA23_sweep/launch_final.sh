#!/usr/bin/env bash
# 締切当日: fold 検証で上位なのに全データ版が無いレシピを一斉に学習する。
cd "$(dirname "$0")"
PY=../../.venv/bin/python3
run() {  # run <gpu> <config名>
  local g=$1 n=$2
  echo "GPU$g <- $n"
  CUDA_VISIBLE_DEVICES=$g nohup $PY train.py --fold 0 --config "configs/expA23_$n.yaml" \
    > "logs/final_$n.log" 2>&1 &
  sleep 8
}
mkdir -p logs
run 0 full_r_ft_soft
run 1 full_r_xl
run 2 full_q_dense_dlv3
run 3 full_r_ft_long
