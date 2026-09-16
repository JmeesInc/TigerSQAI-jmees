#!/usr/bin/env bash
# GPU 空き待ち付きの実験キュー: expA06 fold1-4 + OOF -> expB02 fold1-4 + OOF
# (他プロジェクトのジョブと GPU を共有しているため、空き >= NEED_MB のカードを待って使う)
set -u
cd "$(dirname "$0")"

wait_gpu() {  # $1 = 必要 MB。空いた GPU 番号を echo
  local need=$1
  while true; do
    for g in 0 1; do
      local used total free
      used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i $g)
      total=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits -i $g)
      free=$((total - used))
      if [ "$free" -ge "$need" ]; then echo $g; return; fi
    done
    sleep 180
  done
}

for f in 1 2 3 4; do
  g=$(wait_gpu 22800)
  echo "=== expA06 fold$f start (GPU $g) ==="
  ( cd expA06_f2c_loss && CUDA_VISIBLE_DEVICES=$g ./run.sh "$f" >> train_console.log 2>&1 )
done
echo "=== expA06 all folds done; OOF predict+eval ==="
g=$(wait_gpu 8000)
( cd expA06_f2c_loss && CUDA_VISIBLE_DEVICES=$g ./run.sh oof > oof_all.log 2>&1 ) && echo "=== expA06 FULL OOF EVAL DONE ==="

for f in 1 2 3 4; do
  g=$(wait_gpu 37000)   # B02 は maxvit学習+7B で ~36GB 必要
  echo "=== expB02 fold$f start (GPU $g) ==="
  ( cd expB02_dino_maxvit && CUDA_VISIBLE_DEVICES=$g ./run.sh "$f" >> train_console.log 2>&1 )
done
echo "=== expB02 all folds done; OOF predict+eval ==="
g=$(wait_gpu 18000)
( cd expB02_dino_maxvit && CUDA_VISIBLE_DEVICES=$g ./run.sh oof > oof_all.log 2>&1 ) && echo "=== expB02 FULL OOF EVAL DONE ==="
echo "=== queue complete ==="
