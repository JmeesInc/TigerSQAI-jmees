#!/usr/bin/env bash
# GPU2 が空いたら fold1 確認の第 2 陣（aug/loss 軸）に入る
cd "$(dirname "$0")"
source ../../.venv/bin/activate
while true; do
  used=$(nvidia-smi -i 2 --query-gpu=memory.used --format=csv,noheader,nounits)
  [ "${used:-9999}" -lt 1000 ] && break
  sleep 120
done
python3 sweep.py --gpus 2 --fold 1 --configs configs/expA23_a_nohflip.yaml configs/expA23_l_dicedet.yaml
