#!/usr/bin/env bash
# GPU2: l_boundary が終わったら、先行している h_upernet_swin_l の fold1 確認に入る
cd "$(dirname "$0")"
source ../../.venv/bin/activate
while pgrep -f "sweep.py --gpus 2 --fold 0" > /dev/null; do sleep 60; done
python3 sweep.py --gpus 2 --fold 1 --configs configs/expA23_h_upernet_swin_l.yaml configs/expA23_d_base.yaml
