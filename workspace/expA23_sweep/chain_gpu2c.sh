#!/usr/bin/env bash
# chain_gpu2b (PID を引数で受ける) の終了を待って fold1 確認の第 3 陣に入る。
# pgrep でのパターン待ちは取りこぼしたので、PID の生存確認に変更。
cd "$(dirname "$0")"
source ../../.venv/bin/activate
wait_pid="$1"; shift
while kill -0 "$wait_pid" 2>/dev/null; do sleep 60; done
python3 sweep.py --gpus 2 --fold 1 --configs "$@"
