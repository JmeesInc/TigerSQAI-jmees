#!/usr/bin/env bash
# 合成データ事前学習 -> 実データ fine-tune の A/B
#
#   ./run.sh prepare   # 生成画像 + 合成ラベル -> expA06 互換データセット
#   ./run.sh pretrain  # 合成データで事前学習
#   ./run.sh finetune  # 事前学習重みから実データ fold0 を学習
#   ./run.sh baseline  # 事前学習なしの対照（同一設定・同一 seed）
set -euo pipefail
cd "$(dirname "$0")/../.."
PY=.venv/bin/python
EXP=workspace/expG02_synthpretrain
PRETRAIN_CKPT=$EXP/results/expG02_pretrain_synth/fold0/best.ckpt

case "${1:?prepare|pretrain|finetune|baseline}" in
prepare)
  $PY $EXP/prepare_synth_dataset.py \
    --generated "${2:?生成画像ディレクトリ}" \
    --labels "${3:?合成ラベルディレクトリ}" \
    --output $EXP/data
  ;;
pretrain)
  $PY $EXP/train.py --fold 0 --config $EXP/config_pretrain.yaml
  ;;
finetune)
  $PY $EXP/train.py --fold 0 --config $EXP/config_finetune.yaml --init-from $PRETRAIN_CKPT
  ;;
baseline)
  # 対照: 事前学習なし。fine-tune と同一 config / 同一 seed で比較可能にする
  $PY $EXP/train.py --fold 0 --config $EXP/config_finetune.yaml
  ;;
esac
