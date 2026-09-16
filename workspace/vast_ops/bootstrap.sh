#!/usr/bin/env bash
# Vast box 上で実行: データ展開 -> 依存 install -> tmux 内で学習開始
# 呼び出し: EXP=expA06_f2c_loss FOLD=0 [WANDB_API_KEY=...] bash ~/bootstrap.sh
set -uo pipefail
EXP="${EXP:?EXP (実験ディレクトリ名)}"
FOLD="${FOLD:?FOLD}"
R=~/TigerSQAI

mkdir -p "${R}"
[ -d "${R}/workspace/data_proc/images_1024" ] || \
  tar --use-compress-program 'zstd -d' -xf ~/tiger_train_data.tar.zst -C "${R}"
tar -xzf ~/tiger_code.tgz -C "${R}"

# deps (vast の pytorch イメージ前提: torch は同梱)
python3 - <<'EOF' 2>/dev/null || pip install -q lightning==2.6.5 "monai==1.5.1" \
  "segmentation-models-pytorch==0.5.0" "timm==1.0.22" "albumentations==2.0.8" \
  opencv-python-headless pandas pyyaml wandb
import lightning, monai, segmentation_models_pytorch, timm, albumentations, cv2, pandas, yaml, wandb
EOF
python3 -c "import lightning, monai, segmentation_models_pytorch, timm, albumentations, cv2, pandas, wandb; print('deps ok')" || exit 1

command -v tmux >/dev/null || (apt-get update -qq && apt-get install -y -qq tmux) >/dev/null 2>&1

SESSION="tiger_${EXP}_f${FOLD}"
tmux kill-session -t "${SESSION}" 2>/dev/null
tmux new-session -d -s "${SESSION}" \
  "cd ${R}/workspace/${EXP} && WANDB_API_KEY='${WANDB_API_KEY:-}' \
   python3 train.py --fold ${FOLD} >> train_console.log 2>&1; \
   echo EXIT_\$? >> train_console.log"
echo "launched tmux ${SESSION}"
