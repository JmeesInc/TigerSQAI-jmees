#!/usr/bin/env bash
# expA22_anatomy_rules: expA06 + 解剖ルール loss (隣接・排他)。学習・OOF 推論のエントリポイント
# usage:
#   ./run.sh smoke            # 2 epoch / 5% バッチの疎通確認
#   ./run.sh 0                # fold0 を学習 (last.ckpt があれば自動再開)
#   ./run.sh all              # fold0..4 を逐次学習
#   ./run.sh oof [--folds ..] # OOF 推論 + 公式評価
# GPU は CUDA_VISIBLE_DEVICES で指定 (例: CUDA_VISIBLE_DEVICES=1 ./run.sh 0)
set -euo pipefail
cd "$(dirname "$0")"

# プロジェクト venv (transformers/hub 非互換の回避。SESSION_NOTES.md 参照)
VENV="$(cd ../.. && pwd)/.venv"
if [ -f "$VENV/bin/activate" ]; then
  source "$VENV/bin/activate"
fi

case "${1:-}" in
  smoke)
    python3 train.py --fold 0 --epochs 2 --limit-batches 0.05 --smoke
    ;;
  all)
    for f in 0 1 2 3 4; do
      python3 train.py --fold "$f"
    done
    ;;
  oof)
    shift
    python3 predict_oof.py "$@"
    ;;
  [0-4])
    python3 train.py --fold "$1"
    ;;
  *)
    echo "usage: ./run.sh {smoke|0..4|all|oof [args]}" >&2
    exit 1
    ;;
esac
