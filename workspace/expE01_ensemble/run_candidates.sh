#!/usr/bin/env bash
# 候補アンサンブルを順に推論 -> 原寸 Dice 評価する。
#   ./run_candidates.sh <gpu>
# 候補は下の CANDS 配列に "tag|members..." 形式で並べる。
set -uo pipefail
cd "$(dirname "$0")"
PY="$(cd ../.. && pwd)/.venv/bin/python3"
GPU="${1:-1}"
EVAL="$(cd ../.. && pwd)/workspace/expA23_sweep/eval_dice_fullres.py"

CANDS=(
  "candB_new7|A23:expA23_l_dicedet A23:expA23_l_rules A23:expA23_e_convnext_xl_384 A23:expA23_d_deeplabv3p A23:expA23_d_base A23:expA23_c_dicedet_nohflip A23:expA23_h_upernet_swin_l"
  "candC_mix|A23:expA23_l_dicedet A23:expA23_l_rules A23:expA23_e_convnext_xl_384 A23:expA23_d_deeplabv3p A06 A09 A10 A11 A05"
  "candD_dicedet2|A23:expA23_l_dicedet A23:expA23_l_rules"
)

for c in "${CANDS[@]}"; do
  tag="${c%%|*}"; members="${c#*|}"
  echo "=== $tag $(date +%H:%M) ==="
  CUDA_VISIBLE_DEVICES="$GPU" "$PY" predict_ens2.py --members $members \
    --folds 0 1 2 3 4 --device cuda:0 --tag "$tag" --no-eval --save-probs \
    >> "logs_${tag}.log" 2>&1
  "$PY" "$EVAL" --pred "workspace/expE01_ensemble/results/${tag}" 2>&1 | tail -3
done
echo "=== 全候補 完了 $(date +%H:%M) ==="
