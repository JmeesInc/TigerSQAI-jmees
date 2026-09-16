#!/usr/bin/env bash
# 48 ワーカのランダム読みで HDD が 15MB/s まで落ちるので、レシピ単位で逐次読みして
# ページキャッシュに載せる（14 本 x 7.4GB = 104GB、キャッシュ余裕 380GB）。
cd "$(dirname "$0")"
for r in r_xl q_both_dlv3 q_cholec_dlv3 q_evshallow_dlv3 r_toolpaste r_ft_fine r_detbd \
         k_dicedet_anat3d ft_t2_fine k_dicedet_rules s_kdr_seed43 e_convnext_xl_384 l_dicedet q_endovis18_dlv3; do
  s=$(date +%s)
  cat results/expA23_$r/probs/fine/*.npy results/expA23_$r/probs/coarse/*.npy > /dev/null
  echo "$r $(( $(date +%s) - s ))s"
done
echo done
