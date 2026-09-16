#!/bin/bash
# 生成ワーカーと stage B ワーカーを並走させる.
# stage B は --watch で生成完了分から順に擬似ラベル化するので, 生成の完了を待たない.
#   GEN_GPUS / STAGEB_GPUS で使用 GPU を指定 (他セッションが使っている GPU は外すこと)
set -u
cd "$(dirname "$0")/../.."
source .venv/bin/activate
OUT=workspace/expS02_flf2v/outputs
mkdir -p "$OUT/logs"

GEN_GPUS=${GEN_GPUS:-"0 2"}
STAGEB_GPUS=${STAGEB_GPUS:-"3"}
FRAMES=${FRAMES:-25}
STEPS=${STEPS:-30}

read -ra G <<< "$GEN_GPUS"
read -ra B <<< "$STAGEB_GPUS"

for i in "${!G[@]}"; do
  CUDA_VISIBLE_DEVICES=${G[$i]} nohup python3 workspace/expS02_flf2v/scripts/gen_flf2v.py \
      --queue "$OUT/queue.csv" --shard "$i" --num-shards "${#G[@]}" \
      --frames "$FRAMES" --steps "$STEPS" \
      > "$OUT/logs/gen_shard$i.log" 2>&1 &
  echo "gen shard $i/${#G[@]} -> GPU ${G[$i]} (pid $!)"
done

sleep 3
for j in "${!B[@]}"; do
  CUDA_VISIBLE_DEVICES=${B[$j]} PYTHONPATH=reference/tigersqai_challenge nohup \
      python3 workspace/expS02_flf2v/scripts/stage_b.py \
      --queue "$OUT/queue.csv" --shard "$j" --num-shards "${#B[@]}" --watch 120 \
      > "$OUT/logs/stageb_shard$j.log" 2>&1 &
  echo "stageB shard $j/${#B[@]} -> GPU ${B[$j]} (pid $!)"
done
