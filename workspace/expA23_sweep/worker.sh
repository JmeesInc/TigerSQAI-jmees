#!/usr/bin/env bash
# 共有キュー (queue5.txt) からジョブを 1 行ずつ取り出して実行する GPU ワーカ。
#
#   ./worker.sh <gpu> [queue_file]
#
# ジョブ行の形式:
#   <fold>:<config>     学習 (例 `2:expA23_l_dicedet`)
#   oof:<config>        5fold の OOF 推論 + 公式評価
#
# 設計の要点:
#   * **flock で 1 行を原子的に pop** する。ワーカを何台足しても二重実行しない
#   * 走行中は `logs/<config>_fold<N>.log` に追記
#   * 学習が終わったら **best.ckpt / latest.ckpt を削除**し、`best_fp16.pt` だけ残す
#     (1 run 5.4GB -> 0.45GB。推論・アンサンブルは fp16 重みで足りる)
#   * キューが空になったら終了。あとから queue5.txt に追記すれば、
#     新しいワーカを起動するだけで続きを流せる
set -uo pipefail
cd "$(dirname "$0")"
# python は host ごとに違う (dl1: リポジトリ直下の .venv / dl2: mm/bin/python)。
# rsync で上書きされても壊れないよう、存在するものを自動で選ぶ
PY="${PY_OVERRIDE:-}"
if [ -z "$PY" ]; then
  for cand in "$(cd ../.. && pwd)/.venv/bin/python3" "$HOME/workdir/TigerSQAI/mm/bin/python"; do
    if [ -x "$cand" ]; then PY="$cand"; break; fi
  done
fi
[ -x "$PY" ] || { echo "python が見つからない"; exit 1; }
export CUDA_DEVICE_ORDER=PCI_BUS_ID
GPU="$1"
QUEUE="${2:-queue5.txt}"
LOCK="${QUEUE}.lock"

# 起動時: 対象 GPU に自分のジョブが載る余地ができるまで待つ。
# 既定は「ほぼ空」だが、別スレッドが 48GB カードを 17GB 程度で使っている間は
# 待ち続けても無駄なので、MAX_USED_MB を上げて **同居**させる (48GB あるので 2 本は載る)。
MAX_USED_MB="${MAX_USED_MB:-1000}"
while true; do
  used=$(nvidia-smi -i "$GPU" --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null)
  [ "${used:-999999}" -lt "$MAX_USED_MB" ] && break
  sleep 60
done

push_back() {
  # 依存未解決のジョブをキュー末尾に戻す
  flock 9
  echo "$1" >> "$QUEUE"
}

pop_job() {
  # flock 下で先頭行を取り出し、キューから消す
  flock 9
  job=$(head -n 1 "$QUEUE" 2>/dev/null || true)
  if [ -n "$job" ]; then
    tail -n +2 "$QUEUE" > "${QUEUE}.tmp" && mv "${QUEUE}.tmp" "$QUEUE"
  fi
  printf '%s' "$job"
}

while true; do
  exec 9>"$LOCK"
  job=$(pop_job)
  exec 9>&-
  [ -z "$job" ] && { echo "[gpu$GPU] queue empty $(date +%H:%M)"; break; }

  job="${job#retry:}"          # 再投入分は接頭辞を外して実行
  kind="${job%%:*}"; cfg="${job#*:}"
  if [ "$kind" = "probs" ]; then
    # アンサンブル探索・α 後処理のための OOF 確率保存 (5 fold 揃ってから)
    n_done=$(ls results/${cfg}/fold*/best_fp16.pt 2>/dev/null | wc -l)
    if [ "$n_done" -lt 5 ]; then
      echo "[gpu$GPU] PROBS $cfg 保留 (${n_done}/5 fold) $(date +%H:%M)"
      exec 9>"$LOCK"; push_back "$job"; exec 9>&-
      sleep 120
      continue
    fi
    echo "[gpu$GPU] PROBS $cfg $(date +%H:%M)"
    CUDA_VISIBLE_DEVICES="$GPU" "$PY" save_probs.py --config "configs/${cfg}.yaml"       --device cuda:0 >> "logs/${cfg}_probs.log" 2>&1
    echo "[gpu$GPU] PROBS $cfg rc=$? $(date +%H:%M)"
  elif [ "$kind" = "oof" ]; then
    # 5 fold 揃うまでは実行しない (揃っていなければ末尾へ戻して次のジョブへ)
    n_done=$(ls results/${cfg}/fold*/best_fp16.pt 2>/dev/null | wc -l)
    if [ "$n_done" -lt 5 ]; then
      echo "[gpu$GPU] OOF $cfg 保留 (${n_done}/5 fold) $(date +%H:%M)"
      exec 9>"$LOCK"; push_back "$job"; exec 9>&-
      sleep 120
      continue
    fi
    echo "[gpu$GPU] OOF $cfg $(date +%H:%M)"
    CUDA_VISIBLE_DEVICES="$GPU" "$PY" predict_oof.py --config "configs/${cfg}.yaml" \
      --device cuda:0 >> "logs/${cfg}_oof.log" 2>&1
    echo "[gpu$GPU] OOF $cfg rc=$? $(date +%H:%M)"
  else
    fold="$kind"
    echo "[gpu$GPU] fold$fold $cfg $(date +%H:%M)"
    CUDA_VISIBLE_DEVICES="$GPU" "$PY" train.py --fold "$fold" --config "configs/${cfg}.yaml" \
      >> "logs/${cfg}_fold${fold}.log" 2>&1
    rc=$?
    d="results/${cfg}/fold${fold}"
    if [ -f "$d/best_fp16.pt" ]; then
      rm -f "$d/best.ckpt" "$d/latest.ckpt" "$d/last.ckpt"
    fi
    echo "[gpu$GPU] fold$fold $cfg rc=$rc $(date +%H:%M)"
    # 即死したジョブ (環境不備など) でキューを溶かさないよう、1 回だけ末尾へ戻す
    if [ "$rc" -ne 0 ] && [ ! -f "$d/best_fp16.pt" ] && [[ "$job" != retry:* ]]; then
      echo "[gpu$GPU] 失敗したので 1 回だけ再投入: $job"
      exec 9>"$LOCK"; push_back "retry:$job"; exec 9>&-
      sleep 30
    fi
  fi
done
