#!/usr/bin/env bash
# 定期監視を 1 コマンドで済ませる: 状態表示 + バックログ自動補充。
#
# デーモン (daemon.sh) が実消化を担当し、これは安全網:
#   * バックログが薄ければ plan.txt から補充する（GPU を遊ばせない）
#   * デーモンが死んでいたら再起動する（設定バグで全滅した事故があった）
#   * dl1/dl2 の稼働状況を数行で出す
#
#   ./refill.sh [最低バックログ数]
set -u
cd "$(dirname "$0")"
MIN="${1:-4}"
PLAN=plan.txt
Q=queue_backlog.txt
touch "$Q" "$PLAN"

# --- デーモン生存確認（4 基分）
alive=$(pgrep -fc "daemon.sh [0-3] $Q" || true)
if [ "${alive:-0}" -lt 4 ]; then
  for g in 0 1 2 3; do
    pgrep -f "daemon.sh $g $Q" >/dev/null || { nohup ./daemon.sh "$g" "$Q" > "daemon_gpu$g.log" 2>&1 & }
  done
  echo "daemon: $alive/4 → 再起動した"
fi

# --- バックログ補充
have=$(grep -c . "$Q" 2>/dev/null || echo 0)
if [ "$have" -lt "$MIN" ] && [ -s "$PLAN" ]; then
  need=$((MIN * 2 - have))
  (
    flock 9
    head -n "$need" "$PLAN" >> "$Q"
    tail -n +$((need + 1)) "$PLAN" > "$PLAN.tmp" && mv "$PLAN.tmp" "$PLAN"
  ) 9>"$Q.lock"
  echo "補充: $have → $(grep -c . "$Q") 本（plan 残り $(grep -c . "$PLAN")）"
fi

# --- 状態
echo "dl1 $(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits | tr '\n' ' ')| 実行 $(pgrep -fc 'train.py --fold' || echo 0) | queue $(grep -c . "$Q") | plan $(grep -c . "$PLAN")"
ssh -o ConnectTimeout=10 dl2 "cd ${REPO_DL2:-~/TigerSQAI}/workspace/expA23_sweep 2>/dev/null && echo \"dl2 \$(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits|tr '\n' ' ')| 実行 \$(pgrep -fc 'train.py --fold' || echo 0) | queue \$(grep -c . queue_backlog.txt 2>/dev/null || echo 0)\"" 2>/dev/null || echo "dl2 応答なし"

# --- 直近の完了と失敗だけ
echo "直近完了: $(ls -t results/expA23_*/fold*/summary.json 2>/dev/null | head -3 | sed 's|results/expA23_||;s|/summary.json||' | tr '\n' ' ')"
bad=$(grep -l -E "Traceback|OutOfMemory|CUDA error" results/expA23_*/fold*/*.log 2>/dev/null | tail -2 | sed 's|results/expA23_||;s|/[^/]*\.log||' | tr '\n' ' ')
[ -n "$bad" ] && echo "要確認: $bad"
exit 0
