#!/usr/bin/env bash
# GPU 常駐ワーカ: キューが空でも終了せず待ち続ける。
#
# worker_run.sh はキューを使い切ると exit するため、投入が途切れると GPU が遊ぶ
# （実際 05:35-09:40 の 4 時間遊ばせた）。こちらは常駐して待ち、
# **キューに追記するだけで自動的に次が走る**形にする。
#
#   ./daemon.sh <gpu> [queue_file]      停止: touch daemon.stop
set -u
GPU="$1"; QUEUE="${2:-queue_backlog.txt}"
cd "$(dirname "$0")"
echo "[gpu$GPU] daemon 開始 $(date +%H:%M) queue=$QUEUE"
while [ ! -f daemon.stop ]; do
  if [ -s "$QUEUE" ]; then
    MAX_USED_MB="${MAX_USED_MB:-30000}" ./worker_run.sh "$GPU" "$QUEUE"
  fi
  sleep 60
done
echo "[gpu$GPU] daemon 停止 $(date +%H:%M)"
