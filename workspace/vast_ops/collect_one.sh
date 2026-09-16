#!/usr/bin/env bash
# 1台の box を監視し、学習完了で成果物を pull → 検証 → インスタンス destroy。
#
#   bash collect_one.sh <instance_id> <ssh_port> <ssh_host> <exp> <fold>
#
# 不変条件 (MVAA の教訓):
#   * checkpoint は box のディスクにしか無い → pull 成功が destroy の前提。destroy は必ず行う
#   * ssh は短時間呼び出しのみ (長い ssh は harness に殺され box の tmux ごと落ちた事故あり)
set -uo pipefail
ID="${1:?instance id}"; PORT="${2:?port}"; HOST="${3:?host}"; EXP="${4:?exp}"; FOLD="${5:?fold}"
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "${DIR}/../.." && pwd)"
LOCAL="${REPO}/workspace/${EXP}/results/${EXP}/fold${FOLD}"
REMOTE="~/TigerSQAI/workspace/${EXP}/results/${EXP}/fold${FOLD}"
RLOG="~/TigerSQAI/workspace/${EXP}/train_console.log"
SSHOPT="-o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o ConnectTimeout=20"
EVERY="${EVERY:-300}"; MAX_WAIT="${MAX_WAIT:-100000}"
say() { echo "[$(date '+%F %T')] ${EXP}_f${FOLD}@${HOST} $*"; }

t0=$(date +%s)
while :; do
  done_line=$(timeout 30 ssh ${SSHOPT} -p "${PORT}" "root@${HOST}" \
      "grep -oE 'fold${FOLD}/best.ckpt \(val/score=[0-9.]+' ${RLOG} 2>/dev/null | tail -1" 2>/dev/null | tail -1)
  exit_line=$(timeout 30 ssh ${SSHOPT} -p "${PORT}" "root@${HOST}" \
      "grep -c '^EXIT_' ${RLOG} 2>/dev/null || echo 0" 2>/dev/null | tail -1)
  alive=$(timeout 30 ssh ${SSHOPT} -p "${PORT}" "root@${HOST}" \
      "pgrep -fc 'train.p[y] --fold ${FOLD}' 2>/dev/null || echo 0" 2>/dev/null | tail -1)
  say "done='${done_line:-}' exit_lines=${exit_line:-?} procs=${alive:-?}"
  if [ -n "${done_line}" ] && [ "${alive:-1}" = "0" ]; then
    mkdir -p "${LOCAL}"
    say "finished -> pulling"
    ok=1
    for f in best.ckpt latest.ckpt training_log.json config.yaml; do
      scp ${SSHOPT} -P "${PORT}" "root@${HOST}:${REMOTE}/${f}" "${LOCAL}/" >/dev/null 2>&1 || { say "PULL FAILED: ${f}"; ok=0; }
    done
    scp ${SSHOPT} -P "${PORT}" "root@${HOST}:${RLOG}" "${LOCAL}/vast_train_console.log" >/dev/null 2>&1
    if [ "${ok}" = "1" ] && [ -s "${LOCAL}/best.ckpt" ]; then
      say "pull OK ($(du -sh "${LOCAL}" | cut -f1)) -> destroying instance ${ID}"
      echo y | vastai destroy instance "${ID}" && say "destroyed" || say "DESTROY FAILED — 手動で destroy すること (課金継続中)"
      echo "${done_line}"
      exit 0
    fi
    say "PULL INCOMPLETE — destroy せず再試行 (課金継続中に注意)"
  fi
  if [ "${alive:-1}" = "0" ] && [ "${exit_line:-0}" != "0" ] && [ -z "${done_line}" ]; then
    say "TRAINING DIED (EXIT line, no done) — ログを確認。destroy は手動判断"
    timeout 30 ssh ${SSHOPT} -p "${PORT}" "root@${HOST}" "tail -30 ${RLOG}" 2>/dev/null | tail -15
    exit 2
  fi
  [ $(( $(date +%s) - t0 )) -gt "${MAX_WAIT}" ] && { say "TIMEOUT"; exit 3; }
  sleep "${EVERY}"
done
