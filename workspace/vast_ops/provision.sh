#!/usr/bin/env bash
# Vast box 1台を end-to-end でプロビジョニング: code+data upload -> bootstrap -> tmux 学習開始
#
#   bash provision.sh <ssh_port> <ssh_host> <exp> <fold>
#   例: bash provision.sh 28090 ssh9.vast.ai expA06_f2c_loss 2
#
# 前提: package_data.sh / package_code.sh 実行済み。WANDB_API_KEY は ~/.netrc から自動取得。
set -uo pipefail
PORT="${1:?ssh port}"; HOST="${2:?ssh host}"; EXP="${3:?exp dir}"; FOLD="${4:?fold}"
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SSHOPT="-o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o ConnectTimeout=25"
DATA="${DIR}/tiger_train_data.tar.zst"
SIZE=$(stat -c %s "${DATA}")
WKEY=$(awk '/api.wandb.ai/{f=1} f&&/password/{print $2; exit}' ~/.netrc 2>/dev/null || true)
say() { echo "[$(date '+%F %T')] ${EXP}_f${FOLD}@${HOST}:${PORT} $*"; }

say "code + bootstrap upload"
scp ${SSHOPT} -P "${PORT}" "${DIR}/tiger_code.tgz" "${DIR}/bootstrap.sh" "root@${HOST}:~/" >/dev/null 2>&1 \
  || { say "CODE SCP FAILED"; exit 1; }

have=$(timeout 30 ssh ${SSHOPT} -p "${PORT}" "root@${HOST}" "stat -c %s ~/tiger_train_data.tar.zst 2>/dev/null || echo 0" 2>/dev/null | tail -1)
if [ "${have:-0}" -lt "${SIZE}" ]; then
  say "data upload ($(( SIZE / 1000000 )) MB)"
  rsync -e "ssh ${SSHOPT} -p ${PORT}" --partial --inplace --timeout=180 \
      "${DATA}" "root@${HOST}:~/" || { say "UPLOAD FAILED"; exit 1; }
fi
got=$(timeout 30 ssh ${SSHOPT} -p "${PORT}" "root@${HOST}" "stat -c %s ~/tiger_train_data.tar.zst" 2>/dev/null | tail -1)
[ "${got:-0}" -ge "${SIZE}" ] || { say "SIZE MISMATCH ${got} != ${SIZE}"; exit 1; }
say "data present"

say "bootstrap (extract + deps + tmux train)"
timeout 1200 ssh ${SSHOPT} -p "${PORT}" "root@${HOST}" \
  "EXP=${EXP} FOLD=${FOLD} WANDB_API_KEY='${WKEY}' bash ~/bootstrap.sh" \
  2>&1 | grep -vE "Welcome|Have fun|Warning" | sed "s/^/  /"
say "launched"
