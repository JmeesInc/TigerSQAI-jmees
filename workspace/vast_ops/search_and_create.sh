#!/usr/bin/env bash
# オファー検索 → インスタンス作成のヘルパ。
#
#   bash search_and_create.sh search [min_vram_gb] [max_dph]   # 候補一覧
#   bash search_and_create.sh create <offer_id> [disk_gb]      # 作成 (pytorch イメージ)
#   bash search_and_create.sh list                             # 稼働中インスタンス
#
# GPU 要件の目安:
#   A系 (MaxViT+f2c): VRAM >= 24GB (余裕を見て 40GB+), disk 40GB
#   B系 (DINOv3-7B):  VRAM >= 40GB, disk 90GB (DINOv3 26GB を HF から DL するため)
set -euo pipefail
CMD="${1:?search|create|list}"
case "${CMD}" in
  search)
    VRAM="${2:-40}"; DPH="${3:-0.8}"
    vastai search offers \
      "gpu_ram>=${VRAM} num_gpus=1 dph<=${DPH} reliability>0.98 inet_down>500 cuda_vers>=12.1 rentable=true" \
      -o dph --limit 15
    ;;
  create)
    OFFER="${2:?offer id}"; DISK="${3:-90}"
    vastai create instance "${OFFER}" \
      --image pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime \
      --disk "${DISK}" --ssh --direct
    echo "作成後: vastai show instances で ssh host/port を確認 → provision.sh"
    ;;
  list)
    vastai show instances
    ;;
esac
