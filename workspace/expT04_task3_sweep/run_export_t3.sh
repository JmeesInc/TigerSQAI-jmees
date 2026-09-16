#!/usr/bin/env bash
cd "$(dirname "$0")"
CUDA_VISIBLE_DEVICES=3 ../../.venv/bin/python3 export_t3_final.py \
  --features features_candE_fix.csv \
  --extra-features features_anat_candE.csv features_grid_candE.csv \
  --oof oof_ens_fixfull_raw.csv \
  --w-lgb 0.6 --threshold-lambda 0.5 \
  --out model_t3_v7
