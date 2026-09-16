#!/usr/bin/env bash
cd "$(dirname "$0")"
PY=../../.venv/bin/python3
E="expA23_l_dicedet expA23_d_deeplabv3p expA23_e_convnext_xl_384"
echo "===== A) 在庫のみ・修正後 (315) ====="
CUDA_VISIBLE_DEVICES=3 $PY t3_ensemble.py --features features_candE_fix.csv --encoders $E --seeds 3 --lgb-seeds 3 --tag ens_fix
echo "===== B) 在庫(修正) + 解剖文脈 + 粗グリッド ====="
CUDA_VISIBLE_DEVICES=3 $PY t3_ensemble.py --features features_candE_fix.csv \
  --extra-features features_anat_candE.csv features_grid_candE.csv \
  --encoders $E --seeds 3 --lgb-seeds 3 --tag ens_fixfull
echo "===== C) 参考: 旧(バグ有り)在庫 ====="
CUDA_VISIBLE_DEVICES=3 $PY t3_ensemble.py --features features_candE.csv --encoders $E --seeds 3 --lgb-seeds 3 --tag ens_old
