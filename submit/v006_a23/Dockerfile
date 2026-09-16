FROM pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime

RUN pip install --no-cache-dir \
    segmentation-models-pytorch==0.5.0 \
    timm==1.0.22 \
    opencv-python-headless \
    pandas \
    lightgbm==4.7.0 \
    xgboost==3.2.0 \
    scikit-learn

WORKDIR /app
# v014: v13 + 後処理修正（alpha.json は .dockerignore で除外、island_ppm.json = クラス別 2%）
COPY model_v13/ /app/model/
# v012: Task3 を XGBoost vector-leaf(1741) x3 + MLP[d_deeplabv3p](GAP+315) x3, w=0.7, λ=1.0 に差し替え
#       (OOF 0.8056/0.9253, v7 構成 0.7890/0.9165)。encoder は d_deeplabv3p の 5 fold のみ
COPY model_t3_v12/ /app/model_t3/
COPY model_t3enc_v12/ /app/model_t3enc/
COPY process.py model_def.py t3_features.py labelmap.csv /app/

# ネット遮断環境で timm/HF に触れないように
ENV ISLAND_PPM=500 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONUNBUFFERED=1 TORCH_HOME=/opt/checkpoints

CMD ["python3", "process.py"]
