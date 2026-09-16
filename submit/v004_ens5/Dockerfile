FROM pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime

RUN pip install --no-cache-dir \
    segmentation-models-pytorch==0.5.0 \
    timm==1.0.22 \
    opencv-python-headless \
    pandas

WORKDIR /app
COPY model/ /app/model/
COPY model_extra/ /app/model_extra/
COPY model_t3/ /app/model_t3/
COPY process.py model_def.py labelmap.csv /app/

# ネット遮断環境で timm が HF に触れないように
ENV HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1

CMD ["python3", "process.py"]
