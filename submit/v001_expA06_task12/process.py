"""TigerSQAI Task1+2 inference container entrypoint.

Source: workspace/expA06_f2c_loss (MaxViT-Base tf_512 + dual Unet++ + 強aug + f2c loss)
CV (公式 5fold OOF 全量): Task1 Dice 0.6663 / HD 0.2646, Task2 Dice 0.6530 / HD 0.2574

I/O 契約 (Synapse wiki 639935):
  入力  /input/*.png  (flat, 読み取り専用。test = 140 frames / 10 cases)
  出力  /output/task1/<同名>.png  (fine 31 色 RGB, 入力と同解像度)
        /output/task2/<同名>.png  (merged 16 色 RGB)
  task3 は出力しない (task12 提出)。引数なし・ネット遮断・exit 0 必須。

5 fold の平均 softmax アンサンブル。学習解像度 1024x576 で推論し、平均確率を
元解像度へ bilinear アップサンプル後 argmax → labelmap.csv の正確な RGB で PNG 出力。
"""

from __future__ import annotations

import os
import sys
import time

import cv2
import numpy as np
import pandas as pd
import torch

from model_def import DualHeadUnetPP

INPUT_DIR = os.environ.get("INPUT_DIR", "/input")
OUTPUT_DIR = os.environ.get("OUTPUT_DIR", "/output")
MODEL_DIR = os.environ.get("MODEL_DIR", os.path.join(os.path.dirname(__file__), "model"))
LABELMAP = os.path.join(os.path.dirname(__file__), "labelmap.csv")

IMG_H, IMG_W = 576, 1024
ENCODER = "tu-maxvit_base_tf_512.in21k_ft_in1k"
FOLDS = [0, 1, 2, 3, 4]
MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def load_models(device: str) -> list[torch.nn.Module]:
    models = []
    folds = [int(x) for x in os.environ.get("FOLDS", ",".join(map(str, FOLDS))).split(",")]
    for f in folds:
        m = DualHeadUnetPP(
            encoder_name=ENCODER,
            encoder_weights=None,  # 重みは同梱 ckpt から。ネットアクセスなし
            num_classes_fine=31,
            num_classes_coarse=16,
            img_size=(IMG_H, IMG_W),
        )
        state = torch.load(os.path.join(MODEL_DIR, f"fold{f}.pt"), map_location="cpu")
        m.load_state_dict(state, strict=True)
        m = m.half() if device == "cuda" else m.float()  # CPU は half 未サポートの op あり
        models.append(m.to(device).eval())
        log(f"fold{f} loaded")
    return models


def build_id2rgb(lm: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    fine = np.zeros((256, 3), dtype=np.uint8)
    coarse = np.zeros((256, 3), dtype=np.uint8)
    for r in lm.itertuples():
        fine[int(r.fine_id)] = (r.fine_r, r.fine_g, r.fine_b)
        coarse[int(r.merged_id)] = (r.merged_r, r.merged_g, r.merged_b)
    return fine, coarse


@torch.no_grad()
def main() -> int:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    log(f"device={device}")
    os.makedirs(os.path.join(OUTPUT_DIR, "task1"), exist_ok=True)
    os.makedirs(os.path.join(OUTPUT_DIR, "task2"), exist_ok=True)

    id2rgb_fine, id2rgb_coarse = build_id2rgb(pd.read_csv(LABELMAP))
    models = load_models(device)

    names = sorted(n for n in os.listdir(INPUT_DIR) if n.lower().endswith(".png"))
    log(f"{len(names)} input frames")
    assert names, f"no PNG inputs in {INPUT_DIR}"

    for i, name in enumerate(names):
        bgr = cv2.imread(os.path.join(INPUT_DIR, name), cv2.IMREAD_COLOR)
        assert bgr is not None, f"failed to read {name}"
        oh, ow = bgr.shape[:2]
        rgb = cv2.cvtColor(cv2.resize(bgr, (IMG_W, IMG_H), interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2RGB)
        x = (rgb.astype(np.float32) / 255.0 - MEAN) / STD
        x = torch.from_numpy(x.transpose(2, 0, 1))[None].to(device)
        if device == "cuda":
            x = x.half()

        probs_f = probs_c = None
        for m in models:
            lf, lc = m(x)
            pf = lf.float().softmax(1)
            pc = lc.float().softmax(1)
            probs_f = pf if probs_f is None else probs_f + pf
            probs_c = pc if probs_c is None else probs_c + pc

        for probs, id2rgb, sub in [(probs_f, id2rgb_fine, "task1"), (probs_c, id2rgb_coarse, "task2")]:
            up = torch.nn.functional.interpolate(probs, size=(oh, ow), mode="bilinear", align_corners=False)
            ids = up.argmax(1)[0].to(torch.uint8).cpu().numpy()
            out = cv2.cvtColor(id2rgb[ids], cv2.COLOR_RGB2BGR)
            ok = cv2.imwrite(os.path.join(OUTPUT_DIR, sub, name), out)
            assert ok, f"failed to write {sub}/{name}"
        if (i + 1) % 20 == 0 or i == len(names) - 1:
            log(f"{i + 1}/{len(names)} done")

    # 出力検証: 入力数 == 出力数
    for sub in ("task1", "task2"):
        n_out = len([n for n in os.listdir(os.path.join(OUTPUT_DIR, sub)) if n.endswith(".png")])
        assert n_out == len(names), f"{sub}: {n_out} != {len(names)}"
    log("all outputs written and verified")
    return 0


if __name__ == "__main__":
    sys.exit(main())
