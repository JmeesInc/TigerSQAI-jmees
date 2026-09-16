"""TigerSQAI Task1+2+3 inference container entrypoint (v005).

v004 からの修正 2 点 (2026-09-08 の公式 Docker Instructions 精読で判明した仕様誤認):
  1. **出力先のスワップ**: 公式定義は Task1 = merged/coarse (15+bg), Task2 = fine (30+bg)。
     v001〜v004 は fine→task1/, merged→task2/ と逆に書いており色テーブル不一致になっていた。
  2. **Task3 の station 入力を撤廃**: 「Do not use the station information to get a prediction」。
     expT01 系ヘッド (station one-hot 入力) を expT03 (画像特徴のみ) に差し替えた。
     expT03 の finetune 版は dl2 で OOM 未完のため、**凍結ヘッド 5 fold 平均のみ**。

Source:
  Task1/2: 5 レシピ x 5 fold = 25 モデルの softmax 平均 (expE01 ens5)
           model/       expA06_f2c_loss   (Task3 の凍結ヘッドもこの encoder を共有する)
           model_extra/ expA09_lymph_aux / expA10_toolmask / expA11_v2data / expA05_maxvit
           CV (公式 5fold OOF, 526枚/42case): coarse 0.6731/0.2351, fine 0.6820/0.2383
           → 公式番号では **Task1(coarse) 0.6731/0.2351, Task2(fine) 0.6820/0.2383**
  Task3:   workspace/expT03_task3_noleak/results 凍結ヘッド 5 モデル平均
           CV (公式 evaluate_cls, 5fold OOF 518行): F1@0.5 0.7070 / AUROC 0.8867

注意: Task3 の凍結ヘッドは expA06 の encoder 特徴に対して学習されているため、
      model/ の 5 モデルとのみ 1:1 で対応する。model_extra/ は seg 平均にのみ寄与する。

出力: /output/task1/<name>.png (merged 16色), /output/task2/<name>.png (fine 31色),
      /output/task3.csv (case_id,6L,...,13L 全入力フレーム分の確率)
"""

from __future__ import annotations

import os
import sys
import time

import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from model_def import DualHeadUnetPP

INPUT_DIR = os.environ.get("INPUT_DIR", "/input")
OUTPUT_DIR = os.environ.get("OUTPUT_DIR", "/output")
BASE = os.path.dirname(__file__)
MODEL_DIR = os.environ.get("MODEL_DIR", os.path.join(BASE, "model"))
MODEL_EXTRA_DIR = os.environ.get("MODEL_EXTRA_DIR", os.path.join(BASE, "model_extra"))
MODEL_T3_DIR = os.environ.get("MODEL_T3_DIR", os.path.join(BASE, "model_t3"))
LABELMAP = os.path.join(BASE, "labelmap.csv")

IMG_H, IMG_W = 576, 1024
ENCODER = "tu-maxvit_base_tf_512.in21k_ft_in1k"
FOLDS = [int(x) for x in os.environ.get("FOLDS", "0,1,2,3,4").split(",")]
MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)
STATIONS = ["6L", "6R", "7L", "7R", "8", "9", "10L", "10R", "11L", "11R", "12L", "12R", "13L", "13R"]


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def make_head() -> nn.Sequential:
    """expT03 の Task3 ヘッド: encoder 最終特徴 GAP(768) のみを入力とする (station 情報は不使用)."""
    return nn.Sequential(
        nn.Linear(768, 256), nn.ReLU(inplace=True),
        nn.Dropout(0.3), nn.Linear(256, len(STATIONS)),
    )


def build_seg() -> "DualHeadUnetPP":
    return DualHeadUnetPP(encoder_name=ENCODER, encoder_weights=None,
                          num_classes_fine=31, num_classes_coarse=16, img_size=(IMG_H, IMG_W))


def load_extra(device: str) -> list:
    """model_extra/ の追加レシピ (seg 平均にのみ使う)。存在しなければ空."""
    out = []
    if not os.path.isdir(MODEL_EXTRA_DIR):
        return out
    for fn in sorted(f for f in os.listdir(MODEL_EXTRA_DIR) if f.endswith(".pt")):
        m = build_seg()
        m.load_state_dict(torch.load(os.path.join(MODEL_EXTRA_DIR, fn), map_location="cpu"),
                          strict=True)
        out.append(m.half().to(device).eval())
        log(f"extra model loaded: {fn}")
    return out


def load_all(device: str):
    seg_models, frozen_heads = [], []
    for f in FOLDS:
        m = build_seg()
        m.load_state_dict(torch.load(os.path.join(MODEL_DIR, f"fold{f}.pt"), map_location="cpu"), strict=True)
        seg_models.append(m.half().to(device).eval())

        h = make_head()
        hs = torch.load(os.path.join(MODEL_T3_DIR, f"frozen_head_fold{f}.pt"), map_location="cpu")
        h.load_state_dict({k.removeprefix("head."): v for k, v in hs.items()}, strict=True)
        frozen_heads.append(h.half().to(device).eval())
        log(f"fold{f} models loaded")
    return seg_models, frozen_heads


def build_id2rgb(lm: pd.DataFrame):
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
    seg_models, frozen_heads = load_all(device)
    extra_models = load_extra(device)
    log(f"seg ensemble: {len(seg_models)} (+{len(extra_models)} extra) = "
        f"{len(seg_models) + len(extra_models)} models / task3 heads: {len(frozen_heads)}")
    if device == "cpu":  # CPU では half 未対応 op があるため float 化
        seg_models = [m.float() for m in seg_models]
        extra_models = [m.float() for m in extra_models]
        frozen_heads = [h.float() for h in frozen_heads]

    names = sorted(n for n in os.listdir(INPUT_DIR) if n.lower().endswith(".png"))
    log(f"{len(names)} input frames")
    assert names, f"no PNG inputs in {INPUT_DIR}"

    t3_rows = []
    for i, name in enumerate(names):
        stem = name.rsplit(".", 1)[0]
        bgr = cv2.imread(os.path.join(INPUT_DIR, name), cv2.IMREAD_COLOR)
        assert bgr is not None, f"failed to read {name}"
        oh_, ow_ = bgr.shape[:2]
        rgb = cv2.cvtColor(cv2.resize(bgr, (IMG_W, IMG_H), interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2RGB)
        x = (rgb.astype(np.float32) / 255.0 - MEAN) / STD
        x = torch.from_numpy(x.transpose(2, 0, 1))[None].to(device)
        if device == "cuda":
            x = x.half()

        probs_f = probs_c = None
        t3_probs = []
        for m, fh in zip(seg_models, frozen_heads):
            feats = m.encoder(x)
            lf = m.head_fine(m.decoder_fine(feats))
            lc = m.head_coarse(m.decoder_coarse(feats))
            pf, pc = lf.float().softmax(1), lc.float().softmax(1)
            probs_f = pf if probs_f is None else probs_f + pf
            probs_c = pc if probs_c is None else probs_c + pc
            g = feats[-1].mean(dim=(2, 3))
            t3_probs.append(torch.sigmoid(fh(g).float()))
        for m in extra_models:  # 追加レシピは seg 平均のみ (Task3 ヘッドは持たない)
            feats = m.encoder(x)
            pf = m.head_fine(m.decoder_fine(feats)).float().softmax(1)
            pc = m.head_coarse(m.decoder_coarse(feats)).float().softmax(1)
            probs_f = probs_f + pf
            probs_c = probs_c + pc
        p3 = torch.stack(t3_probs).mean(0)[0].cpu().numpy()
        t3_rows.append({"case_id": stem, **{s: float(p3[k]) for k, s in enumerate(STATIONS)}})

        # 公式定義: task1 = merged/coarse, task2 = fine
        for probs, id2rgb, sub in [(probs_c, id2rgb_coarse, "task1"), (probs_f, id2rgb_fine, "task2")]:
            up = torch.nn.functional.interpolate(probs, size=(oh_, ow_), mode="bilinear", align_corners=False)
            ids = up.argmax(1)[0].to(torch.uint8).cpu().numpy()
            ok = cv2.imwrite(os.path.join(OUTPUT_DIR, sub, name), cv2.cvtColor(id2rgb[ids], cv2.COLOR_RGB2BGR))
            assert ok, f"failed to write {sub}/{name}"
        if (i + 1) % 20 == 0 or i == len(names) - 1:
            log(f"{i + 1}/{len(names)} done")

    df = pd.DataFrame(t3_rows, columns=["case_id"] + STATIONS)
    assert list(df.columns) == ["case_id"] + STATIONS
    assert ((df[STATIONS] >= 0) & (df[STATIONS] <= 1)).all().all()
    df.to_csv(os.path.join(OUTPUT_DIR, "task3.csv"), index=False)

    for sub in ("task1", "task2"):
        n_out = len([n for n in os.listdir(os.path.join(OUTPUT_DIR, sub)) if n.endswith(".png")])
        assert n_out == len(names), f"{sub}: {n_out} != {len(names)}"
    assert len(df) == len(names)
    log("all outputs written and verified (task1=coarse / task2=fine / task3.csv)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
