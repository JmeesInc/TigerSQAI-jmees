"""expA23: **推論時だけ**入力解像度を上げると効くかを測る（再学習なし）.

学習は 576x1024 だが、ConvNeXt + DeepLabV3+/Unet++ は全層畳み込みなので他の解像度でも動く。
いくつかのコンペで「推論時に少しだけ解像度を上げると伸びる」例が報告されている。
weight=3 の細い構造（神経・動脈）を取りこぼしているのが本コンペ最大の失点源なので、
推論時の実効受容野／細部の保持が変わることで拾える可能性がある。

各スケールで val 画像を推論 → **原寸へ bilinear** → argmax → 公式規約の weighted Dice。
（提出パイプラインと同じ経路。島削除は別途なのでここでは掛けない）

Usage:
    python3 infer_scale.py --arm q_cholec_dlv3 --fold 0 --device cuda:0
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import yaml

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(REPO / "reference" / "tigersqai_challenge"))
from model import build_model  # noqa: E402

MEAN = np.array([0.485, 0.456, 0.406], np.float32)
STD = np.array([0.229, 0.224, 0.225], np.float32)
SCALES = [(512, 896), (544, 960), (576, 1024), (608, 1088)]


def build_lut(task: str):
    lm = pd.read_csv(REPO / "data/labelmap.csv")
    lut = np.zeros(256 ** 3, np.uint8)
    if task == "fine":
        from metrics.classes import CLASSES as C
        for _, r in lm.iterrows():
            lut[int(r.fine_r) * 65536 + int(r.fine_g) * 256 + int(r.fine_b)] = int(r.fine_id)
        gt_sub = "masks_fine"
    else:
        from metrics.classes_merged import CLASSES_MERGED as C
        for _, r in lm.iterrows():
            lut[int(r.merged_r) * 65536 + int(r.merged_g) * 256 + int(r.merged_b)] = int(r.merged_id)
        gt_sub = "masks_coarse"
    ids = np.array([c.label_id for c in C]); w = np.array([float(c.weight) for c in C])
    return lut, ids, w, int(ids.max()) + 1, gt_sub


def decode(p, lut):
    b = cv2.imread(str(p), cv2.IMREAD_COLOR)
    rgb = b[:, :, ::-1].astype(np.uint32)
    return lut[rgb[:, :, 0] * 65536 + rgb[:, :, 1] * 256 + rgb[:, :, 2]]


def wdice(pred, gt, ids, w, n):
    idx = (gt.astype(np.uint32) * n + pred).ravel()
    cm = np.bincount(idx, minlength=n * n).reshape(n, n)
    tp = np.diag(cm).astype(float); ng, npd = cm.sum(1).astype(float), cm.sum(0).astype(float)
    b0 = (ng == 0) & (npd == 0); o0 = ((ng == 0) | (npd == 0)) & ~b0
    d = 2 * tp / np.maximum(ng + npd, 1)
    d = np.where(b0, 1.0, np.where(o0, 0.0, d))
    return float((d[ids] * w).sum() / w.sum())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", default="q_cholec_dlv3")
    ap.add_argument("--fold", type=int, default=0)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    cfg = yaml.safe_load(open(HERE / f"configs/expA23_{args.arm}.yaml"))
    res = HERE / "results" / f"expA23_{args.arm}"
    ck = None
    for cand in [res / f"fold{args.fold}"] + sorted(res.glob(f"fold{args.fold}_0*")):
        for name in ("best_fp16.pt", "last_fp16.pt"):
            if (cand / name).exists():
                ck = cand / name; break
        if ck:
            break
    assert ck, f"重みが無い: {res}/fold{args.fold}"
    m = dict(cfg["model"]); m["encoder_weights"] = None
    m.pop("init_from", None); m.pop("init_from_partial", None)
    net = build_model(m, (cfg["data"]["img_h"], cfg["data"]["img_w"]))
    sd = torch.load(ck, map_location="cpu")
    sd = {k.removeprefix("model."): v.float() for k, v in sd.items() if k.startswith("model.")} or \
         {k: v.float() for k, v in sd.items()}
    net.load_state_dict(sd, strict=True)
    net = net.half().to(args.device).eval()
    print(f"{args.arm} fold{args.fold}: {ck.name}", flush=True)

    folds = pd.read_csv(REPO / "workspace/fold/v2/folds.csv")
    val = folds[folds.fold == args.fold]
    if args.limit:
        val = val.head(args.limit)
    T = {t: build_lut(t) for t in ("fine", "coarse")}
    out = {}
    for (ih, iw) in SCALES:
        acc = {"fine": {}, "coarse": {}}
        for _, row in val.iterrows():
            bgr = cv2.imread(str(REPO / "data/images" / row.filename))
            oh, ow = bgr.shape[:2]
            rgb = cv2.cvtColor(cv2.resize(bgr, (iw, ih), interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2RGB)
            x = (rgb.astype(np.float32) / 255.0 - MEAN) / STD
            x = torch.from_numpy(x.transpose(2, 0, 1))[None].to(args.device).half()
            with torch.no_grad():
                lf, lc = net(x)
            case = row.filename.rsplit(".", 1)[0].rsplit("_", 1)[0]
            for task, lg in (("fine", lf), ("coarse", lc)):
                lut, ids, w, n, gt_sub = T[task]
                up = F.interpolate(lg.float(), size=(oh, ow), mode="bilinear", align_corners=False)
                pred = up.argmax(1)[0].to(torch.uint8).cpu().numpy()
                g = decode(REPO / "data" / gt_sub / row.filename, lut)
                acc[task].setdefault(case, []).append(wdice(pred, g, ids, w, n))
        sc = {t: float(np.mean([np.mean(v) for v in acc[t].values()])) for t in acc}
        sc["mean"] = (sc["fine"] + sc["coarse"]) / 2
        out[f"{ih}x{iw}"] = {k: round(v, 4) for k, v in sc.items()}
        print(f"{ih}x{iw}: fine {sc['fine']:.4f}  coarse {sc['coarse']:.4f}  平均 {sc['mean']:.4f}", flush=True)
    (HERE / f"infer_scale_{args.arm}_f{args.fold}.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
