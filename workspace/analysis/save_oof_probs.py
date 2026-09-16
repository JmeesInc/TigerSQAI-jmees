"""ens5 の OOF を再推論し, argmax 前の平均 softmax 確率を保存する.

後処理 (クラス別スケーリング / 最小面積フィルタ) を学習なしで探索するために必要。
容量を抑えるため half 解像度 (288x512) の fp16 で保存する。
スケーリング係数は解像度に依存しないので, 探索は half で行い適用は原寸で行える。
"""
from __future__ import annotations
import argparse, logging, sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "workspace/expE01_ensemble"))
from dataset import IMAGENET_MEAN, IMAGENET_STD          # noqa: E402
from predict_ens import MEMBERS, load_member             # noqa: E402

log = logging.getLogger("probs")


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--folds-csv", default="workspace/fold/v3/folds.csv")
    ap.add_argument("--out", default="workspace/analysis/oof_probs")
    ap.add_argument("--members", nargs="+", default=list(MEMBERS))
    ap.add_argument("--folds", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    ap.add_argument("--img-h", type=int, default=576)
    ap.add_argument("--img-w", type=int, default=1024)
    ap.add_argument("--save-h", type=int, default=288)
    ap.add_argument("--save-w", type=int, default=512)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    out = REPO / args.out
    (out / "fine").mkdir(parents=True, exist_ok=True)
    (out / "coarse").mkdir(parents=True, exist_ok=True)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s",
                        handlers=[logging.StreamHandler(), logging.FileHandler(out / "save.log")])

    folds = pd.read_csv(REPO / args.folds_csv)
    mean = torch.tensor(IMAGENET_MEAN, device=args.device).view(1, 3, 1, 1)
    std = torch.tensor(IMAGENET_STD, device=args.device).view(1, 3, 1, 1)
    img_dir = REPO / "workspace/data_proc/images_1024"

    for fold in args.folds:
        val = folds[folds.fold == fold]
        models = [load_member(m, fold, args.device).half() for m in args.members]
        log.info("fold %d: %d 枚 / %d モデル", fold, len(val), len(models))
        for k, row in enumerate(val.itertuples()):
            fp = out / "fine" / f"{row.filename[:-4]}.npy"
            if fp.exists():
                continue
            bgr = cv2.imread(str(img_dir / row.filename))
            if bgr is None:
                log.warning("読めない: %s", row.filename); continue
            rgb = cv2.cvtColor(cv2.resize(bgr, (args.img_w, args.img_h),
                                          interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2RGB)
            x = torch.from_numpy(rgb).permute(2, 0, 1)[None].float().to(args.device) / 255.0
            x = ((x - mean) / std).half()
            pf = pc = None
            for m in models:
                lf, lc = m(x)
                sf, sc = lf.float().softmax(1), lc.float().softmax(1)
                pf = sf if pf is None else pf + sf
                pc = sc if pc is None else pc + sc
            for p, sub in ((pf / len(models), "fine"), (pc / len(models), "coarse")):
                small = F.interpolate(p, size=(args.save_h, args.save_w),
                                      mode="bilinear", align_corners=False)[0]
                np.save(out / sub / f"{row.filename[:-4]}.npy",
                        small.cpu().numpy().astype(np.float16))
            if (k + 1) % 25 == 0:
                log.info("  fold %d: %d/%d", fold, k + 1, len(val))
        del models
        torch.cuda.empty_cache()
    log.info("完了 -> %s", out)


if __name__ == "__main__":
    main()
