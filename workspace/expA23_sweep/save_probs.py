"""expA23: あるレシピの **OOF 平均 softmax 確率**を half 解像度 fp16 で保存する.

これがあると以下が **GPU なし・再推論なし**でできる:
  * アンサンブルの組合せ探索（メンバーの確率を足すだけ）
  * クラス別スケーリング係数 α の後処理探索（`workspace/analysis/search_postproc*.py` と同じ手続き）

`workspace/analysis/save_oof_probs.py` の expA23 版。あちらは ens5 固定だったが、
こちらは config 名でレシピを指定でき、fold ごとに「その fold の val 画像」を
その fold のモデルで推論する（= OOF）。

解像度: 288x512（原寸の約 1/4 面積）。α もアンサンブル重みも解像度に依存しないので、
探索は half で行い、適用は原寸で行える。

Usage:
    python3 save_probs.py --config configs/expA23_l_dicedet.yaml
出力: results/<name>/probs/{fine,coarse}/<image>.npy
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import yaml

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent))

from dataset import IMAGENET_MEAN, IMAGENET_STD  # noqa: E402
from predict_oof import load_model  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
log = logging.getLogger("probs")
OUT_H, OUT_W = 288, 512


@torch.no_grad()
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--folds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text())
    name = cfg["experiment"]["name"]
    res_root = REPO / cfg["paths"]["results_root"] / name
    out_f = res_root / "probs" / "fine"
    out_c = res_root / "probs" / "coarse"
    out_f.mkdir(parents=True, exist_ok=True)
    out_c.mkdir(parents=True, exist_ok=True)

    folds = pd.read_csv(REPO / cfg["cv"]["folds_csv"])
    h, w = cfg["data"]["img_h"], cfg["data"]["img_w"]
    mean = torch.tensor(IMAGENET_MEAN, device=args.device).view(1, 3, 1, 1)
    std = torch.tensor(IMAGENET_STD, device=args.device).view(1, 3, 1, 1)

    n_total = 0
    for fold in args.folds:
        fold_dir = res_root / f"fold{fold}"
        ckpt = fold_dir / "best.ckpt"
        if not ckpt.exists():
            ckpt = fold_dir / "best_fp16.pt"
        if not ckpt.exists():
            log.warning("fold%d: ckpt が無いのでスキップ (%s)", fold, fold_dir)
            continue
        model = load_model(cfg, ckpt, args.device)
        val_df = folds[folds.fold == fold]
        for _, row in val_df.iterrows():
            bgr = cv2.imread(str(REPO / cfg["paths"]["orig_images_dir"] / row.filename))
            rgb = cv2.cvtColor(cv2.resize(bgr, (w, h), interpolation=cv2.INTER_AREA),
                               cv2.COLOR_BGR2RGB)
            x = torch.from_numpy(rgb).permute(2, 0, 1)[None].float().to(args.device) / 255.0
            x = (x - mean) / std
            with torch.autocast("cuda", dtype=torch.float16):
                lf, lc = model(x)
            for logit, out_dir in ((lf, out_f), (lc, out_c)):
                p = F.interpolate(logit.float().softmax(1), size=(OUT_H, OUT_W),
                                  mode="bilinear", align_corners=False)[0]
                np.save(out_dir / f"{row.filename.rsplit('.', 1)[0]}.npy",
                        p.cpu().numpy().astype(np.float16))
            n_total += 1
        del model
        torch.cuda.empty_cache()
        log.info("fold%d: %d 枚保存", fold, len(val_df))
    log.info("合計 %d 枚 -> %s", n_total, res_root / "probs")


if __name__ == "__main__":
    main()
