"""expT04: 凍結 encoder の GAP 特徴を **fold ごとに全画像分** 先に計算してキャッシュする.

なぜ必要か:
  ヘッドだけを学習するのに毎 epoch encoder を forward していたため、
  他ジョブと GPU を共有すると 10 分/epoch まで落ちて実用にならなかった。
  encoder は凍結なので **特徴は 1 回計算すれば足りる**。

OOF の正しさ:
  fold k のモデルは fold k の case を学習に含まない。
  そこで **fold k のモデルで全画像の特徴を作り**、
  ヘッドの学習には fold!=k の行、評価には fold==k の行だけを使う。
  こうすれば評価対象の行は必ず「その case を見ていない encoder」の特徴になる。

aug:
  水平反転の有無 2 通りだけキャッシュする（明るさ変更は特徴空間では効きが小さい）。

Usage:
    python3 cache_feats.py --encoder expA23_l_dicedet --device cuda:1
出力: feats/<encoder>/fold{k}.npz  (keys: names, feat, feat_flip)
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

REPO = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from train_head import build_encoder, IMG_H, IMG_W, MEAN, STD  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
log = logging.getLogger("cache")


@torch.no_grad()
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--encoder", default="expA23_l_dicedet")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--folds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    args = ap.parse_args()

    gt = pd.read_csv(REPO / "workspace/data_proc/task3_gt_wide.csv")
    names = gt.case_id.tolist()
    img_dir = REPO / "workspace/data_proc/images_1024"
    out_dir = HERE / "feats" / args.encoder
    out_dir.mkdir(parents=True, exist_ok=True)

    for fold in args.folds:
        enc, ch = build_encoder(args.encoder, fold, args.device)
        feats, feats_flip = [], []
        for i, n in enumerate(names):
            img = cv2.cvtColor(cv2.imread(str(img_dir / f"{n}.png")), cv2.COLOR_BGR2RGB)
            for flip, store in ((False, feats), (True, feats_flip)):
                a = img[:, ::-1].copy() if flip else img
                x = (a.astype(np.float32) / 255.0 - MEAN) / STD
                x = torch.from_numpy(x.transpose(2, 0, 1))[None].to(args.device)
                with torch.autocast("cuda", torch.float16):
                    f = enc(x)[-1]
                store.append(f.float().mean((2, 3))[0].cpu().numpy())
            if (i + 1) % 200 == 0:
                log.info("fold%d %d/%d", fold, i + 1, len(names))
        np.savez_compressed(out_dir / f"fold{fold}.npz", names=np.array(names),
                            feat=np.stack(feats).astype(np.float32),
                            feat_flip=np.stack(feats_flip).astype(np.float32))
        log.info("fold%d 保存: %s 次元", fold, feats[0].shape[0])
        del enc
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
