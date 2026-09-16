"""expT04: 予測マスクそのものを縮小して LGBM に渡す特徴（空間レイアウト）.

在庫特徴は「全体で何がどれだけ」しか見ないので、**どこに何があるか**の情報が
重心と bbox にしか残らない。ここではマスクを粗いグリッドに落とし、
セルごとのクラス占有率をそのまま特徴にする（= downsample したマスク）。

  grid G×2G（既定 6×12）× coarse 15 クラス = 1080 次元（面積比）
  さらに各セルの最頻クラス（カテゴリとして LGBM に渡せるよう整数）

Usage:
    python3 features_grid.py --pred workspace/expE01_ensemble/results/candB_new7 --tag candB --grid 6
出力: features_grid_<tag>.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]


def build_luts():
    lm = pd.read_csv(REPO / "data/labelmap.csv")
    lf = np.zeros(256 ** 3, np.uint8); lc = np.zeros(256 ** 3, np.uint8)
    for _, r in lm.iterrows():
        lf[int(r.fine_r) * 65536 + int(r.fine_g) * 256 + int(r.fine_b)] = int(r.fine_id)
        lc[int(r.merged_r) * 65536 + int(r.merged_g) * 256 + int(r.merged_b)] = int(r.merged_id)
    return lf, lc


def decode_small(p, lut, gh, gw):
    b = cv2.imread(str(p), cv2.IMREAD_COLOR)
    assert b is not None, p
    s = cv2.resize(b, (gw * 8, gh * 8), interpolation=cv2.INTER_NEAREST)
    rgb = s[:, :, ::-1].astype(np.uint32)
    return lut[rgb[:, :, 0] * 65536 + rgb[:, :, 1] * 256 + rgb[:, :, 2]]


def grid_features(lab, n_cls, gh, gw, prefix):
    """各セルのクラス占有率（n_cls 次元）と最頻クラス。"""
    out = {}
    H, W = lab.shape
    ch, cw = H // gh, W // gw
    for i in range(gh):
        for j in range(gw):
            cell = lab[i * ch:(i + 1) * ch, j * cw:(j + 1) * cw].ravel()
            cnt = np.bincount(cell, minlength=n_cls).astype(np.float32) / max(len(cell), 1)
            for c in range(n_cls):
                out[f"{prefix}_g{i}_{j}_{c}"] = float(cnt[c])
            out[f"{prefix}_gm{i}_{j}"] = int(cnt.argmax())
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred", default="workspace/expE01_ensemble/results/candB_new7")
    ap.add_argument("--tag", default="candB")
    ap.add_argument("--grid", type=int, default=6, help="縦セル数（横はその 2 倍）")
    args = ap.parse_args()
    root = REPO / args.pred if not Path(args.pred).is_absolute() else Path(args.pred)
    gh, gw = args.grid, args.grid * 2
    lf, lc = build_luts()
    rows = []
    files = sorted((root / "task1").glob("*.png"))
    for i, f in enumerate(files):
        r = {"case_id": f.stem}
        r.update(grid_features(decode_small(root / "task2" / f.name, lc, gh, gw), 16, gh, gw, "c"))
        rows.append(r)
        if i % 100 == 0:
            print(i, len(files), flush=True)
    df = pd.DataFrame(rows)
    out = HERE / f"features_grid_{args.tag}.csv"
    df.to_csv(out, index=False)
    print(df.shape, "->", out)


if __name__ == "__main__":
    main()
