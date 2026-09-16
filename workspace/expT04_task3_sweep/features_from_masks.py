"""expT04: セグメンテーション予測マスクから Task3 用の「解剖在庫」特徴を作る.

9/11 の調査で、**個々のリンパ節成分の局所近傍から station は同定できない** (0.269 ≒ チャンス)
一方で **フレーム全体の構造在庫は情報を持つ (0.344)** ことが分かっている。
ここではその「在庫」を素性として明示的に取り出す。

入力は **ens5 の OOF 予測マスク** (= その画像の case を学習に含まないモデルの予測)。
GT マスクを使うとテスト時に存在しない情報になるので必ず予測側を使う。

特徴 (fine 31 クラス / coarse 16 クラス それぞれ):
  - 面積比 (画素数 / 全画素)
  - 存在フラグ
  - 重心 x, y (画像サイズで正規化, 不在なら -1)
  - 連結成分数 (log1p)
  - bbox の幅・高さ (正規化)

Usage:
    python3 features_from_masks.py --pred workspace/expE01_ensemble/results/ens5
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
SUB = 4  # 1/4 に縮小してから集計 (面積比・重心は不変、成分数はノイズが減る)


def build_luts(labelmap: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """(R,G,B) -> fine_id / merged_id の 1600 万エントリ LUT。"""
    lut_f = np.zeros(256 ** 3, dtype=np.uint8)
    lut_c = np.zeros(256 ** 3, dtype=np.uint8)
    for _, r in labelmap.iterrows():
        lut_f[int(r.fine_r) * 65536 + int(r.fine_g) * 256 + int(r.fine_b)] = int(r.fine_id)
        lut_c[int(r.merged_r) * 65536 + int(r.merged_g) * 256 + int(r.merged_b)] = int(r.merged_id)
    return lut_f, lut_c


def mask_features(lab: np.ndarray, n_classes: int, prefix: str) -> dict:
    h, w = lab.shape
    n_pix = h * w
    out = {}
    for c in range(1, n_classes):  # 背景 0 は除く
        m = (lab == c).astype(np.uint8)
        area = int(m.sum())
        out[f"{prefix}_area_{c}"] = area / n_pix
        out[f"{prefix}_has_{c}"] = float(area > 0)
        if area == 0:
            out[f"{prefix}_cx_{c}"] = -1.0
            out[f"{prefix}_cy_{c}"] = -1.0
            out[f"{prefix}_ncomp_{c}"] = 0.0
            out[f"{prefix}_bw_{c}"] = 0.0
            out[f"{prefix}_bh_{c}"] = 0.0
            continue
        ys, xs = np.nonzero(m)
        out[f"{prefix}_cx_{c}"] = float(xs.mean() / w)
        out[f"{prefix}_cy_{c}"] = float(ys.mean() / h)
        n_comp, _ = cv2.connectedComponents(m, connectivity=8)
        out[f"{prefix}_ncomp_{c}"] = float(np.log1p(n_comp - 1))
        out[f"{prefix}_bw_{c}"] = float((xs.max() - xs.min() + 1) / w)
        out[f"{prefix}_bh_{c}"] = float((ys.max() - ys.min() + 1) / h)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred", default="workspace/expE01_ensemble/results/ens5")
    ap.add_argument("--out", default=str(HERE / "features_ens5.csv"))
    args = ap.parse_args()

    labelmap = pd.read_csv(REPO / "data" / "labelmap.csv")
    lut_f, lut_c = build_luts(labelmap)
    n_fine = int(labelmap.fine_id.max()) + 1
    n_coarse = int(labelmap.merged_id.max()) + 1

    pred = REPO / args.pred
    # 【重要】このリポジトリの OOF 出力は task1 = fine 色 / task2 = coarse 色
    # （eval_dice_fullres.py・island_postproc.py と同じ慣習）。提出コンテナの
    # task1=coarse / task2=fine とは逆なので取り違えないこと。
    files = sorted((pred / "task1").glob("*.png"))
    assert files, f"予測マスクが無い: {pred}/task1"
    rows = []
    for i, f in enumerate(files):
        rec = {"case_id": f.stem}
        for task, lut, n in (("task1", lut_f, n_fine), ("task2", lut_c, n_coarse)):
            p = pred / task / f.name
            if not p.exists():
                continue
            rgb = cv2.cvtColor(cv2.imread(str(p)), cv2.COLOR_BGR2RGB)[::SUB, ::SUB]
            idx = (rgb[..., 0].astype(np.int64) * 65536 + rgb[..., 1].astype(np.int64) * 256
                   + rgb[..., 2].astype(np.int64))
            lab = lut[idx]
            rec.update(mask_features(lab, n, "f" if task == "task1" else "c"))
        rows.append(rec)
        if (i + 1) % 100 == 0:
            print(f"{i + 1}/{len(files)}", file=sys.stderr, flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(args.out, index=False)
    print(f"{len(df)} frames x {df.shape[1] - 1} features -> {args.out}")


if __name__ == "__main__":
    main()
