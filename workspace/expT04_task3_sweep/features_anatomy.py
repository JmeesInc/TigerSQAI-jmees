"""expT04: Task3 用の「解剖学的な文脈」特徴を予測マスクから作る（AnatomyLoss の Task3 版）.

在庫特徴（面積・存在・重心・成分数・bbox）は「何が写っているか」しか見ていない。
station の可視性は **リンパ節が どの構造の隣にあるか** で決まるので、ここでは

  1. 隣接（接触）行列: 4 近傍で接するクラス対を 0/1 と接触長で表す
     → 解剖グラフ（AnatomyLoss）が使うのと同じ量を、今度は分類の入力にする
  2. 背景側（周囲）クラス: リンパ節・脂肪の各成分について、その輪郭を占める
     「隣のクラス」の割合（= その塊が何に囲まれているか）
  3. target ごとの近さ: 各クラスの重心・最近傍距離をリンパ節基準で測る
     （station ラベルは使わない。純粋に構造間の近傍性）

出力: features_anat_<tag>.csv （case_id + 特徴）
Usage:
    python3 features_anatomy.py --pred workspace/expE01_ensemble/results/candB_new7 --tag candB
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
SUB = 4                      # 1/4 に縮小してから集計
LN_FINE = 19                 # Lymph node (fine)
FAT_FINE = 20                # Fatty tissue
# 近傍性を測る相手（術野の目印になる大きめの構造）
ANCHORS = [3, 4, 5, 6, 11, 12, 15, 16, 17, 18, 23]   # 気管/主気管支/食道/心膜/下肺静脈/大動脈/奇静脈/上大静脈/肺/肺動脈


def build_luts():
    lm = pd.read_csv(REPO / "data/labelmap.csv")
    lf = np.zeros(256 ** 3, np.uint8); lc = np.zeros(256 ** 3, np.uint8)
    for _, r in lm.iterrows():
        lf[int(r.fine_r) * 65536 + int(r.fine_g) * 256 + int(r.fine_b)] = int(r.fine_id)
        lc[int(r.merged_r) * 65536 + int(r.merged_g) * 256 + int(r.merged_b)] = int(r.merged_id)
    return lf, lc


def decode(p, lut):
    b = cv2.imread(str(p), cv2.IMREAD_COLOR)
    assert b is not None, p
    s = cv2.resize(b, (b.shape[1] // SUB, b.shape[0] // SUB), interpolation=cv2.INTER_NEAREST)
    rgb = s[:, :, ::-1].astype(np.uint32)
    return lut[rgb[:, :, 0] * 65536 + rgb[:, :, 1] * 256 + rgb[:, :, 2]]


def contact_features(lab: np.ndarray, n: int, prefix: str) -> dict:
    """4 近傍で接するクラス対の接触長（正規化）。上三角のみ。"""
    pairs = {}
    for a, b in ((lab[:, :-1], lab[:, 1:]), (lab[:-1, :], lab[1:, :])):
        d = a != b
        x, y = a[d].astype(np.int64), b[d].astype(np.int64)
        lo, hi = np.minimum(x, y), np.maximum(x, y)
        k = lo * n + hi
        for kk, c in zip(*np.unique(k, return_counts=True)):
            pairs[int(kk)] = pairs.get(int(kk), 0) + int(c)
    tot = max(sum(pairs.values()), 1)
    out = {}
    for i in range(1, n):
        for j in range(i + 1, n):
            v = pairs.get(i * n + j, 0)
            out[f"{prefix}_ct_{i}_{j}"] = v / tot
    return out


def surround_features(lab: np.ndarray, target: int, n: int, prefix: str) -> dict:
    """target クラスの各成分が「何に囲まれているか」の割合（背景側クラスの分布）。"""
    out = {f"{prefix}_sur_{c}": 0.0 for c in range(n)}
    out[f"{prefix}_n"] = 0.0
    m = (lab == target).astype(np.uint8)
    if m.sum() == 0:
        return out
    nl, cc, stats, _ = cv2.connectedComponentsWithStats(m, 8)
    tot = 0
    for i in range(1, nl):
        if stats[i, cv2.CC_STAT_AREA] < 4:
            continue
        comp = (cc == i).astype(np.uint8)
        ring = (cv2.dilate(comp, np.ones((5, 5), np.uint8)) - comp) > 0
        vals = lab[ring]
        vals = vals[vals != target]
        if len(vals) == 0:
            continue
        for c, cnt in zip(*np.unique(vals, return_counts=True)):
            out[f"{prefix}_sur_{int(c)}"] += int(cnt)
        tot += len(vals)
        out[f"{prefix}_n"] += 1
    if tot:
        for c in range(n):
            out[f"{prefix}_sur_{c}"] /= tot
    return out


def proximity_features(lab: np.ndarray, prefix: str) -> dict:
    """リンパ節から各アンカー構造までの最短距離（画像対角で正規化, 不在は 1.0）と重心差。"""
    h, w = lab.shape
    diag = float(np.hypot(h, w))
    out = {}
    ln = (lab == LN_FINE)
    ln_c = np.array(np.nonzero(ln)).mean(1)[::-1] if ln.any() else None
    for c in ANCHORS:
        m = (lab == c)
        if not m.any() or not ln.any():
            out[f"{prefix}_d_{c}"] = 1.0
            out[f"{prefix}_dx_{c}"] = 0.0
            out[f"{prefix}_dy_{c}"] = 0.0
            continue
        dt = cv2.distanceTransform((~m).astype(np.uint8), cv2.DIST_L2, 3)
        out[f"{prefix}_d_{c}"] = float(dt[ln].min()) / diag
        cc = np.array(np.nonzero(m)).mean(1)[::-1]
        out[f"{prefix}_dx_{c}"] = float(cc[0] - ln_c[0]) / w
        out[f"{prefix}_dy_{c}"] = float(cc[1] - ln_c[1]) / h
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred", default="workspace/expE01_ensemble/results/candB_new7")
    ap.add_argument("--tag", default="candB")
    args = ap.parse_args()
    root = REPO / args.pred if not Path(args.pred).is_absolute() else Path(args.pred)
    lf, lc = build_luts()
    rows = []
    files = sorted((root / "task1").glob("*.png"))       # task1 = fine 色（リポジトリ慣習）
    assert files, root
    for i, f in enumerate(files):
        fine = decode(f, lf)
        coarse = decode(root / "task2" / f.name, lc)
        r = {"case_id": f.stem}
        r.update(contact_features(coarse, 16, "c"))                  # coarse の接触行列 (105 次元)
        r.update(surround_features(fine, LN_FINE, 31, "ln"))         # リンパ節の周囲 (32)
        r.update(surround_features(fine, FAT_FINE, 31, "fat"))       # 脂肪の周囲 (32)
        r.update(proximity_features(fine, "p"))                      # 近傍性 (33)
        rows.append(r)
        if i % 100 == 0:
            print(i, len(files), flush=True)
    df = pd.DataFrame(rows)
    out = HERE / f"features_anat_{args.tag}.csv"
    df.to_csv(out, index=False)
    print(df.shape, "->", out)


if __name__ == "__main__":
    main()
