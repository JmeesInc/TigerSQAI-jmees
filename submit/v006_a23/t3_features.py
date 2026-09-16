"""v006: Task3 用「解剖在庫」特徴（学習側 workspace/expT04_task3_sweep/features_from_masks.py と
**完全に同じ計算**でなければならない）.

学習側は RGB 予測 PNG を 1/SUB に間引いてから LUT でクラス ID 化していたが、
ここではアンサンブルの argmax ID マップを同じ間引き方で受け取る（LUT は画素独立なので等価）。

一致は `verify_features.py` で実マスクを使って突き合わせて確認する。
"""

from __future__ import annotations

import cv2
import numpy as np

SUB = 4


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


# ---------------------------------------------------------------- 解剖文脈 / 粗グリッド
# 学習側 features_anatomy.py / features_grid.py と **同一の関数**（verify_features.py で突合）
LN_FINE = 19
FAT_FINE = 20
ANCHORS = [3, 4, 5, 6, 11, 12, 15, 16, 17, 18, 23]


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


def all_features(fine: np.ndarray, coarse: np.ndarray, grid: int = 6) -> dict:
    """在庫 + 解剖文脈 + 粗グリッド をまとめて返す（列順は学習側の CSV 結合順に合わせる）。

    fine/coarse は **1/SUB に間引いた** クラス ID マップ（在庫・解剖文脈用）。
    グリッドは別解像度（gh*8 x gw*8）で取るので呼び出し側で渡す。
    """
    out = {}
    out.update(mask_features(fine, 31, "f"))
    out.update(mask_features(coarse, 16, "c"))
    out.update(contact_features(coarse, 16, "c"))
    out.update(surround_features(fine, LN_FINE, 31, "ln"))
    out.update(surround_features(fine, FAT_FINE, 31, "fat"))
    out.update(proximity_features(fine, "p"))
    return out
