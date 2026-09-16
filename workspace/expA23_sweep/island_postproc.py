"""expA23: 孤立島（小さい連結成分）の削除を後処理として検討する.

予測マスクには「本体から離れた数十〜数百画素の島」が出る。公式指標では
  * Dice: 小さい偽陽性は分母をわずかに増やすだけ（影響小）
  * 正規化 Hausdorff: **GT から最も遠い誤検出画素**で決まるので、遠方の島は致命的
  * さらに「GT に無いクラスを 1 画素でも出すと そのクラス 0 点」なので、
    *そのクラス唯一の予測が島だった場合*は削除で 0 点 → 1.0 点に跳ねる
逆に weight=3 の小さい真陽性（神経・動脈など）を消すと 1.0 → 0 点になるため、
クラスごとに効き方が違う。ここを実測する。

方式:
  A. 面積しきい値 T: 各クラスの連結成分のうち面積 < T を「周囲の多数決クラス」で埋める
  B. 上位 K 成分のみ残す（K は解剖グラフの K_max = GT 成分数の p90）
評価は原寸 weighted Dice（公式規約）。center を跨いで効くかも見る（leave-one-center-out）。

Usage:
    python3 island_postproc.py --pred workspace/expE01_ensemble/results/candB_new7
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(REPO / "reference" / "tigersqai_challenge"))


def build_lut(task: str):
    lm = pd.read_csv(REPO / "data/labelmap.csv")
    lut = np.zeros(256 ** 3, np.uint8)
    if task == "fine":
        from metrics.classes import CLASSES as C
        for _, r in lm.iterrows():
            lut[int(r.fine_r) * 65536 + int(r.fine_g) * 256 + int(r.fine_b)] = int(r.fine_id)
        gt_sub, sub = "masks_fine", "task1"
    else:
        from metrics.classes_merged import CLASSES_MERGED as C
        for _, r in lm.iterrows():
            lut[int(r.merged_r) * 65536 + int(r.merged_g) * 256 + int(r.merged_b)] = int(r.merged_id)
        gt_sub, sub = "masks_coarse", "task2"
    ids = np.array([c.label_id for c in C])
    w = np.array([float(c.weight) for c in C])
    return lut, ids, w, int(ids.max()) + 1, gt_sub, sub, {c.label_id: c.name for c in C}


def decode(p, lut):
    b = cv2.imread(str(p), cv2.IMREAD_COLOR)
    assert b is not None, p
    rgb = b[:, :, ::-1].astype(np.uint32)
    return lut[rgb[:, :, 0] * 65536 + rgb[:, :, 1] * 256 + rgb[:, :, 2]]


def wdice_per_class(p, g, n):
    idx = (g.astype(np.uint32) * n + p).ravel()
    cm = np.bincount(idx, minlength=n * n).reshape(n, n)
    tp = np.diag(cm).astype(float)
    ng, npd = cm.sum(1).astype(float), cm.sum(0).astype(float)
    b0 = (ng == 0) & (npd == 0)
    o0 = ((ng == 0) | (npd == 0)) & ~b0
    d = 2 * tp / np.maximum(ng + npd, 1)
    return np.where(b0, 1.0, np.where(o0, 0.0, d))


def remove_islands(lab: np.ndarray, thr: int, kmax: np.ndarray | None = None) -> np.ndarray:
    """面積 thr 未満の成分（と K_max 超過の小成分）を、周囲の多数決クラスで埋める。"""
    out = lab.copy()
    for c in np.unique(lab):
        if c == 0:
            continue
        m = (lab == c).astype(np.uint8)
        nlab, cc, stats, _ = cv2.connectedComponentsWithStats(m, 8)
        if nlab <= 1:
            continue
        areas = stats[1:, cv2.CC_STAT_AREA]
        drop = areas < thr
        if kmax is not None and c < len(kmax) and kmax[c] > 0 and (~drop).sum() > kmax[c]:
            keep_idx = np.argsort(-areas)[: int(kmax[c])]
            keep = np.zeros(len(areas), bool); keep[keep_idx] = True
            drop |= ~keep
        for i in np.where(drop)[0]:
            x, y, w_, h_, a = stats[i + 1]
            sl = (slice(max(y - 2, 0), y + h_ + 2), slice(max(x - 2, 0), x + w_ + 2))
            comp = cc[sl] == (i + 1)
            ring = cv2.dilate(comp.astype(np.uint8), np.ones((5, 5), np.uint8)) > 0
            ring &= ~comp
            vals = out[sl][ring]
            vals = vals[vals != c]
            out[sl][comp] = np.bincount(vals).argmax() if len(vals) else 0
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred", default="workspace/expE01_ensemble/results/candB_new7")
    ap.add_argument("--thresholds", type=int, nargs="+", default=[0, 64, 256, 1024, 4096])
    ap.add_argument("--relative", action="store_true",
                    help="しきい値を画像面積比 (ppm) として扱う。解像度混在 (1080p/4K) への頑健性を見る")
    ap.add_argument("--kmax", action="store_true", help="K_max による成分数制限も併用")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--out", default=str(HERE / "island_postproc.json"))
    args = ap.parse_args()

    pred_root = REPO / args.pred if not Path(args.pred).is_absolute() else Path(args.pred)
    res = {}
    for task in ("fine", "coarse"):
        lut, ids, w, n, gt_sub, sub, names = build_lut(task)
        kmax = None
        if args.kmax:
            z = np.load(REPO / f"workspace/anatomy_graph/out/rules_{task}.npz", allow_pickle=True)
            kmax = z["K_max"]
        files = sorted((pred_root / sub).glob("*.png"))
        if args.limit:
            files = files[: args.limit]
        # per-case × threshold の Dice と、クラス別の変化
        acc = {t: {} for t in args.thresholds}
        cls_delta = {t: np.zeros((len(ids), 2)) for t in args.thresholds}
        for j, f in enumerate(files):
            p0 = decode(f, lut)
            g = decode(REPO / "data" / gt_sub / f.name, lut)
            case = f.stem.rsplit("_", 1)[0]
            base = None
            for t in args.thresholds:
                tt = int(t * p0.size / 1_000_000) if args.relative else t
                p = p0 if (tt == 0 and not args.kmax) else remove_islands(p0, tt, kmax)
                d = wdice_per_class(p, g, n)
                if base is None:
                    base = d
                acc[t].setdefault(case, []).append(float((d[ids] * w).sum() / w.sum()))
                cls_delta[t][:, 0] += d[ids] - base[ids]
                cls_delta[t][:, 1] += 1
            if j % 100 == 0:
                print(f"{task} {j}/{len(files)}", flush=True)
        res[task] = {}
        for t in args.thresholds:
            case_mean = {k: float(np.mean(v)) for k, v in acc[t].items()}
            res[task][str(t)] = {"dice": round(float(np.mean(list(case_mean.values()))), 4),
                                 "per_case": {k: round(v, 4) for k, v in case_mean.items()}}
            worst = np.argsort(cls_delta[t][:, 0])[:5]
            best = np.argsort(-cls_delta[t][:, 0])[:5]
            res[task][str(t)]["cls_down"] = [[names[int(ids[i])], round(float(cls_delta[t][i, 0] / max(cls_delta[t][i, 1], 1)), 5)] for i in worst]
            res[task][str(t)]["cls_up"] = [[names[int(ids[i])], round(float(cls_delta[t][i, 0] / max(cls_delta[t][i, 1], 1)), 5)] for i in best]
            print(f"{task} T={t}: Dice={res[task][str(t)]['dice']:.4f}", flush=True)
    Path(args.out).write_text(json.dumps(res, ensure_ascii=False, indent=1))
    print("->", args.out)


if __name__ == "__main__":
    main()
