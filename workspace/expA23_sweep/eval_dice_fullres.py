"""expA23: 予測 PNG を **原寸のまま** 公式規約の weighted Dice で採点する（HD は計算しない）.

なぜ必要か:
  低解像度（288x512 や 144x256）で採点すると、weight=3 の細い構造が GT から消え、
  「両方空 = 1.0」規約のせいで **「出さないモデル」が系統的に得をする**。
  実際 dicedet と ens5 は解像度によって fine の順位が逆転した。
  → メンバー選択も原寸で行う必要がある。ただし HD（EDT）は重いので、
     **選択は Dice だけ原寸で**行い、最終候補にだけ公式フルパイプライン（HD 込み）を掛ける。

Usage:
    python3 eval_dice_fullres.py --pred workspace/expE01_ensemble/results/ens_xxx
    # pred 配下に task1/ (fine 色) と task2/ (coarse 色) がある前提（このリポジトリの慣習）
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "reference" / "tigersqai_challenge"))


def build_lut(task: str):
    lm = pd.read_csv(REPO / "data" / "labelmap.csv")
    lut = np.zeros(256 ** 3, dtype=np.uint8)
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
    ids = np.array([c.label_id for c in C])
    w = np.array([float(c.weight) for c in C])
    n_lab = int(ids.max()) + 1
    return lut, ids, w, n_lab, gt_sub


def decode(path: Path, lut: np.ndarray) -> np.ndarray:
    """4K 画像でもメモリを食わないよう uint32 で索引を作る (int64 だと 1 枚 66MB)。"""
    bgr = cv2.imread(str(path))
    idx = (bgr[..., 2].astype(np.uint32) << 16) | (bgr[..., 1].astype(np.uint32) << 8) \
        | bgr[..., 0].astype(np.uint32)
    del bgr
    return lut[idx]


def weighted_dice(pred: np.ndarray, gt: np.ndarray, ids, w, n_lab: int) -> float:
    idx = gt.ravel().astype(np.int32) * n_lab + pred.ravel().astype(np.int32)
    cm = np.bincount(idx, minlength=n_lab * n_lab).reshape(n_lab, n_lab)
    del idx
    tp = np.diag(cm).astype(float)
    n_gt, n_pred = cm.sum(1).astype(float), cm.sum(0).astype(float)
    both0 = (n_gt == 0) & (n_pred == 0)
    one0 = ((n_gt == 0) | (n_pred == 0)) & ~both0
    d = 2 * tp / np.maximum(n_gt + n_pred, 1)
    d = np.where(both0, 1.0, np.where(one0, 0.0, d))
    return float((d[ids] * w).sum() / w.sum())


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred", required=True, help="task1/ task2/ を含むディレクトリ")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    pred_root = Path(args.pred)
    if not pred_root.is_absolute():
        pred_root = REPO / pred_root
    res = {}
    import re
    for task, sub in (("fine", "task1"), ("coarse", "task2")):
        lut, ids, w, n_lab, gt_sub = build_lut(task)
        files = sorted((pred_root / sub).glob("*.png"))
        if not files:
            print(f"{sub}: 予測が無い"); continue
        per_case: dict[str, list[float]] = {}
        for f in files:
            p = decode(f, lut)
            g = decode(REPO / "data" / gt_sub / f.name, lut)
            # 公式と同じ規則: ファイル名 stem の最後の "_" より前 (evaluate_seg.py)
            # 例外 2 枚 (..._12L_frame_3.png) はこの規則だと独立 case になる = 公式どおり
            case = f.stem.rsplit("_", 1)[0]
            per_case.setdefault(case, []).append(weighted_dice(p, g, ids, w, n_lab))
        case_mean = {k: float(np.mean(v)) for k, v in per_case.items()}
        score = float(np.mean(list(case_mean.values())))
        res[task] = {"dice": round(score, 4), "n_img": len(files), "n_case": len(case_mean)}
        # 公式番号: Task1 = coarse / Task2 = fine
        print(f"{sub} ({task}): Dice={score:.4f}  ({len(files)} 枚 / {len(case_mean)} case)")
    if res:
        res["mean_dice"] = round(np.mean([res[t]["dice"] for t in res if t in ("fine", "coarse")]), 4)
        print(f"平均 Dice = {res['mean_dice']:.4f}")
        out = Path(args.out) if args.out else pred_root / "dice_fullres.json"
        out.write_text(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
