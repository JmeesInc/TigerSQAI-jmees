"""A06(v1) vs A09/A10/A11(v2) を **共通 524 枚** の Dice で比較する.

v2 の 2 枚 (na_station) が A06 の評価集合に無いため、素の final_dice 同士は
比較できない。ここでは全実験に共通する画像だけで公式と同一の階層集約
(画像→case→全体、クラス重み 3/2/1、背景除外) を再計算する。
HD は高価なので Dice のみ。

Usage: python3 common_dice.py [--task task1] [--workers 24]
"""
from __future__ import annotations

import argparse
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "reference" / "tigersqai_challenge"))

EXPS = {
    "A06": REPO / "workspace/expA06_f2c_loss/results/expA06_f2c_loss/oof",
    "A09": REPO / "workspace/expA09_lymph_aux/results/expA09_lymph_aux/oof",
    "A10": REPO / "workspace/expA10_toolmask/results/expA10_toolmask/oof",
    "A11": REPO / "workspace/expA11_v2data/results/expA11_v2data/oof",
    "ens2": REPO / "workspace/expE01_ensemble/results/ens2",    # A06+A10
    "ens3": REPO / "workspace/expE01_ensemble/results/ens3",    # A06+A09+A10
    "ens4": REPO / "workspace/expE01_ensemble/results/ens4",
    "ens5": REPO / "workspace/expE01_ensemble/results/ens5",   # +A05
    "A16": REPO / "workspace/expA16_unetpp_cholec/results/expA16_unetpp_cholec_ep20/oof",    # +A11
}
DATA_PROC = REPO / "workspace" / "data_proc"
_G: dict = {}


def _init(task: str) -> None:
    from metrics.classes import CLASSES
    from metrics.classes_merged import CLASSES_MERGED
    _G["classes"] = CLASSES if task == "task1" else CLASSES_MERGED
    _G["lut"] = np.load(DATA_PROC / ("lut_fine.npy" if task == "task1" else "lut_coarse.npy"))
    _G["gt_dir"] = DATA_PROC / ("labels_fine" if task == "task1" else "labels_coarse")
    _G["task"] = task


def _pack(rgb: np.ndarray) -> np.ndarray:
    rgb = rgb.astype(np.uint32)
    return (rgb[..., 0] << 16) | (rgb[..., 1] << 8) | rgb[..., 2]


def eval_one(args: tuple[str, str]) -> tuple[str, str, dict]:
    import metrics.metrics as M
    exp, name = args
    gt = cv2.imread(str(_G["gt_dir"] / name), cv2.IMREAD_GRAYSCALE)
    bgr = cv2.imread(str(EXPS[exp] / _G["task"] / name))
    pred = _G["lut"][_pack(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))]
    return exp, name, M.dice_per_class(pred, gt, _G["classes"])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", default="task1")
    ap.add_argument("--workers", type=int, default=24)
    args = ap.parse_args()

    sets = {e: {p.name for p in (d / args.task).glob("*.png")} for e, d in EXPS.items()}
    for e, s in sets.items():
        print(f"{e}: {len(s)} images")
    common = sorted(set.intersection(*sets.values()))
    print(f"共通: {len(common)} images")

    jobs = [(e, n) for e in EXPS for n in common]
    rows = []
    with ProcessPoolExecutor(args.workers, initializer=_init, initargs=(args.task,)) as ex:
        for exp, name, d in ex.map(eval_one, jobs, chunksize=8):
            case = Path(name).stem.rsplit("_", 1)[0]
            for cid, v in d.items():
                rows.append((exp, case, cid, v))
    df = pd.DataFrame(rows, columns=["exp", "case", "cid", "dice"])

    _init(args.task)
    w = {c.label_id: c.weight for c in _G["classes"]}
    # 公式集約: 画像→case→全体 のあとクラス重み平均 (重みが定数なので順序不変)
    agg = df.groupby(["exp", "cid", "case"]).dice.mean().groupby(["exp", "cid"]).mean().reset_index()
    agg["w"] = agg.cid.map(w)
    final = agg.groupby("exp").apply(
        lambda g: (g.dice * g.w).sum() / g.w.sum(), include_groups=False
    )
    print(f"\n===== {args.task} 共通{len(common)}枚での weighted Dice =====")
    for e in EXPS:
        print(f"  {e}: {final[e]:.4f}")

    # case 単位のペア比較 (A11 基準)
    case_w = df.groupby(["exp", "case", "cid"]).dice.mean().reset_index()
    case_w["w"] = case_w.cid.map(w)
    per_case = case_w.groupby(["exp", "case"]).apply(
        lambda g: (g.dice * g.w).sum() / g.w.sum(), include_groups=False
    ).unstack(0)
    print(f"\n===== case 単位ペア比較 (n={len(per_case)}, 基準=A06) =====")
    for e in per_case.columns:
        if e == "A06":
            continue
        d = per_case[e] - per_case["A06"]
        print(f"  {e:5s} - A06: mean={d.mean():+.4f}  勝ち {int((d > 0).sum())}/{len(d)}  "
              f"median={d.median():+.4f}  std={d.std():.4f}")
    per_case.round(4).to_csv(Path(__file__).parent / f"common_dice_{args.task}.csv")


if __name__ == "__main__":
    main()
