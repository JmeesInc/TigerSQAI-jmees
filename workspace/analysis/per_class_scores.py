"""expA00/A01/A02 の OOF 予測をクラス別 Dice / HD で集計する.

- Dice/HD は公式 metrics.metrics.dice_per_class / hausdorff_per_class を使用
  (HD のみ fast_hd の厳密等価 EDT 実装にパッチ。適用前に公式実装と一致検証)
- 集約は公式と同じ階層: 画像 → case 平均 → 全体平均 (クラスごとに独立に実施)
- gt は workspace/data_proc/labels_* (前処理済み ID ラベル)、pred は RGB→ID を LUT で変換

Usage: python3 per_class_scores.py [--workers 16]
出力: per_class_task1.csv / per_class_task2.csv + stdout サマリ
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
sys.path.insert(0, str(REPO / "workspace" / "expA00_task12_baseline"))  # fast_hd

EXPS = {
    "A00": REPO / "workspace/expA00_task12_baseline/results/expA00_task12_baseline/oof",
    "A01": REPO / "workspace/expA01_strongaug/results/expA01_strongaug/oof",
    "A02": REPO / "workspace/expA02_toolpaste/results/expA02_toolpaste/oof",
    "A05": REPO / "workspace/expA05_maxvit/results/expA05_maxvit/oof",
    "A06": REPO / "workspace/expA06_f2c_loss/results/expA06_f2c_loss/oof",
    "B03": REPO / "workspace/expB03_dino_f2c/results/expB03_dino_f2c/oof",
    "A09": REPO / "workspace/expA09_lymph_aux/results/expA09_lymph_aux/oof",
    "A10": REPO / "workspace/expA10_toolmask/results/expA10_toolmask/oof",
    "A11": REPO / "workspace/expA11_v2data/results/expA11_v2data/oof",
}
DATA_PROC = REPO / "workspace" / "data_proc"

_G: dict = {}


def _init_worker() -> None:
    """worker 毎に公式 metrics を import し fast HD をパッチ、LUT をロード."""
    from fast_hd import patch_official_metrics
    patch_official_metrics()
    from metrics.classes import CLASSES
    from metrics.classes_merged import CLASSES_MERGED
    _G["classes"] = {"task1": CLASSES, "task2": CLASSES_MERGED}
    _G["lut"] = {
        "task1": np.load(DATA_PROC / "lut_fine.npy"),
        "task2": np.load(DATA_PROC / "lut_coarse.npy"),
    }
    _G["gt_dir"] = {"task1": DATA_PROC / "labels_fine", "task2": DATA_PROC / "labels_coarse"}


def _pack(rgb: np.ndarray) -> np.ndarray:
    rgb = rgb.astype(np.uint32)
    return (rgb[..., 0] << 16) | (rgb[..., 1] << 8) | rgb[..., 2]


def eval_one(args: tuple[str, str, str]) -> tuple[str, str, str, dict, dict]:
    """(exp, task, filename) → per-class dice / hd dicts."""
    import metrics.metrics as M

    exp, task, name = args
    classes = _G["classes"][task]
    gt = cv2.imread(str(_G["gt_dir"][task] / name), cv2.IMREAD_GRAYSCALE)
    bgr = cv2.imread(str(EXPS[exp] / task / name))
    pred = _G["lut"][task][_pack(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))]
    d = M.dice_per_class(pred, gt, classes)
    h = M.hausdorff_per_class(pred, gt, classes)
    return exp, task, name, d, h


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--exps", nargs="+", default=None, help="対象実験を絞る (例: --exps A05)")
    ap.add_argument("--suffix", default="", help="出力 CSV のサフィックス")
    args = ap.parse_args()
    if args.exps:
        for k in list(EXPS):
            if k not in args.exps:
                del EXPS[k]

    first_exp = next(iter(EXPS))
    names = sorted(p.name for p in (EXPS[first_exp] / "task1").glob("*.png"))
    jobs = [(e, t, n) for e in EXPS for t in ("task1", "task2") for n in names]
    print(f"{len(names)} images x {len(EXPS)} exps x 2 tasks = {len(jobs)} evals")

    rows = []
    with ProcessPoolExecutor(max_workers=args.workers, initializer=_init_worker) as ex:
        for i, (exp, task, name, d, h) in enumerate(ex.map(eval_one, jobs, chunksize=4)):
            case = Path(name).stem.rsplit("_", 1)[0]  # 公式の case パース
            for cid in d:
                rows.append((exp, task, case, cid, d[cid], h[cid]))
            if (i + 1) % 300 == 0:
                print(f"  {i + 1}/{len(jobs)}")
    df = pd.DataFrame(rows, columns=["exp", "task", "case", "cid", "dice", "hd"])

    _init_worker()  # クラス名参照用
    for task in ("task1", "task2"):
        classes = _G["classes"][task]
        meta = pd.DataFrame(
            [(c.label_id, c.name, c.weight) for c in classes], columns=["cid", "name", "weight"]
        )
        sub = df[df.task == task]
        # 画像 → case 平均 → 全体平均 (クラス毎)
        agg = (
            sub.groupby(["exp", "cid", "case"])[["dice", "hd"]].mean()
            .groupby(["exp", "cid"]).mean().reset_index()
        )
        wide = agg.pivot(index="cid", columns="exp", values=["dice", "hd"])
        wide.columns = [f"{m}_{e}" for m, e in wide.columns]
        out = meta.merge(wide.reset_index(), on="cid")
        if "dice_A00" in out.columns:
            for e in ("A01", "A02", "A05"):
                if f"dice_{e}" in out.columns:
                    out[f"dice_{e}_diff"] = out[f"dice_{e}"] - out["dice_A00"]
                    out[f"hd_{e}_diff"] = out[f"hd_{e}"] - out["hd_A00"]
        out = out.round(4)
        path = Path(__file__).parent / f"per_class_{task}{args.suffix}.csv"
        out.to_csv(path, index=False)
        print(f"\n===== {task} per-class (Dice, 公式階層集約) =====")
        cols = ["cid", "name", "weight"] + [c for c in out.columns if c.startswith(("dice_", "hd_")) and not c.endswith("_diff")]
        print(out[cols].to_string(index=False))
        print(f"saved {path}")


if __name__ == "__main__":
    main()
