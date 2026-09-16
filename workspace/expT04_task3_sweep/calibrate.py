"""expT04: Task3 の確率を **閾値 0.5 固定の Weighted F1 に合わせて較正**する.

公式は `f1_score(y, p >= 0.5)`。**閾値が動かせない以上、確率側を動かすのが唯一のレバー**。
クラスごとに最適閾値 t_c を求め、t_c が 0.5 に来るよう区分線形に写像する:

    p < t_c : p' = 0.5 * p / t_c
    p >= t_c: p' = 0.5 + 0.5 * (p - t_c) / (1 - t_c)

**単調写像なので AUROC は不変**、変わるのは F1 だけ。

честность (honesty): fold f の行に当てる t_c は **fold f 以外の OOF 行だけ**から推定する
(nested)。提出用には全 OOF から推定した t_c を `thresholds.json` に保存する。

Usage:
    python3 calibrate.py --pred ../expT03_task3_noleak/results/oof_task3.csv
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO / "reference" / "tigersqai_challenge"))

STATIONS = ["6L", "6R", "7L", "7R", "8", "9", "10L", "10R", "11L", "11R", "12L", "12R", "13L", "13R"]


def official_eval(pred_csv: Path) -> dict:
    from metrics.classes_stations import CLASSES_STATIONS
    from metrics.evaluate_cls import evaluate
    return evaluate(gt_csv=REPO / "workspace/data_proc/task3_gt_wide.csv",
                    pred_csv=pred_csv, classes=CLASSES_STATIONS)


def best_threshold(y: np.ndarray, p: np.ndarray) -> float:
    """そのクラスの F1 を最大にする閾値 (候補は観測確率の中点)。"""
    from sklearn.metrics import f1_score
    if y.sum() == 0 or y.sum() == len(y):
        return 0.5
    cands = np.unique(np.clip(p, 1e-4, 1 - 1e-4))
    if len(cands) > 200:
        cands = np.quantile(cands, np.linspace(0.01, 0.99, 200))
    best, bt = -1.0, 0.5
    for t in cands:
        f = f1_score(y, (p >= t).astype(int), zero_division=0)
        if f > best:
            best, bt = f, float(t)
    return bt


def remap(p: np.ndarray, t: float) -> np.ndarray:
    t = float(np.clip(t, 1e-3, 1 - 1e-3))
    lo = 0.5 * p / t
    hi = 0.5 + 0.5 * (p - t) / (1 - t)
    return np.where(p < t, lo, hi).clip(0, 1)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred", required=True)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    pred = pd.read_csv(args.pred)
    gt = pd.read_csv(REPO / "workspace/data_proc/task3_gt_wide.csv")
    folds = pd.read_csv(REPO / "workspace/fold/v2/folds.csv")
    stem2fold = {f.rsplit(".", 1)[0]: fo for f, fo in zip(folds.filename, folds.fold)}
    df = pred.merge(gt, on="case_id", suffixes=("_p", "_y"))
    df["fold"] = df.case_id.map(stem2fold)

    before = official_eval(Path(args.pred))
    cal = pred.copy().set_index("case_id")
    for fold in sorted(df.fold.unique()):
        tr, va = df.fold != fold, df.fold == fold
        for st in STATIONS:
            t = best_threshold(df.loc[tr, f"{st}_y"].values, df.loc[tr, f"{st}_p"].values)
            ids = df.loc[va, "case_id"].values
            cal.loc[ids, st] = remap(df.loc[va, f"{st}_p"].values, t)
    out = Path(args.out) if args.out else HERE / (Path(args.pred).stem + "_cal.csv")
    cal.reset_index().to_csv(out, index=False)
    after = official_eval(out)

    # 提出用: 全 OOF から推定した閾値
    thr = {st: best_threshold(df[f"{st}_y"].values, df[f"{st}_p"].values) for st in STATIONS}
    (HERE / f"thresholds_{Path(args.pred).stem}.json").write_text(json.dumps(thr, indent=2))

    print(f"before: F1@0.5 {before['final_f1']:.4f} / AUROC {before['final_auroc']:.4f}")
    print(f"after : F1@0.5 {after['final_f1']:.4f} / AUROC {after['final_auroc']:.4f}"
          f"  (差 {after['final_f1'] - before['final_f1']:+.4f})")
    print("per-class threshold:", {k: round(v, 3) for k, v in thr.items()})
    print(f"-> {out}")


if __name__ == "__main__":
    main()
