"""OOF 予測 CSV を公式 evaluate_cls (CLASSES_STATIONS) で採点する.

Usage: python3 eval_official.py [--pred results/oof_task3.csv]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "reference" / "tigersqai_challenge"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred", default=str(Path(__file__).parent / "results/oof_task3.csv"))
    args = ap.parse_args()

    from metrics.classes_stations import CLASSES_STATIONS
    from metrics.evaluate_cls import evaluate

    res = evaluate(
        gt_csv=REPO / "workspace/data_proc/task3_gt_wide.csv",
        pred_csv=Path(args.pred),
        classes=CLASSES_STATIONS,
    )
    print(f"Weighted F1@0.5 = {res['final_f1']:.4f}")
    print(f"AUROC           = {res['final_auroc']:.4f}")
    for k in sorted(res.keys()):
        if k.startswith("per_class") or k in ("class_f1", "class_auroc"):
            print(k, {c: round(v, 3) for c, v in list(res[k].items())[:14]} if isinstance(res[k], dict) else res[k])


if __name__ == "__main__":
    main()
