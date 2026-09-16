"""expA23: 公式指標 (weighted Dice + 正規化 Hausdorff) を **原寸・全 OOF** で計算し直す.

これまでのレシピ選定・後処理の判断はすべて **Dice のみ**で行ってきた。
HD は学習中の proxy（EDT・半解像度・終盤 3ep）でしか見ておらず、
公式の正規化 HD で順位が変わっていない保証がない。
（Dice と proxy HD の順位相関は −0.84 だったが、proxy と公式 HD が一致する保証は別物）

公式評価器 `metrics/01_evaluate_challenge.py` は 526 枚で 1 プロセス約 56 分かかるので、
**case 単位で分割して並列実行**し、公式と同じ階層平均（画像→case→全体）で合算する。

Usage:
    python3 eval_official_hd.py --pred workspace/expE01_ensemble/results/candB_new7 --tag candB
    python3 eval_official_hd.py --pred <dir> --tag <name> --jobs 32
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(REPO / "reference" / "tigersqai_challenge"))


def _one(args):
    """1 画像分の (case, task, dice, hd)。公式の関数をそのまま呼ぶ。"""
    fname, pred_fine, pred_coarse = args
    # 公式実装をそのまま使う（weighted_image_scores が Dice と HD を同じ規約で返す）
    import cv2
    from metrics import classes as C_fine
    from metrics import classes_merged as C_coarse
    from metrics.metrics import weighted_image_scores
    out = {}
    for task, pred_path, mod, gt_dir in (
            ("fine", pred_fine, C_fine, "masks_fine"),
            ("coarse", pred_coarse, C_coarse, "masks_coarse")):
        p_rgb = cv2.cvtColor(cv2.imread(str(pred_path), cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
        g_rgb = cv2.cvtColor(cv2.imread(str(REPO / "data" / gt_dir / fname), cv2.IMREAD_COLOR),
                             cv2.COLOR_BGR2RGB)
        p = mod.rgb_mask_to_label_mask(p_rgb)
        g = mod.rgb_mask_to_label_mask(g_rgb)
        classes = mod.CLASSES if task == "fine" else mod.CLASSES_MERGED
        wt = sum(c.weight for c in classes)
        sc = weighted_image_scores(p, g, classes, wt)
        out[task] = (float(sc["dice"]), float(sc["hd"]))
    return fname, out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred", required=True, help="task1=fine 色 / task2=coarse 色 のディレクトリ")
    ap.add_argument("--tag", required=True)
    ap.add_argument("--jobs", type=int, default=24)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--out", default=str(HERE / "official_hd.json"))
    args = ap.parse_args()

    root = REPO / args.pred if not Path(args.pred).is_absolute() else Path(args.pred)
    files = sorted((root / "task1").glob("*.png"))
    if args.limit:
        files = files[: args.limit]
    assert files, root
    tasks = [(f.name, f, root / "task2" / f.name) for f in files]
    with mp.Pool(args.jobs) as pool:
        res = pool.map(_one, tasks, chunksize=2)

    per_case = {"fine": {}, "coarse": {}}
    for fname, out in res:
        case = fname.rsplit(".", 1)[0].rsplit("_", 1)[0]
        for t, (d, h) in out.items():
            per_case[t].setdefault(case, []).append((d, h))
    summary = {}
    for t in ("fine", "coarse"):
        cm = {c: (float(np.mean([x[0] for x in v])), float(np.mean([x[1] for x in v])))
              for c, v in per_case[t].items()}
        summary[t] = {"dice": round(float(np.mean([v[0] for v in cm.values()])), 4),
                      "hd": round(float(np.mean([v[1] for v in cm.values()])), 4),
                      "n_case": len(cm), "n_img": len(files)}
    # 公式番号: Task1 = coarse / Task2 = fine
    summary["official"] = {"task1_dice": summary["coarse"]["dice"], "task1_hd": summary["coarse"]["hd"],
                           "task2_dice": summary["fine"]["dice"], "task2_hd": summary["fine"]["hd"]}
    p = Path(args.out)
    allr = json.loads(p.read_text()) if p.exists() else {}
    allr[args.tag] = summary
    p.write_text(json.dumps(allr, indent=1, ensure_ascii=False))
    o = summary["official"]
    print(f"{args.tag}: T1 coarse Dice {o['task1_dice']:.4f} / HD {o['task1_hd']:.4f}   "
          f"T2 fine Dice {o['task2_dice']:.4f} / HD {o['task2_hd']:.4f}", flush=True)


if __name__ == "__main__":
    main()
