"""task1 予測の fine→coarse 写像 vs coarse decoder 出力の比較 (学習不要 ablation).

labelmap.csv の fine_id→merged_id は全射で、GT レベルでは map(fine GT)==coarse GT が
画素単位で成立する。→ task1 予測を写像したものを task2 予測として公式評価し、
専用 coarse decoder の出力と比べる。

Usage: python3 fine2coarse_ablation.py [--exp expA05_maxvit]
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "reference" / "tigersqai_challenge"))
sys.path.insert(0, str(REPO / "workspace" / "expA00_task12_baseline"))  # fast_hd

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
log = logging.getLogger("f2c")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp", default="expA05_maxvit")
    args = ap.parse_args()

    oof = REPO / "workspace" / args.exp / "results" / args.exp / "oof"
    out_dir = oof / "task2_from_task1"
    out_dir.mkdir(exist_ok=True)

    lm = pd.read_csv(REPO / "data/labelmap.csv")
    # fine RGB -> coarse RGB の直接 LUT (packed 24bit)
    lut = np.zeros((1 << 24, 3), dtype=np.uint8)
    for r in lm.itertuples():
        packed = (int(r.fine_r) << 16) | (int(r.fine_g) << 8) | int(r.fine_b)
        lut[packed] = (r.merged_r, r.merged_g, r.merged_b)

    names = sorted(p.name for p in (oof / "task1").glob("*.png"))
    log.info("mapping %d task1 predictions -> coarse", len(names))
    for i, n in enumerate(names):
        rgb = cv2.cvtColor(cv2.imread(str(oof / "task1" / n)), cv2.COLOR_BGR2RGB).astype(np.uint32)
        packed = (rgb[..., 0] << 16) | (rgb[..., 1] << 8) | rgb[..., 2]
        cv2.imwrite(str(out_dir / n), cv2.cvtColor(lut[packed], cv2.COLOR_RGB2BGR))
        if (i + 1) % 100 == 0:
            log.info("  %d/%d", i + 1, len(names))

    from fast_hd import patch_official_metrics
    patch_official_metrics()
    from metrics.classes_merged import CLASSES_MERGED, rgb_mask_to_label_mask
    from metrics.evaluate_seg import evaluate

    gt_dir = oof / "gt_task2"  # 既存 OOF 評価が作った symlink dir を再利用
    assert gt_dir.exists(), f"missing {gt_dir} (先に通常の OOF 評価を実行しておくこと)"
    log.info("evaluating mapped predictions with official code...")
    res = evaluate(gt_dir, out_dir, classes=CLASSES_MERGED, rgb_to_label_fn=rgb_mask_to_label_mask)
    log.info("task2(from task1 map) | Dice=%.4f HD=%.4f (%d cases)",
             res["final_dice"], res["final_hd"], len(res["case_ids"]))

    ref = json.loads((oof / "oof_metrics.json").read_text())["task2"]
    log.info("task2(coarse decoder)  | Dice=%.4f HD=%.4f", ref["final_dice"], ref["final_hd"])
    (oof / "task2_from_task1_metrics.json").write_text(json.dumps(
        {"final_dice": res["final_dice"], "final_hd": res["final_hd"],
         "case_dice": res["case_dice"], "case_hd": res["case_hd"]}, indent=2))


if __name__ == "__main__":
    main()
