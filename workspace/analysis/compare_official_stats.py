"""expA00 / expA01 / expA02 を公式の統計・ランキング (metrics/stats.py) で比較する.

入力は各実験の OOF 公式評価結果 (oof_metrics.json — evaluate_seg.evaluate の出力を保存したもの)。
公式 analyse_seg = rank(Dice)+rank(HD) の複合ランク、shared bootstrap CI95 (n=1000)、
pairwise Wilcoxon をそのまま適用する（再評価なし）。

Usage: python3 compare_official_stats.py
出力: stdout の表 + compare_stats.json
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "reference" / "tigersqai_challenge"))

from metrics.stats import analyse_seg  # noqa: E402

EXPS = {
    "expA00_baseline": REPO / "workspace/expA00_task12_baseline/results/expA00_task12_baseline/oof/oof_metrics.json",
    "expA01_strongaug": REPO / "workspace/expA01_strongaug/results/expA01_strongaug/oof/oof_metrics.json",
    "expA02_toolpaste": REPO / "workspace/expA02_toolpaste/results/expA02_toolpaste/oof/oof_metrics.json",
    "expA05_maxvit": REPO / "workspace/expA05_maxvit/results/expA05_maxvit/oof/oof_metrics.json",
    "expA06_f2c_loss": REPO / "workspace/expA06_f2c_loss/results/expA06_f2c_loss/oof/oof_metrics.json",
    "expB01_dino_frozen": REPO / "workspace/expB01_dino_frozen/results/expB01_dino_frozen/oof/oof_metrics.json",
    "expB03_dino_f2c": REPO / "workspace/expB03_dino_f2c/results/expB03_dino_f2c/oof/oof_metrics.json",
}


def main() -> None:
    raw = {name: json.loads(p.read_text()) for name, p in EXPS.items()}
    out = {}
    for task in ("task1", "task2"):
        seg_results = {}
        for name, r in raw.items():
            t = r[task]
            seg_results[name] = {
                "case_ids": list(t["case_dice"].keys()),
                "case_dice": t["case_dice"],
                "case_hd": t["case_hd"],
                "final_dice": t["final_dice"],
                "final_hd": t["final_hd"],
            }
        res = analyse_seg(seg_results)
        out[task] = res

        print(f"\n===== {task} (公式 analyse_seg, n_bootstrap={res['n_bootstrap']}) =====")
        print(f"{'method':<18} {'Dice':>7} {'CI95':>17} {'HD':>7} {'CI95':>17} {'rankD':>5} {'rankH':>5} {'rank':>5}")
        for m, d in res["methods"].items():
            ci_d = f"[{d['dice_ci95'][0]:.4f},{d['dice_ci95'][1]:.4f}]"
            ci_h = f"[{d['hd_ci95'][0]:.4f},{d['hd_ci95'][1]:.4f}]"
            print(f"{m:<18} {d['final_dice']:>7.4f} {ci_d:>17} {d['final_hd']:>7.4f} {ci_h:>17}"
                  f" {d['rank_dice']:>5.1f} {d['rank_hd']:>5.1f} {d['seg_rank']:>5.2f}")
        print("Wilcoxon p (Dice):")
        for a, row in res["wilcoxon_dice"].items():
            for b, p in row.items():
                if a < b and p == p:
                    print(f"  {a} vs {b}: p={p:.4f}")
        print("Wilcoxon p (HD):")
        for a, row in res["wilcoxon_hd"].items():
            for b, p in row.items():
                if a < b and p == p:
                    print(f"  {a} vs {b}: p={p:.4f}")

    (Path(__file__).parent / "compare_stats.json").write_text(json.dumps(out, indent=2, default=str))
    print("\nsaved compare_stats.json")


if __name__ == "__main__":
    main()
