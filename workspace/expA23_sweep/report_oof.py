"""expA23: 各レシピの **公式 OOF スコア**を一覧にする.

注意: このリポジトリの `oof/task1` は歴史的に **fine**、`task2` は **coarse** を指す
（公式の Task1=coarse / Task2=fine とは逆）。表では公式の呼び方に直して出す。

Usage: python3 report_oof.py
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent


def main() -> None:
    rows = []
    for f in sorted(HERE.glob("results/*/oof/oof_metrics.json")):
        m = json.loads(f.read_text())
        name = f.parents[1].name
        fine, coarse = m.get("task1", {}), m.get("task2", {})
        rows.append({
            "recipe": name.replace("expA23_", ""),
            # 公式番号: Task1 = coarse(merged) / Task2 = fine
            "T1_dice(coarse)": round(coarse.get("final_dice", float("nan")), 4),
            "T1_hd": round(coarse.get("final_hd", float("nan")), 4),
            "T2_dice(fine)": round(fine.get("final_dice", float("nan")), 4),
            "T2_hd": round(fine.get("final_hd", float("nan")), 4),
            "n_img": fine.get("n_images"),
            "n_case": fine.get("n_cases"),
        })
    if not rows:
        print("まだ公式 OOF 結果がありません")
        return
    df = pd.DataFrame(rows)
    df["mean_dice"] = ((df["T1_dice(coarse)"] + df["T2_dice(fine)"]) / 2).round(4)
    df["mean_hd"] = ((df["T1_hd"] + df["T2_hd"]) / 2).round(4)
    df = df.sort_values("mean_dice", ascending=False)
    print(df.to_string(index=False))
    print("\n参考 (公式 5fold OOF): ens5 = T1 0.6731/0.2351, T2 0.6820/0.2383")
    df.to_csv(HERE / "results" / "oof_report.csv", index=False)


if __name__ == "__main__":
    main()
