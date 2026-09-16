"""expT04: Task3 の特徴セットを差し替えて比較する（在庫 / 解剖文脈 / 粗グリッド）.

  inv   : 在庫（面積・存在・重心・成分数・bbox, 315）
  anat  : 解剖文脈（接触行列・周囲クラス・リンパ節からの近傍性, 202）= AnatomyLoss の Task3 版
  grid  : 予測マスクを 6x12 に落としたセル占有率（1224）
の単体と組合せを、同じ fold・同じ LGBM 設定で比較する。
評価は公式（重み付き F1@0.5 / AUROC）。閾値較正は行わない（生の比較）。

Usage: python3 feature_ablation.py --tag candE
"""

from __future__ import annotations

import argparse
import itertools
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(REPO / "reference" / "tigersqai_challenge"))
from fusion_compare import STATIONS, official_eval  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="candE")
    ap.add_argument("--seeds", type=int, default=2)
    args = ap.parse_args()
    import lightgbm as lgb

    gt = pd.read_csv(REPO / "workspace/data_proc/task3_gt_wide.csv")
    folds = pd.read_csv(REPO / "workspace/fold/v2/folds.csv")
    s2f = {f.rsplit(".", 1)[0]: fo for f, fo in zip(folds.filename, folds.fold)}
    gt["fold"] = gt.case_id.map(s2f)
    gt["center"] = gt.case_id.str.extract(r"(center_\d+)")
    sets = {}
    for name, path in (("inv", f"features_{args.tag}_fix.csv"),
                       ("anat", f"features_anat_{args.tag}.csv"),
                       ("grid", f"features_grid_{args.tag}.csv")):
        d = pd.read_csv(HERE / path)
        sets[name] = d
    df = gt
    for name, d in sets.items():
        df = df.merge(d.rename(columns={c: f"{name}::{c}" for c in d.columns if c != "case_id"}),
                      on="case_id", how="inner")
    Y = df[STATIONS].values.astype(np.float32)
    cols = {n: [c for c in df.columns if c.startswith(f"{n}::")] for n in sets}
    print("次元:", {n: len(c) for n, c in cols.items()}, "行", len(df))

    combos = [("inv",), ("anat",), ("grid",), ("inv", "anat"), ("inv", "grid"),
              ("anat", "grid"), ("inv", "anat", "grid")]
    res = {}
    for combo in combos:
        X = df[sum([cols[c] for c in combo], [])].values.astype(np.float32)
        oof = np.zeros_like(Y)
        for k in range(5):
            tr, te = (df.fold != k).values, (df.fold == k).values
            for j, st in enumerate(STATIONS):
                y = Y[tr, j]
                if y.sum() in (0, len(y)):
                    oof[te, j] = float(y.mean()); continue
                ps = []
                for s in range(args.seeds):
                    c = lgb.LGBMClassifier(n_estimators=300, learning_rate=0.05, num_leaves=15,
                                           min_child_samples=20, subsample=0.8, subsample_freq=1,
                                           colsample_bytree=0.5, reg_lambda=1.0,
                                           random_state=42 + s * 7, verbose=-1)
                    c.fit(X[tr], y); ps.append(c.predict_proba(X[te])[:, 1])
                oof[te, j] = np.mean(ps, 0)
        out = pd.DataFrame(oof, columns=STATIONS); out.insert(0, "case_id", df.case_id.values)
        p = HERE / f"_ab_{'+'.join(combo)}.csv"; out.to_csv(p, index=False)
        r = official_eval(p)
        res["+".join(combo)] = {"f1": round(r["final_f1"], 4), "auroc": round(r["final_auroc"], 4),
                                "dim": int(sum(len(cols[c]) for c in combo))}
        print(f"{'+'.join(combo):20s} dim {res['+'.join(combo)]['dim']:5d}  "
              f"F1 {r['final_f1']:.4f}  AUROC {r['final_auroc']:.4f}", flush=True)
    (HERE / f"feature_ablation_{args.tag}.json").write_text(json.dumps(res, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()
