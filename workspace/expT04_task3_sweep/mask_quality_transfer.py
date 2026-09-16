"""expT04: 在庫/解剖/グリッド特徴の「出所の質」が Task3 に与える影響を測る.

提出コンテナは学習時（候補 E 相当 9 モデル）より強い 19 モデルのアンサンブルでマスクを作る。
学習時と推論時でマスクの質が違うとき、分類器は得をするのか損をするのか。
上限として GT マスク、近い質として候補 B（7 モデル）も入れて、
  train=A / test=B の全組合せを同じ fold・同じ LGBM で測る。

Usage: python3 mask_quality_transfer.py
"""

from __future__ import annotations

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

SRC = {
    "candB(7モデル)": ("features_candB_fix.csv", "features_anat_candB.csv", "features_grid_candB.csv"),
    "candE(9モデル)": ("features_candE_fix.csv", "features_anat_candE.csv", "features_grid_candE.csv"),
    "GT": ("features_gt_fix.csv", "features_anat_gt.csv", "features_grid_gt.csv"),
}


def main():
    import lightgbm as lgb
    gt = pd.read_csv(REPO / "workspace/data_proc/task3_gt_wide.csv")
    folds = pd.read_csv(REPO / "workspace/fold/v2/folds.csv")
    s2f = {f.rsplit(".", 1)[0]: fo for f, fo in zip(folds.filename, folds.fold)}
    gt["fold"] = gt.case_id.map(s2f)

    feats = {}
    for name, files in SRC.items():
        d = None
        for f in files:
            p = HERE / f
            if not p.exists():
                print(f"!! {p} が無いので {name} を飛ばす"); d = None; break
            e = pd.read_csv(p)
            d = e if d is None else d.merge(e, on="case_id", how="inner")
        if d is not None:
            feats[name] = d
    common = set(gt.case_id)
    for d in feats.values():
        common &= set(d.case_id)
    common = sorted(common)
    print(f"共通 {len(common)} 行 / 特徴 {[(k, v.shape[1]-1) for k, v in feats.items()]}")
    base = gt[gt.case_id.isin(common)].sort_values("case_id").reset_index(drop=True)
    Y = base[STATIONS].values.astype(np.float32)
    cols = [c for c in feats["candE(9モデル)"].columns if c != "case_id"]
    X = {k: v[v.case_id.isin(common)].sort_values("case_id")[cols].values.astype(np.float32)
         for k, v in feats.items()}

    res = {}
    for tr_name in feats:
        for te_name in feats:
            oof = np.zeros_like(Y)
            for k in range(5):
                tr, te = (base.fold != k).values, (base.fold == k).values
                for j in range(len(STATIONS)):
                    y = Y[tr, j]
                    if y.sum() in (0, len(y)):
                        oof[te, j] = float(y.mean()); continue
                    c = lgb.LGBMClassifier(n_estimators=300, learning_rate=0.05, num_leaves=15,
                                           min_child_samples=20, subsample=0.8, subsample_freq=1,
                                           colsample_bytree=0.5, reg_lambda=1.0,
                                           random_state=42, verbose=-1)
                    c.fit(X[tr_name][tr], y)
                    oof[te, j] = c.predict_proba(X[te_name][te])[:, 1]
            o = pd.DataFrame(oof, columns=STATIONS); o.insert(0, "case_id", base.case_id.values)
            p = HERE / "_mq.csv"; o.to_csv(p, index=False); r = official_eval(p)
            res[f"{tr_name} -> {te_name}"] = {"f1": round(r["final_f1"], 4),
                                              "auroc": round(r["final_auroc"], 4)}
            print(f"train={tr_name:14s} test={te_name:14s}  F1 {r['final_f1']:.4f}  "
                  f"AUROC {r['final_auroc']:.4f}", flush=True)
    (HERE / "mask_quality_transfer.json").write_text(json.dumps(res, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()
