"""expT04: マスクの出所を混ぜて学習し、入力の質に頑健な Task3 分類器を作る.

mask_quality_transfer.py で分かったこと:
  * 予測マスク同士 (candB 7 モデル ↔ candE 9 モデル) の質の違いには頑健
  * GT で学習 → 予測で推論は F1 0.45 と壊滅。GT 学習は「きれいなマスク」前提の規則になる
  * 予測で学習 → GT で推論は F1 −0.025 / AUROC +0.008（決定境界がずれる）
  * 上限 (GT→GT) は 0.8501 で、現行 0.7666 との差 0.08 が「マスク品質の伸びしろ」

そこで **複数の出所の特徴を行方向に連結して 1 つのモデルを学習** する（= 入力分布の augmentation）。
同じ case は同じ fold に入るので分割は壊れない。テストは出所ごとに別々に測り、
「どの質のマスクが来ても落ちない」かを見る。

Usage: python3 mask_mix_aug.py
"""

from __future__ import annotations

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

SRC = {
    "candB": ("features_candB_fix.csv", "features_anat_candB.csv", "features_grid_candB.csv"),
    "candE": ("features_candE_fix.csv", "features_anat_candE.csv", "features_grid_candE.csv"),
    "GT": ("features_gt_fix.csv", "features_anat_gt.csv", "features_grid_gt.csv"),
}
MIXES = [("candE",), ("candB", "candE"), ("candB", "candE", "GT"), ("candE", "GT"), ("GT",)]


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
            e = pd.read_csv(HERE / f)
            d = e if d is None else d.merge(e, on="case_id", how="inner")
        feats[name] = d
    common = set(gt.case_id)
    for d in feats.values():
        common &= set(d.case_id)
    common = sorted(common)
    base = gt[gt.case_id.isin(common)].sort_values("case_id").reset_index(drop=True)
    Y = base[STATIONS].values.astype(np.float32)
    cols = [c for c in feats["candE"].columns if c != "case_id"]
    X = {k: v[v.case_id.isin(common)].sort_values("case_id")[cols].values.astype(np.float32)
         for k, v in feats.items()}
    print(f"{len(common)} 行 x {len(cols)} 次元 / 出所 {list(X)}", flush=True)

    res = {}
    for mix in MIXES:
        name = "+".join(mix)
        oof = {t: np.zeros_like(Y) for t in X}
        for k in range(5):
            tr, te = (base.fold != k).values, (base.fold == k).values
            Xtr = np.vstack([X[s][tr] for s in mix])
            Ytr = np.vstack([Y[tr]] * len(mix))
            for j in range(len(STATIONS)):
                y = Ytr[:, j]
                if y.sum() in (0, len(y)):
                    for t in X:
                        oof[t][te, j] = float(y.mean())
                    continue
                c = lgb.LGBMClassifier(n_estimators=300, learning_rate=0.05, num_leaves=15,
                                       min_child_samples=20, subsample=0.8, subsample_freq=1,
                                       colsample_bytree=0.5, reg_lambda=1.0,
                                       random_state=42, verbose=-1, n_jobs=16)
                c.fit(Xtr, y)
                for t in X:
                    oof[t][te, j] = c.predict_proba(X[t][te])[:, 1]
        row = {}
        for t in X:
            o = pd.DataFrame(oof[t], columns=STATIONS); o.insert(0, "case_id", base.case_id.values)
            p = HERE / "_mix.csv"; o.to_csv(p, index=False); r = official_eval(p)
            row[t] = {"f1": round(r["final_f1"], 4), "auroc": round(r["final_auroc"], 4)}
        res[name] = row
        worst = min(row, key=lambda t: row[t]["f1"])
        print(f"train={name:18s} " + "  ".join(f"{t}: {row[t]['f1']:.4f}/{row[t]['auroc']:.4f}"
                                               for t in ("candB", "candE", "GT"))
              + f"   最悪 {row[worst]['f1']:.4f} ({worst})", flush=True)
    (HERE / "mask_mix_aug.json").write_text(json.dumps(res, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()
