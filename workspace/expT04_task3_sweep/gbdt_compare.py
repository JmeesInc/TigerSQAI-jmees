"""expT04: Task3 の勾配ブースティングを LightGBM / XGBoost(scalar) / XGBoost(vector-leaf) で比較する.

Task3 は 14 station のマルチラベル。従来は **station ごとに独立な 2 値モデル** を建てていた
(LightGBM 14 本 / fold)。XGBoost の vector-leaf (`multi_strategy="multi_output_tree"`) は
**1 本の木を 14 出力で共有**し、葉に 14 次元のスコアを持つ。
station の可視性は「いま術野のどこを見ているか」で決まるので、どの station も同じ
分割質問（気管が写っているか、奇静脈弓が見えるか…）から恩恵を受けるはずで、
ブログの言う partition compatibility が高い状況にあたる。

vector は 1 ラウンドで木が 1 本しか増えないので、**ラウンド数を多め・学習率を低め**に取る。

Usage:
    python3 gbdt_compare.py --features features_candE.csv \
        --extra features_anat_candE.csv features_grid_candE.csv
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(REPO / "reference" / "tigersqai_challenge"))
from fusion_compare import STATIONS, official_eval  # noqa: E402


def load(features, extra):
    gt = pd.read_csv(REPO / "workspace/data_proc/task3_gt_wide.csv")
    folds = pd.read_csv(REPO / "workspace/fold/v2/folds.csv")
    s2f = {f.rsplit(".", 1)[0]: fo for f, fo in zip(folds.filename, folds.fold)}
    gt["fold"] = gt.case_id.map(s2f)
    inv = pd.read_csv(HERE / features)
    for e in extra:
        inv = inv.merge(pd.read_csv(HERE / e), on="case_id", how="inner")
    df = gt.merge(inv, on="case_id", how="inner").reset_index(drop=True)
    cols = [c for c in inv.columns if c != "case_id"]
    return df, df[cols].values.astype(np.float32), df[STATIONS].values.astype(np.float32)


def ev(oof, df, tag):
    o = pd.DataFrame(oof, columns=STATIONS)
    o.insert(0, "case_id", df.case_id.values)
    p = HERE / f"_gbdt_{tag}.csv"
    o.to_csv(p, index=False)
    r = official_eval(p)
    return r["final_f1"], r["final_auroc"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", default="features_candE.csv")
    ap.add_argument("--extra", nargs="*", default=["features_anat_candE.csv", "features_grid_candE.csv"])
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()
    import lightgbm as lgb
    import xgboost as xgb

    df, X, Y = load(args.features, args.extra)
    print(f"{X.shape[0]} 行 x {X.shape[1]} 次元 / {len(STATIONS)} station", flush=True)
    res = {}

    def run(name, fit_predict):
        t0 = time.time()
        oof = np.zeros_like(Y)
        for k in range(5):
            tr, te = (df.fold != k).values, (df.fold == k).values
            oof[te] = fit_predict(X[tr], Y[tr], X[te])
        f1, au = ev(oof, df, name)
        res[name] = {"f1": round(f1, 4), "auroc": round(au, 4), "sec": round(time.time() - t0, 1)}
        print(f"{name:34s} F1 {f1:.4f}  AUROC {au:.4f}  ({res[name]['sec']}s)", flush=True)
        np.save(HERE / f"_gbdt_{name}.npy", oof)

    # --- 1. LightGBM: station ごとに独立な 2 値モデル（現行）
    def lgb_fit(Xtr, Ytr, Xte):
        out = np.zeros((len(Xte), Ytr.shape[1]), np.float32)
        for j in range(Ytr.shape[1]):
            y = Ytr[:, j]
            if y.sum() in (0, len(y)):
                out[:, j] = float(y.mean()); continue
            c = lgb.LGBMClassifier(n_estimators=300, learning_rate=0.05, num_leaves=15,
                                   min_child_samples=20, subsample=0.8, subsample_freq=1,
                                   colsample_bytree=0.5, reg_lambda=1.0, random_state=42, verbose=-1)
            c.fit(Xtr, y); out[:, j] = c.predict_proba(Xte)[:, 1]
        return out
    run("lgb_per_station(現行)", lgb_fit)

    # --- 2-3. XGBoost: scalar / vector-leaf
    def xgb_fit(strategy, n_est, lr, depth=6):
        def f(Xtr, Ytr, Xte):
            m = xgb.XGBClassifier(
                n_estimators=n_est, learning_rate=lr, max_depth=depth,
                subsample=0.8, colsample_bytree=0.5, reg_lambda=1.0,
                tree_method="hist", device=args.device, verbosity=0, random_state=42,
                multi_strategy=strategy)
            m.fit(Xtr, Ytr)
            p = m.predict_proba(Xte)
            return np.asarray(p, dtype=np.float32)
        return f
    for n_est, lr in ((300, 0.05), (1000, 0.02)):
        run(f"xgb_scalar n={n_est} lr={lr}", xgb_fit("one_output_per_tree", n_est, lr))
    for n_est, lr in ((300, 0.05), (1000, 0.02), (3000, 0.01)):
        run(f"xgb_vector n={n_est} lr={lr}", xgb_fit("multi_output_tree", n_est, lr))

    (HERE / "gbdt_compare.json").write_text(json.dumps(res, indent=1, ensure_ascii=False))
    best = max(res, key=lambda k: res[k]["f1"])
    print(f"\n最良: {best}  F1 {res[best]['f1']}  AUROC {res[best]['auroc']}")


if __name__ == "__main__":
    main()
