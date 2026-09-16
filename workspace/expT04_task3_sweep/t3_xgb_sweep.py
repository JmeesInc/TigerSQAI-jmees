"""expT04: XGBoost 枝のハイパラ・特徴・学習データ探索（Task3 の主枝なので優先）.

基準構成 = vector-leaf, depth 6, n=1000 lr=0.02, colsample 0.5, subsample 0.8, 1741 次元, candE のみ。
そこから 1 軸ずつ動かす（one-at-a-time）。各構成の OOF は members_xgb/<name>.npy に保存し、
後で t3_search2 の greedy / 合成に流用できるようにする。
評価は nested 較正 λ=1.0 の F1@0.5 + AUROC（公式 evaluate_cls）。

Usage:
    CUDA_VISIBLE_DEVICES=2 python3 t3_xgb_sweep.py --stage run
    python3 t3_xgb_sweep.py --stage report
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
from fusion_compare import STATIONS  # noqa: E402
from t3_search import Scorer, load_df  # noqa: E402
from t3_search2 import load_gt_feats  # noqa: E402

OUT = HERE / "members_xgb"
OUT.mkdir(exist_ok=True)

BASE = dict(depth=6, n=1000, lr=0.02, col=0.5, sub=0.8, mcw=1, lam=1.0, alpha=0.0, gamma=0.0,
            strategy="multi_output_tree", feat="f1741", gtmix=False)
VARIANTS = {
    "base": {},
    "depth4": {"depth": 4},
    "depth8": {"depth": 8},
    "depth10": {"depth": 10},
    "n2000_lr01": {"n": 2000, "lr": 0.01},
    "n600_lr03": {"n": 600, "lr": 0.03},
    "n300_lr05": {"n": 300, "lr": 0.05},
    "col03": {"col": 0.3},
    "col08": {"col": 0.8},
    "mcw3": {"mcw": 3},
    "mcw5": {"mcw": 5},
    "lam5": {"lam": 5.0},
    "sub06": {"sub": 0.6},
    "scalar": {"strategy": "one_output_per_tree"},
    "gtmix": {"gtmix": True},
    "feat_inv_anat": {"feat": "f517"},
    "feat315": {"feat": "f315"},
    # --- 第 2 ラウンド: 正則化の深掘り（lam5 が +0.0086 だった）
    "lam3": {"lam": 3.0},
    "lam10": {"lam": 10.0},
    "lam20": {"lam": 20.0},
    "lam5_alpha1": {"lam": 5.0, "alpha": 1.0},
    "alpha1": {"alpha": 1.0},
    "lam5_col03": {"lam": 5.0, "col": 0.3},
    "lam5_mcw5": {"lam": 5.0, "mcw": 5},
    "lam5_n2000": {"lam": 5.0, "n": 2000, "lr": 0.01},
    "lam5_depth8": {"lam": 5.0, "depth": 8},
    "lam10_n2000": {"lam": 10.0, "n": 2000, "lr": 0.01},
    "gamma1": {"gamma": 1.0},
    "lam5_gamma1": {"lam": 5.0, "gamma": 1.0},
}


def feats_for(df, feats, key):
    if key in feats:
        return feats[key]
    if key == "f517":     # 在庫 315 + 解剖文脈 202（グリッド抜き）
        inv = pd.read_csv(HERE / "features_candE_fix.csv").merge(
            pd.read_csv(HERE / "features_anat_candE.csv"), on="case_id").set_index("case_id").reindex(df.case_id)
        return inv.values.astype(np.float32)
    raise KeyError(key)


def run_one(name, cfg, df, feats, Y, GT, device, seed=42):
    import xgboost as xgb
    p = OUT / f"{name}_s{seed}.npy"
    if p.exists():
        return np.load(p)
    X = feats_for(df, feats, cfg["feat"])
    G = GT[:, :X.shape[1]] if cfg["gtmix"] else None   # GT 特徴は candE と同じ列順（t3_search2 で assert 済み）
    oof = np.zeros((len(df), len(STATIONS)))
    t0 = time.time()
    for k in range(5):
        tr, va = (df.fold != k).values, (df.fold == k).values
        Xtr, Ytr = X[tr], Y[tr]
        if G is not None:
            Xtr, Ytr = np.vstack([Xtr, G[tr]]), np.vstack([Ytr, Y[tr]])
        m = xgb.XGBClassifier(n_estimators=cfg["n"], learning_rate=cfg["lr"], max_depth=cfg["depth"],
                              subsample=cfg["sub"], colsample_bytree=cfg["col"], min_child_weight=cfg["mcw"],
                              reg_lambda=cfg["lam"], reg_alpha=cfg["alpha"], gamma=cfg["gamma"], tree_method="hist", device=device, verbosity=0,
                              random_state=seed, multi_strategy=cfg["strategy"])
        m.fit(Xtr, Ytr)
        oof[va] = np.asarray(m.predict_proba(X[va]), dtype=np.float32)
    np.save(p, oof)
    print(f"  {name:16s} seed{seed} {time.time() - t0:.0f}s", flush=True)
    return oof


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["run", "report"], default="run")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--only", nargs="*", default=None)
    ap.add_argument("--seeds", type=int, nargs="*", default=[42])
    args = ap.parse_args()
    df, feats, Y = load_df()
    GT, _ = load_gt_feats(df)
    sc = Scorer(df, Y)
    if args.stage == "run":
        names = args.only or list(VARIANTS)
        for name in names:
            cfg = {**BASE, **VARIANTS[name]}
            for seed in args.seeds:
                oof = run_one(name, cfg, df, feats, Y, GT, args.device, seed)
                f1, au = sc(oof, 1.0)
                print(f"{name:16s} seed{seed}  F1 {f1:.4f} AUROC {au:.4f} sum {f1 + au:.4f}", flush=True)
    rows = []
    for p in sorted(OUT.glob("*.npy")):
        f1, au = sc(np.load(p), 1.0)
        rows.append({"variant": p.stem, "f1": f1, "auroc": au, "sum": round(f1 + au, 4)})
    res = pd.DataFrame(rows).sort_values("sum", ascending=False)
    res.to_csv(HERE / "t3_xgb_sweep.csv", index=False)
    print("\n=== XGB 単体（nested 較正 λ=1.0、F1+AUROC 順）===")
    print(res.to_string(index=False))


if __name__ == "__main__":
    main()
