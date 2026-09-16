"""expT04: Task3 最終モデル v12 の書き出し（9/16 再探索の結論をそのまま再現する）.

構成（t3_search.py / t3_search2.py / t3_xgb_sweep.py で確定、OOF 0.8056 / 0.9253）:
  * GBDT 枝: XGBoost vector-leaf (multi_output_tree), depth 6, n=1000, lr 0.02, colsample 0.5,
             subsample 0.8, λ=1, 入力 = 在庫 315 + 解剖文脈 202 + 粗グリッド 1224 = 1741 次元、seed 3
  * MLP 枝:  d_deeplabv3p の fold encoder GAP (1536) + 在庫 315、hidden 256、seed 3
  * 合成:    w_gbdt = 0.7、閾値較正 λ = 1.0（全 OOF から station ごとに 1 組）
  * fold ごとに学習し、推論時は 5 fold 平均

出力: model_t3_v12/
  mlp_d_deeplabv3p_s<seed>_fold<k>.pt   (state_dict + mu/sd + encoder/fold)
  xgb_s<seed>_fold<k>.json              (XGBClassifier.save_model)
  meta.json  features(1741, GBDT 用) / features_mlp(315) / w_gbdt / thresholds / encoders

検証: 書き出した fold-k モデルで fold-k 行を推論し、探索時の member OOF（members/*.npy）と一致することを確認する。
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(REPO / "reference" / "tigersqai_challenge"))
from export_t3_final import train_mlp_keep  # noqa: E402
from fusion_compare import STATIONS  # noqa: E402
from t3_search import MEM, Scorer, fast_best_threshold, load_df  # noqa: E402
from t3_search2 import load_gap_one  # noqa: E402

ENC = "expA23_d_deeplabv3p"
XGB_PARAMS = dict(n_estimators=1000, learning_rate=0.02, max_depth=6, subsample=0.8, colsample_bytree=0.5,
                  reg_lambda=1.0, tree_method="hist", verbosity=0, multi_strategy="multi_output_tree")


def feature_names():
    c315 = [c for c in pd.read_csv(HERE / "features_candE_fix.csv", nrows=1).columns if c != "case_id"]
    extra = []
    for f in ("features_anat_candE.csv", "features_grid_candE.csv"):
        extra += [c for c in pd.read_csv(HERE / f, nrows=1).columns if c != "case_id"]
    return c315, c315 + extra


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(HERE / "model_t3_v12"))
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--w-gbdt", type=float, default=0.7)
    args = ap.parse_args()
    import xgboost as xgb
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    df, feats, Y = load_df()
    cols315, cols1741 = feature_names()
    assert feats["f315"].shape[1] == len(cols315) and feats["f1741"].shape[1] == len(cols1741)
    X315, X1741 = feats["f315"], feats["f1741"]
    gap = load_gap_one(df, "d_deeplabv3p")
    n = len(df)
    oof_mlp = np.zeros((args.seeds, n, len(STATIONS)))
    oof_xgb = np.zeros((args.seeds, n, len(STATIONS)))

    for k in range(5):
        tr, va = (df.fold != k).values, (df.fold == k).values
        g, gf = gap[k]
        for seed in range(args.seeds):
            Xtr = np.vstack([np.hstack([g[tr], X315[tr]]), np.hstack([gf[tr], X315[tr]])])
            Ytr = np.vstack([Y[tr], Y[tr]])
            mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-6
            net = train_mlp_keep((Xtr - mu) / sd, Ytr, seed=42 + seed, device=args.device)
            torch.save({"state_dict": {kk: v.cpu() for kk, v in net.state_dict().items()},
                        "mu": mu.astype(np.float32), "sd": sd.astype(np.float32),
                        "encoder": ENC, "seed": seed, "fold": k},
                       out / f"mlp_d_deeplabv3p_s{seed}_fold{k}.pt")
            with torch.no_grad():
                xv = torch.tensor((np.hstack([g[va], X315[va]]) - mu) / sd, dtype=torch.float32, device=args.device)
                oof_mlp[seed, va] = torch.sigmoid(net(xv)).cpu().numpy()
            m = xgb.XGBClassifier(**XGB_PARAMS, device=args.device, random_state=42 + seed)
            m.fit(X1741[tr], Y[tr])
            m.save_model(str(out / f"xgb_s{seed}_fold{k}.json"))
            oof_xgb[seed, va] = np.asarray(m.predict_proba(X1741[va]), dtype=np.float32)
        print(f"fold{k} 完了", flush=True)

    # --- 探索時 member との一致確認 ---
    for seed in range(args.seeds):
        for nm, arr in ((f"mlp_d_f315_s{seed}", oof_mlp[seed]), (f"xgbv_f1741_s{seed}", oof_xgb[seed])):
            ref = MEM / f"{nm}.npy"
            if ref.exists():
                d = float(np.abs(np.load(ref) - arr).max())
                print(f"  {nm}: 探索時 OOF との max|Δp| = {d:.6f}")
    # --- 書き出したファイルから読み直して同じ値が出るか（コンテナと同じ経路） ---
    k, seed = 0, 0
    va = (df.fold == k).values
    o = torch.load(out / f"mlp_d_deeplabv3p_s{seed}_fold{k}.pt", map_location="cpu", weights_only=False)
    sdict = o["state_dict"]
    net = torch.nn.Sequential(torch.nn.Linear(sdict["0.weight"].shape[1], sdict["0.weight"].shape[0]), torch.nn.ReLU(),
                              torch.nn.Dropout(0.3), torch.nn.Linear(sdict["3.weight"].shape[1], sdict["3.weight"].shape[0]))
    net.load_state_dict(sdict); net.eval()
    with torch.no_grad():
        x = (np.hstack([gap[k][0][va], X315[va]]) - o["mu"]) / o["sd"]
        p = torch.sigmoid(net(torch.tensor(x, dtype=torch.float32))).numpy()
    print(f"  再読込 MLP fold0 s0: max|Δp| = {np.abs(p - oof_mlp[seed, va]).max():.6f}")
    m2 = xgb.XGBClassifier(); m2.load_model(str(out / f"xgb_s{seed}_fold{k}.json")); m2.set_params(device="cpu")
    p2 = np.asarray(m2.predict_proba(X1741[va]), dtype=np.float32)
    print(f"  再読込 XGB fold0 s0 (CPU 推論): max|Δp| = {np.abs(p2 - oof_xgb[seed, va]).max():.6f}")

    # --- ブレンド OOF、閾値、スコア ---
    P = args.w_gbdt * oof_xgb.mean(0) + (1 - args.w_gbdt) * oof_mlp.mean(0)
    sc = Scorer(df, Y)
    f1, au = sc(P, 1.0)
    print(f"OOF nested 較正 λ=1.0: F1 {f1:.4f} AUROC {au:.4f} (探索時 0.8056 / 0.9253)")
    thr = {st: float(fast_best_threshold(Y[:, j], P[:, j])) for j, st in enumerate(STATIONS)}
    raw = pd.DataFrame(P, columns=STATIONS); raw.insert(0, "case_id", df.case_id.values)
    raw.to_csv(HERE / "oof_v12_raw.csv", index=False)
    meta = {
        "version": "v12", "gbdt": "xgboost", "xgb_params": XGB_PARAMS,
        "features": cols1741, "features_mlp": cols315, "stations": STATIONS,
        "encoders": [ENC], "seeds": args.seeds, "xgb_seeds": args.seeds,
        "w_gbdt": args.w_gbdt, "w_lgb": args.w_gbdt,   # w_lgb は旧 process.py 互換キー
        "thresholds": thr, "threshold_lambda": 1.0,
        "threshold_source": "oof_v12_raw.csv (未較正ブレンド, 全 OOF, λ=1.0)",
        "n_mlp": 5 * args.seeds, "n_xgb": 5 * args.seeds,
        "oof_f1_nested": round(f1, 4), "oof_auroc_nested": round(au, 4),
        "note": ("推論: d_deeplabv3p の GAP + features_mlp(315) -> MLP(5fold x 3seed 平均)、"
                 "features(1741) -> XGBoost vector-leaf(5fold x 3seed 平均) を w_gbdt:1-w_gbdt で合成し thresholds で較正"),
        "feature_csvs": ["features_candE_fix.csv", "features_anat_candE.csv", "features_grid_candE.csv"],
    }
    (out / "meta.json").write_text(json.dumps(meta, indent=2))
    size = sum(p.stat().st_size for p in out.glob("*")) / 2 ** 20
    print(f"-> {out}  ({size:.0f} MB)")


if __name__ == "__main__":
    main()
