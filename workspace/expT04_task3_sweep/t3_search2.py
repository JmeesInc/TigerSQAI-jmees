"""expT04: Task3 アンサンブルの深掘り探索（t3_search.py の続き）.

追加する軸:
  1. encoder 候補の拡張: feats/ にキャッシュがある全 encoder で MLP(GAP+315) を 3 seed 作る
     （9/13 時点の 3 本は q_* 事前学習レシピが存在する前に選んだもの）
  2. XGBoost の学習データに GT マスク由来の特徴行を混ぜる（mask_mix_aug.py の candE+GT。
     LGBM では candE 評価で +0.0055/+0.0024 だった）。推論は candE 特徴のまま
  3. XGBoost seed 追加（5 seed）と 315 次元版
  4. 全 member に対する greedy forward selection（Caruana、復元あり）。
     選択は **fold ごとに他 4 fold で行い当該 fold に適用**（nested）。基準 = F1@0.5(較正後) + AUROC

Usage:
    CUDA_VISIBLE_DEVICES=1 python3 t3_search2.py --stage members
    python3 t3_search2.py --stage rank      # encoder ごとの単体 / XGB との合成
    python3 t3_search2.py --stage greedy
"""

from __future__ import annotations

import argparse
import itertools
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
from fusion_compare import STATIONS, train_mlp  # noqa: E402
from t3_search import MEM, Scorer, fast_best_threshold, load_df, member, remap  # noqa: E402


def load_gt_feats(df):
    inv = pd.read_csv(HERE / "features_gt_fix.csv")
    for extra in ("features_anat_gt.csv", "features_grid_gt.csv"):
        inv = inv.merge(pd.read_csv(HERE / extra), on="case_id", how="inner")
    inv = inv.set_index("case_id").reindex(df.case_id)
    cols = [c for c in inv.columns]
    return inv[cols].values.astype(np.float32), cols


def all_encoders():
    return sorted(p.name.replace("expA23_", "") for p in (HERE / "feats").iterdir()
                  if (p / "fold4.npz").exists())


def load_gap_one(df, short):
    c = {}
    for k in range(5):
        z = np.load(HERE / "feats" / f"expA23_{short}" / f"fold{k}.npz", allow_pickle=True)
        pos = {nm: i for i, nm in enumerate(z["names"])}
        idx = np.array([pos[x] for x in df.case_id])
        c[k] = (z["feat"][idx], z["feat_flip"][idx])
    return c


def build_members(df, feats, Y, device, xgb_seeds):
    n = len(df)
    INV = feats["f315"]
    # 1. 新 encoder の MLP(GAP+315) x 3 seed
    for e in all_encoders():
        if e in ("l_dicedet", "d_deeplabv3p", "e_convnext_xl_384"):
            continue   # t3_search.py で生成済み
        enc = None
        for seed in range(3):
            name = f"mlp_{e}_f315_s{seed}"
            if (MEM / f"{name}.npy").exists():
                continue
            if enc is None:
                enc = load_gap_one(df, e)

            def fn(enc=enc, seed=seed):
                oof = np.zeros((n, len(STATIONS)))
                for k in range(5):
                    tr, va = (df.fold != k).values, (df.fold == k).values
                    gap, gap_f = enc[k]
                    Xtr = np.vstack([np.hstack([gap[tr], INV[tr]]), np.hstack([gap_f[tr], INV[tr]])])
                    Ytr = np.vstack([Y[tr], Y[tr]])
                    Xva = np.hstack([gap[va], INV[va]])
                    mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-6
                    oof[va] = train_mlp((Xtr - mu) / sd, Ytr, (Xva - mu) / sd, seed=42 + seed, device=device)
                return oof
            member(name, fn)
    # 2-3. XGBoost: 追加 seed / GT 混合 / 315 次元
    import xgboost as xgb
    GT1741, gcols = load_gt_feats(df)
    assert list(gcols) == [c for c in pd.read_csv(HERE / "features_candE_fix.csv").columns if c != "case_id"] \
        + [c for c in pd.read_csv(HERE / "features_anat_candE.csv").columns if c != "case_id"] \
        + [c for c in pd.read_csv(HERE / "features_grid_candE.csv").columns if c != "case_id"], "GT 特徴の列順が candE と違う"

    def xgb_oof(X, seed, gtmix=None):
        oof = np.zeros((n, len(STATIONS)))
        for k in range(5):
            tr, va = (df.fold != k).values, (df.fold == k).values
            Xtr, Ytr = X[tr], Y[tr]
            if gtmix is not None:
                Xtr, Ytr = np.vstack([Xtr, gtmix[tr]]), np.vstack([Ytr, Y[tr]])
            m = xgb.XGBClassifier(n_estimators=1000, learning_rate=0.02, max_depth=6,
                                  subsample=0.8, colsample_bytree=0.5, reg_lambda=1.0,
                                  tree_method="hist", device=device, verbosity=0,
                                  random_state=42 + seed, multi_strategy="multi_output_tree")
            m.fit(Xtr, Ytr)
            oof[va] = np.asarray(m.predict_proba(X[va]), dtype=np.float32)
        return oof
    for seed in range(3, xgb_seeds):
        member(f"xgbv_f1741_s{seed}", lambda seed=seed: xgb_oof(feats["f1741"], seed))
    for seed in range(3):
        member(f"xgbvgt_f1741_s{seed}", lambda seed=seed: xgb_oof(feats["f1741"], seed, gtmix=GT1741))
    for seed in range(3):
        member(f"xgbv_f315_s{seed}", lambda seed=seed: xgb_oof(feats["f315"], seed))


def rank(df, Y, members, lam):
    sc = Scorer(df, Y)
    mean = lambda keys: np.mean([members[k] for k in keys], axis=0)  # noqa: E731
    G = mean([k for k in members if k.startswith("xgbv_f1741_s") and int(k[-1]) < 3])
    rows = []
    print("=== XGBoost 派生（単体）===")
    for nm, keys in (("XGBv1741 s0-2", [k for k in members if k.startswith("xgbv_f1741_s") and int(k[-1]) < 3]),
                     ("XGBv1741 全 seed", [k for k in members if k.startswith("xgbv_f1741_s")]),
                     ("XGBv1741 GT混合 s0-2", [k for k in members if k.startswith("xgbvgt_")]),
                     ("XGBv1741 素+GT混合", [k for k in members if k.startswith("xgbv_f1741_s") or k.startswith("xgbvgt_")]),
                     ("XGBv315 s0-2", [k for k in members if k.startswith("xgbv_f315_")])):
        if keys:
            f1, au = sc(mean(keys), lam)
            print(f"{nm:28s} F1 {f1:.4f} AUROC {au:.4f} sum {f1 + au:.4f}")
    print("\n=== encoder ごと: MLP(GAP+315) 単体 と XGBv(s0-2) w=0.7 との合成 ===")
    for e in all_encoders():
        keys = [k for k in members if k.startswith(f"mlp_{e}_f315_")]
        if not keys:
            continue
        M = mean(keys)
        f1m, aum = sc(M, lam)
        f1b, aub = sc(0.7 * G + 0.3 * M, lam)
        rows.append({"encoder": e, "mlp_f1": f1m, "mlp_auc": aum, "mlp_sum": round(f1m + aum, 4),
                     "blend_f1": f1b, "blend_auc": aub, "blend_sum": round(f1b + aub, 4)})
        print(f"{e:22s} MLP {f1m:.4f}/{aum:.4f} ({f1m + aum:.4f})   XGB.7+MLP {f1b:.4f}/{aub:.4f} ({f1b + aub:.4f})", flush=True)
    res = pd.DataFrame(rows).sort_values("blend_sum", ascending=False)
    res.to_csv(HERE / "t3_search2_rank.csv", index=False)
    print("\n" + res.to_string(index=False))
    # 上位 encoder を 1,2,3,4 本まとめたときの合成
    print("\n=== 上位 encoder を積み上げ（XGBv s0-2 w=0.7）===")
    top = res.encoder.tolist()
    for m_ in range(1, min(6, len(top)) + 1):
        M = mean([k for k in members if k.startswith("mlp_") and "_f315_" in k and k.split("_f315_")[0][4:] in top[:m_]])
        for w in (0.6, 0.7, 0.8):
            f1, au = sc(w * G + (1 - w) * M, lam)
            print(f"top{m_} {top[:m_]} w={w}: {f1:.4f}/{au:.4f} sum {f1 + au:.4f}")


def greedy(df, Y, members, lam, n_iter=30):
    """Caruana greedy（復元あり）。fold ごとに他 4 fold で選び当該 fold に適用（nested）。"""
    from sklearn.metrics import f1_score, roc_auc_score
    sc = Scorer(df, Y)
    names = sorted(members)
    P = np.stack([members[k] for k in names])          # (m, n, 14)

    def crit(B, rows):
        """rows 内での F1@0.5(較正後) + AUROC。較正閾値も rows 内で推定（選択専用の簡易版）"""
        f = au = 0.0
        for j in range(len(STATIONS)):
            y, p = Y[rows, j], B[rows, j]
            if 0 < y.sum() < len(y):
                t = fast_best_threshold(y, p)
                q = remap(p, 0.5 + lam * (t - 0.5))
                f += f1_score(y, (q >= 0.5).astype(int), zero_division=0)
                au += roc_auc_score(y, p)
        return (f + au) / len(STATIONS)

    def run(rows):
        chosen, cur, best = [], None, -1
        for _ in range(n_iter):
            cand = None
            for i in range(len(names)):
                B = P[i] if cur is None else (cur * len(chosen) + P[i]) / (len(chosen) + 1)
                s = crit(B, rows)
                if cand is None or s > cand[0]:
                    cand = (s, i, B)
            if cand[0] <= best + 1e-6 and chosen:
                break
            best, cur = cand[0], cand[2]
            chosen.append(cand[1])
        return chosen, best

    oof = np.zeros_like(P[0])
    picks = {}
    for k in range(5):
        tr, va = (df.fold != k).values, (df.fold == k).values
        chosen, s = run(tr)
        w = pd.Series([names[i] for i in chosen]).value_counts()
        picks[f"fold{k}"] = {kk: int(v) for kk, v in w.items()}
        oof[va] = np.mean([P[i] for i in chosen], axis=0)[va]
        print(f"fold{k}: 選択 {len(chosen)} 手 (train 基準 {s:.4f}) -> " + ", ".join(f"{kk}x{v}" for kk, v in w.items()), flush=True)
    f1, au = sc(oof, lam)
    print(f"\nnested greedy: F1 {f1:.4f} AUROC {au:.4f} sum {f1 + au:.4f}")
    chosen, s = run(np.ones(len(df), bool))
    w = pd.Series([names[i] for i in chosen]).value_counts()
    f1u, auu = sc(np.mean([P[i] for i in chosen], axis=0), lam)
    print(f"non-nested greedy（楽観上限）: F1 {f1u:.4f} AUROC {auu:.4f} sum {f1u + auu:.4f} -> " + ", ".join(f"{kk}x{v}" for kk, v in w.items()))
    (HERE / "t3_search2_greedy.json").write_text(json.dumps(
        {"lam": lam, "nested": {"f1": f1, "auroc": au, "picks": picks},
         "full": {"f1": f1u, "auroc": auu, "picks": {kk: int(v) for kk, v in w.items()}}}, indent=1, ensure_ascii=False))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["members", "rank", "greedy"], default="rank")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--lam", type=float, default=1.0)
    ap.add_argument("--xgb-seeds", type=int, default=5)
    args = ap.parse_args()
    df, feats, Y = load_df()
    if args.stage == "members":
        build_members(df, feats, Y, args.device, args.xgb_seeds)
        return
    members = {p.stem: np.load(p) for p in sorted(MEM.glob("*.npy"))}
    print(f"member {len(members)} 本, encoder {all_encoders()}")
    if args.stage == "rank":
        rank(df, Y, members, args.lam)
    else:
        greedy(df, Y, members, args.lam)


if __name__ == "__main__":
    main()
