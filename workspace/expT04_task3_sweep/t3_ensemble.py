"""expT04: Task3 のアンサンブル（シード / encoder / モデル種を束ねる）.

凍結特徴をキャッシュ済みなので **ヘッド 1 本の学習は数秒**。GPU をほとんど使わずに
多様なモデルを量産できる、費用対効果の高い領域。

束ねる軸:
  1. MLP のシード違い（同一 encoder）
  2. **encoder 違い**（l_dicedet / d_deeplabv3p / e_convnext_xl_384 の GAP）
  3. LightGBM のシード・葉数違い

在庫特徴は **コンテナが出すのと同じアンサンブルの予測マスク由来**のものを使う
（`features_candE.csv`）。ここを ens5 由来にすると OOF だけ良くなって本番で再現しない。

Usage:
    python3 t3_ensemble.py --encoders expA23_l_dicedet expA23_d_deeplabv3p --seeds 3
"""

from __future__ import annotations

import argparse
import itertools
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

REPO = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(REPO / "reference" / "tigersqai_challenge"))

from fusion_compare import (STATIONS, best_threshold, fit_lgb, official_eval,  # noqa: E402
                            remap, train_mlp)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", default="features_candE.csv")
    ap.add_argument("--extra-features", nargs="*", default=[],
                    help="在庫に足す特徴 CSV（解剖文脈 features_anat_*.csv / 粗グリッド features_grid_*.csv）")
    ap.add_argument("--encoders", nargs="+", default=["expA23_l_dicedet"])
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--lgb-seeds", type=int, default=3)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--tag", default="ens")
    args = ap.parse_args()

    gt = pd.read_csv(REPO / "workspace/data_proc/task3_gt_wide.csv")
    folds = pd.read_csv(REPO / "workspace/fold/v2/folds.csv")
    s2f = {f.rsplit(".", 1)[0]: fo for f, fo in zip(folds.filename, folds.fold)}
    gt["fold"] = gt.case_id.map(s2f)
    inv = pd.read_csv(HERE / args.features)
    for extra in args.extra_features:
        e = pd.read_csv(HERE / extra)
        inv = inv.merge(e, on="case_id", how="inner", suffixes=("", f"_{Path(extra).stem[-6:]}"))
    df = gt.merge(inv, on="case_id", how="inner").reset_index(drop=True)
    inv_cols = [c for c in inv.columns if c != "case_id"]
    Y = df[STATIONS].values.astype(np.float32)
    INV = df[inv_cols].values.astype(np.float32)
    n = len(df)

    # encoder ごとの fold 別キャッシュ
    enc_cache = {}
    for e in args.encoders:
        d = HERE / "feats" / e
        if not (d / "fold0.npz").exists():
            print(f"skip {e} (キャッシュ無し)")
            continue
        c = {}
        for k in range(5):
            z = np.load(d / f"fold{k}.npz", allow_pickle=True)
            pos = {nm: i for i, nm in enumerate(z["names"])}
            idx = np.array([pos[x] for x in df.case_id])
            c[k] = (z["feat"][idx], z["feat_flip"][idx])
        enc_cache[e] = c
        print(f"{e}: GAP {c[0][0].shape[1]} 次元")
    assert enc_cache, "利用できる encoder キャッシュが無い"

    members: dict[str, np.ndarray] = {}
    # --- MLP: encoder × seed ---
    for e, seed in itertools.product(enc_cache, range(args.seeds)):
        oof = np.zeros((n, len(STATIONS)))
        for k in range(5):
            tr, va = (df.fold != k).values, (df.fold == k).values
            gap, gap_f = enc_cache[e][k]
            Xtr = np.vstack([np.hstack([gap[tr], INV[tr]]), np.hstack([gap_f[tr], INV[tr]])])
            Ytr = np.vstack([Y[tr], Y[tr]])
            Xva = np.hstack([gap[va], INV[va]])
            mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-6
            oof[va] = train_mlp((Xtr - mu) / sd, Ytr, (Xva - mu) / sd,
                                seed=42 + seed, device=args.device)
        members[f"mlp_{e[7:]}_s{seed}"] = oof
        print(f"  mlp {e[7:]} seed{seed} 完了")
    # --- LGBM: seed × 葉数 ---
    for seed, leaves in itertools.product(range(args.lgb_seeds), (7, 15)):
        oof = np.zeros((n, len(STATIONS)))
        for k in range(5):
            tr, va = (df.fold != k).values, (df.fold == k).values
            for j in range(len(STATIONS)):
                oof[va, j] = fit_lgb(INV[tr], Y[tr, j], INV[va], seed=42 + seed * 7 + leaves)
        members[f"lgb_s{seed}_l{leaves}"] = oof
        print(f"  lgb seed{seed} leaves{leaves} 完了")

    def score(P: np.ndarray, cal: bool):
        M = P.copy()
        if cal:
            for k in range(5):
                tr, va = (df.fold != k).values, (df.fold == k).values
                for j in range(len(STATIONS)):
                    M[va, j] = remap(P[va, j], best_threshold(Y[tr, j], P[tr, j]))
        out = pd.DataFrame(M, columns=STATIONS)
        out.insert(0, "case_id", df.case_id.values)
        p = HERE / f"_tmp_{args.tag}.csv"
        out.to_csv(p, index=False)
        r = official_eval(p)
        return r["final_f1"], r["final_auroc"], M

    print("\n--- 単体 ---")
    for name, P in members.items():
        f1, au, _ = score(P, cal=True)
        print(f"{name:28s} 較正後 F1={f1:.4f} AUROC={au:.4f}")

    # 種類ごとの平均 -> さらに重み付き合成
    mlp_keys = [k for k in members if k.startswith("mlp")]
    lgb_keys = [k for k in members if k.startswith("lgb")]
    MLP = np.mean([members[k] for k in mlp_keys], axis=0)
    LGB = np.mean([members[k] for k in lgb_keys], axis=0)
    print("\n--- 種類別平均 ---")
    for nm, P in (("MLP平均", MLP), ("LGB平均", LGB)):
        f1, au, _ = score(P, cal=True)
        print(f"{nm:28s} 較正後 F1={f1:.4f} AUROC={au:.4f}")

    print("\n--- ブレンド ---")
    best = None
    for w in (0.3, 0.4, 0.5, 0.6, 0.7):
        P = w * LGB + (1 - w) * MLP
        f1, au, M = score(P, cal=True)
        print(f"w_lgb={w:.1f}  較正後 F1={f1:.4f} AUROC={au:.4f}")
        if best is None or f1 > best[1]:
            best = (w, f1, au, M)
    w, f1, au, M = best
    # 生 (未較正) のブレンドも保存する。提出用の閾値はこちらから決めるのが正しい
    raw = pd.DataFrame(w * LGB + (1 - w) * MLP, columns=STATIONS)
    raw.insert(0, "case_id", df.case_id.values)
    raw.to_csv(HERE / f"oof_{args.tag}_raw.csv", index=False)
    out = pd.DataFrame(M, columns=STATIONS)
    out.insert(0, "case_id", df.case_id.values)
    out.to_csv(HERE / f"oof_{args.tag}_best.csv", index=False)
    (HERE / f"{args.tag}_summary.json").write_text(json.dumps(
        {"w_lgb": w, "f1_cal": round(f1, 4), "auroc_cal": round(au, 4),
         "n_mlp": len(mlp_keys), "n_lgb": len(lgb_keys),
         "encoders": list(enc_cache), "features": args.features}, indent=2))
    print(f"\n=== 最良 w_lgb={w} F1={f1:.4f} AUROC={au:.4f} ===")


if __name__ == "__main__":
    main()
