"""expT04: Task3 の最終モデル一式を提出用に書き出す.

OOF で確定した構成をそのまま再現する:
  * MLP 枝: 3 encoder (l_dicedet / d_deeplabv3p / e_convnext_xl_384) x 3 seed
            入力 = その encoder の GAP + 解剖在庫 315 次元
  * LGBM 枝: 在庫のみ、seed 3 x 葉数 2 = 6 構成
  * 両枝を w_lgb : (1-w_lgb) で合成し、クラス別閾値で較正
  * fold ごとに学習したものを **推論時は 5 fold 平均**で使う

在庫特徴は **コンテナが出すのと同じアンサンブルの予測マスク由来**（features_candE.csv）。

出力: model_t3/
  mlp_<enc>_s<seed>_fold<k>.pt   (state_dict + 正規化 mu/sd)
  lgb_s<seed>_l<leaves>_fold<k>_<station>.txt
  meta.json (特徴名の順序 / 閾値 / w_lgb / encoder 名)
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
import torch.nn as nn

REPO = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(REPO / "reference" / "tigersqai_challenge"))

from fusion_compare import STATIONS, best_threshold  # noqa: E402

ENCODERS = ["expA23_l_dicedet", "expA23_d_deeplabv3p", "expA23_e_convnext_xl_384"]


def train_mlp_keep(Xtr, Ytr, epochs=60, hidden=256, seed=42, device="cuda"):
    torch.manual_seed(seed)
    net = nn.Sequential(nn.Linear(Xtr.shape[1], hidden), nn.ReLU(inplace=True),
                        nn.Dropout(0.3), nn.Linear(hidden, len(STATIONS))).to(device)
    opt = torch.optim.AdamW(net.parameters(), lr=1e-3, weight_decay=1e-2)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    xt = torch.tensor(Xtr, dtype=torch.float32, device=device)
    yt = torch.tensor(Ytr, dtype=torch.float32, device=device)
    for _ in range(epochs):
        net.train()
        perm = torch.randperm(len(xt), device=device)
        for i in range(0, len(xt), 32):
            idx = perm[i:i + 32]
            loss = nn.functional.binary_cross_entropy_with_logits(net(xt[idx]), yt[idx])
            opt.zero_grad(); loss.backward(); opt.step()
        sch.step()
    return net.eval()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", default="features_candE.csv")
    ap.add_argument("--extra-features", nargs="*", default=[],
                    help="在庫に足す特徴 CSV（解剖文脈 / 粗グリッド）。t3_ensemble.py と同じ結合順")
    ap.add_argument("--threshold-lambda", type=float, default=0.5,
                    help="最適閾値を 0.5 へ収縮する係数（1.0=最適そのまま, 0=全て0.5）")
    ap.add_argument("--oof", default="oof_ens_3enc_best.csv", help="閾値を決める OOF 確率")
    ap.add_argument("--w-lgb", type=float, default=0.4)
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--lgb-seeds", type=int, default=3)
    ap.add_argument("--out", default=str(HERE / "model_t3"))
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    import lightgbm as lgb
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

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

    enc_cache = {}
    for e in ENCODERS:
        c = {}
        for k in range(5):
            z = np.load(HERE / "feats" / e / f"fold{k}.npz", allow_pickle=True)
            pos = {nm: i for i, nm in enumerate(z["names"])}
            idx = np.array([pos[x] for x in df.case_id])
            c[k] = (z["feat"][idx], z["feat_flip"][idx])
        enc_cache[e] = c

    n_mlp = n_lgb = 0
    for k in range(5):
        tr = (df.fold != k).values
        for e, seed in itertools.product(ENCODERS, range(args.seeds)):
            gap, gap_f = enc_cache[e][k]
            Xtr = np.vstack([np.hstack([gap[tr], INV[tr]]), np.hstack([gap_f[tr], INV[tr]])])
            Ytr = np.vstack([Y[tr], Y[tr]])
            mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-6
            net = train_mlp_keep((Xtr - mu) / sd, Ytr, seed=42 + seed, device=args.device)
            torch.save({"state_dict": {kk: v.cpu() for kk, v in net.state_dict().items()},
                        "mu": mu.astype(np.float32), "sd": sd.astype(np.float32),
                        "encoder": e, "seed": seed, "fold": k},
                       out / f"mlp_{e.replace('expA23_', '')}_s{seed}_fold{k}.pt")
            n_mlp += 1
        for seed, leaves in itertools.product(range(args.lgb_seeds), (7, 15)):
            for j, st in enumerate(STATIONS):
                y = Y[tr, j]
                name = f"lgb_s{seed}_l{leaves}_fold{k}_{st}"
                if y.sum() in (0, len(y)):
                    (out / f"{name}.const").write_text(str(float(y.mean())))
                    continue
                # ★ OOF を作った fusion_compare.fit_lgb は num_leaves=7 を固定していて
                #   `leaves` は **random_state を散らすためだけ**に使われていた。
                #   ここで num_leaves=leaves にすると検証していない設定を出荷することになる
                #   （fold0 で OOF 生確率と max|Δp| 0.043 のズレとして検出）。OOF 側に合わせる。
                clf = lgb.LGBMClassifier(n_estimators=300, learning_rate=0.05, num_leaves=7,
                                         min_child_samples=20, subsample=0.8, subsample_freq=1,
                                         colsample_bytree=0.5, reg_lambda=1.0,
                                         random_state=42 + seed * 7 + leaves, verbose=-1)
                clf.fit(INV[tr], y)
                clf.booster_.save_model(str(out / f"{name}.txt"))
                n_lgb += 1
        print(f"fold{k} 完了")

    # 閾値は **全 OOF** から 1 組（提出時は単調写像なので AUROC 不変）
    # 閾値は **未較正 (raw) の OOF** から決める。process.py は raw 確率に写像を掛けるので、
    # 較正済み OOF から決めると二重較正になる（9/13 に踏んだバグ）。
    oof = pd.read_csv(HERE / args.oof).set_index("case_id").reindex(df.case_id)[STATIONS].values
    lam = args.threshold_lambda
    thr = {st: 0.5 + lam * (best_threshold(Y[:, j], oof[:, j]) - 0.5)
           for j, st in enumerate(STATIONS)}
    meta = {
        "features": inv_cols, "stations": STATIONS, "encoders": ENCODERS,
        "seeds": args.seeds, "lgb_seeds": args.lgb_seeds, "lgb_leaves": [7, 15],  # random_state のバリエーション。num_leaves は常に 7
        "lgb_num_leaves": 7,
        "w_lgb": args.w_lgb, "thresholds": thr, "n_mlp": n_mlp, "n_lgb": n_lgb,
        "note": ("推論: 各 encoder の GAP + 特徴 -> MLP(5fold x seed 平均) と "
                 "特徴 -> LGBM(5fold x 構成 平均) を w_lgb で合成し、thresholds で較正"),
        "threshold_source": f"{args.oof} (未較正) + lambda={lam} 収縮",
        "threshold_lambda": lam,
        "feature_csvs": [args.features] + list(args.extra_features),
    }
    (out / "meta.json").write_text(json.dumps(meta, indent=2))
    size = sum(p.stat().st_size for p in out.glob("*")) / 2 ** 20
    print(f"MLP {n_mlp} 本 / LGBM {n_lgb} 本 / 合計 {size:.0f} MB -> {out}")


if __name__ == "__main__":
    main()
