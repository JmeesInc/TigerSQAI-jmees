"""expT04: Task3 の 4 構成を **同一 fold・同一較正**で比較する.

  (i)   mlp_gap        : 凍結 encoder の GAP -> MLP            （expT03 の honest 版）
  (ii)  mlp_gap_inv    : GAP + 解剖在庫 315 次元 -> MLP        （特徴を CNN head に混ぜる）
  (iii) lgb_inv_gap    : 在庫 + GAP を LightGBM の特徴に       （CNN を LGBM に入れる）
  (iv)  lgb_inv        : 在庫のみ LightGBM                      （現行の主力）

すべて `cache_feats.py` の **fold ごとの凍結特徴**を使う:
  fold k のモデルで作った特徴に対し、学習は fold!=k、評価は fold==k。
  → 評価対象の行は必ず「その case を見ていない encoder」の特徴になる。

**epoch 選択は最終 epoch 固定**（expT03 は val 最良 epoch を選んでいたため OOF が楽観的だった）。
較正は fold ごとに他 fold で閾値を推定する nested 方式。

Usage:
    python3 fusion_compare.py --encoder expA23_l_dicedet
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

REPO = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO / "reference" / "tigersqai_challenge"))

STATIONS = ["6L", "6R", "7L", "7R", "8", "9", "10L", "10R", "11L", "11R", "12L", "12R", "13L", "13R"]


def official_eval(pred_csv: Path) -> dict:
    from metrics.classes_stations import CLASSES_STATIONS
    from metrics.evaluate_cls import evaluate
    return evaluate(gt_csv=REPO / "workspace/data_proc/task3_gt_wide.csv",
                    pred_csv=pred_csv, classes=CLASSES_STATIONS)


def best_threshold(y: np.ndarray, p: np.ndarray) -> float:
    from sklearn.metrics import f1_score
    if y.sum() in (0, len(y)):
        return 0.5
    cands = np.unique(np.clip(p, 1e-4, 1 - 1e-4))
    if len(cands) > 200:
        cands = np.quantile(cands, np.linspace(0.01, 0.99, 200))
    best, bt = -1.0, 0.5
    for t in cands:
        f = f1_score(y, (p >= t).astype(int), zero_division=0)
        if f > best:
            best, bt = f, float(t)
    return bt


def remap(p: np.ndarray, t: float) -> np.ndarray:
    t = float(np.clip(t, 1e-3, 1 - 1e-3))
    return np.where(p < t, 0.5 * p / t, 0.5 + 0.5 * (p - t) / (1 - t)).clip(0, 1)


def train_mlp(Xtr, Ytr, Xva, epochs=60, hidden=256, seed=42, device="cuda"):
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
    net.eval()
    with torch.no_grad():
        return torch.sigmoid(net(torch.tensor(Xva, dtype=torch.float32, device=device))).cpu().numpy()


def fit_lgb(Xtr, ytr, Xva, seed=42):
    import lightgbm as lgb
    if ytr.sum() == 0:
        return np.zeros(len(Xva))
    if ytr.sum() == len(ytr):
        return np.ones(len(Xva))
    clf = lgb.LGBMClassifier(n_estimators=300, learning_rate=0.05, num_leaves=7,
                             min_child_samples=20, subsample=0.8, subsample_freq=1,
                             colsample_bytree=0.5, reg_lambda=1.0, random_state=seed, verbose=-1)
    clf.fit(Xtr, ytr)
    return clf.predict_proba(Xva)[:, 1]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--encoder", default="expA23_l_dicedet")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--features", default="features_ens5.csv",
                    help="在庫特徴。**コンテナが出すのと同じアンサンブルの予測マスク**由来のものを使う")
    ap.add_argument("--tag", default="")
    args = ap.parse_args()

    gt = pd.read_csv(REPO / "workspace/data_proc/task3_gt_wide.csv")
    folds = pd.read_csv(REPO / "workspace/fold/v2/folds.csv")
    stem2fold = {f.rsplit(".", 1)[0]: fo for f, fo in zip(folds.filename, folds.fold)}
    gt["fold"] = gt.case_id.map(stem2fold)
    inv = pd.read_csv(HERE / args.features)
    df = gt.merge(inv, on="case_id", how="inner").reset_index(drop=True)
    inv_cols = [c for c in inv.columns if c != "case_id"]
    Y = df[STATIONS].values.astype(np.float32)
    INV = df[inv_cols].values.astype(np.float32)
    n = len(df)

    feat_dir = HERE / "feats" / args.encoder
    cache = {}
    for k in range(5):
        z = np.load(feat_dir / f"fold{k}.npz", allow_pickle=True)
        pos = {nm: i for i, nm in enumerate(z["names"])}
        idx = np.array([pos[c] for c in df.case_id])
        cache[k] = (z["feat"][idx], z["feat_flip"][idx])
    dim = cache[0][0].shape[1]
    print(f"{n} 行 / 在庫 {len(inv_cols)} 次元 / GAP {dim} 次元")

    # 標準化は fold ごとに学習側で決める
    variants = ["mlp_gap", "mlp_gap_inv", "lgb_inv_gap", "lgb_inv"]
    oof = {v: np.zeros((n, len(STATIONS))) for v in variants}
    for k in range(5):
        tr = (df.fold != k).values
        va = (df.fold == k).values
        gap, gap_f = cache[k]
        # 学習側は反転版も足して 2 倍に（aug の代わり）
        for v in variants:
            if v.startswith("mlp"):
                base_tr = [gap[tr], gap_f[tr]]
                base_va = gap[va]
                if v == "mlp_gap_inv":
                    base_tr = [np.hstack([g, INV[tr]]) for g in base_tr]
                    base_va = np.hstack([base_va, INV[va]])
                Xtr = np.vstack(base_tr)
                Ytr = np.vstack([Y[tr], Y[tr]])
                mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-6
                oof[v][va] = train_mlp((Xtr - mu) / sd, Ytr, (base_va - mu) / sd,
                                       device=args.device)
            else:
                Xtr = np.hstack([INV[tr], gap[tr]]) if v == "lgb_inv_gap" else INV[tr]
                Xva = np.hstack([INV[va], gap[va]]) if v == "lgb_inv_gap" else INV[va]
                for j, st in enumerate(STATIONS):
                    oof[v][va, j] = fit_lgb(Xtr, Y[tr, j], Xva)
        print(f"fold{k} 完了")

    res = {}
    for v in variants:
        out = pd.DataFrame(oof[v], columns=STATIONS)
        out.insert(0, "case_id", df.case_id.values)
        p = HERE / f"oof_fusion_{args.tag}{v}.csv"
        out.to_csv(p, index=False)
        r = official_eval(p)
        # nested 較正
        cal = out.copy().set_index("case_id")
        for k in range(5):
            tr, va = (df.fold != k).values, (df.fold == k).values
            for j, st in enumerate(STATIONS):
                t = best_threshold(Y[tr, j], oof[v][tr, j])
                cal.loc[df.case_id.values[va], st] = remap(oof[v][va, j], t)
        pc = HERE / f"oof_fusion_{args.tag}{v}_cal.csv"
        cal.reset_index().to_csv(pc, index=False)
        rc = official_eval(pc)
        res[v] = {"f1": round(r["final_f1"], 4), "auroc": round(r["final_auroc"], 4),
                  "f1_cal": round(rc["final_f1"], 4), "auroc_cal": round(rc["final_auroc"], 4)}
        print(f"{v:14s} F1={r['final_f1']:.4f} AUROC={r['final_auroc']:.4f} "
              f"| 較正後 F1={rc['final_f1']:.4f} AUROC={rc['final_auroc']:.4f}")
    (HERE / f"fusion_compare{args.tag and '_' + args.tag.rstrip('_')}.json").write_text(
        json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
