"""expT04: Task3 アンサンブル構成の再探索（t3_ensemble.py で抜けていた軸を埋める）.

t3_ensemble.py が探索していたのは **w_lgb 5 点だけ**で、以下は固定だった:
  * MLP の入力特徴 = LGBM と同じ (315 or 1741)。1741 にすると MLP 単体の AUROC が
    0.907 -> 0.891 に落ちるのに、そのまま採用されていた
  * encoder 3 本固定。コンテナは予算制御で priority 順 (l -> d -> e) に本数を減らすので
    1 encoder / 2 encoder 構成の OOF が無い
  * GBDT 枝 = LightGBM のみ。gbdt_compare で XGBoost vector-leaf が単体で上回っていたのに未統合
  * 評価は nested 較正 λ=1.0。出荷 (export_t3_final.py) は λ=0.5 収縮なので、
    出荷構成そのものの OOF 値が測られていない

ここでは member ごとの OOF を `members/` にキャッシュし、上の軸を総当たりで測る。
Usage:
    CUDA_VISIBLE_DEVICES=0 python3 t3_search.py --stage members   # OOF 生成（GPU）
    python3 t3_search.py --stage grid                               # 集計のみ
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
from fusion_compare import STATIONS, best_threshold, fit_lgb, official_eval, remap, train_mlp  # noqa: E402

ENCODERS = ["expA23_l_dicedet", "expA23_d_deeplabv3p", "expA23_e_convnext_xl_384"]
SHORT = {"expA23_l_dicedet": "l", "expA23_d_deeplabv3p": "d", "expA23_e_convnext_xl_384": "e"}
MEM = HERE / "members"
MEM.mkdir(exist_ok=True)


def load_df():
    gt = pd.read_csv(REPO / "workspace/data_proc/task3_gt_wide.csv")
    folds = pd.read_csv(REPO / "workspace/fold/v2/folds.csv")
    s2f = {f.rsplit(".", 1)[0]: fo for f, fo in zip(folds.filename, folds.fold)}
    gt["fold"] = gt.case_id.map(s2f)
    inv = pd.read_csv(HERE / "features_candE_fix.csv")
    cols315 = [c for c in inv.columns if c != "case_id"]
    for extra in ("features_anat_candE.csv", "features_grid_candE.csv"):
        inv = inv.merge(pd.read_csv(HERE / extra), on="case_id", how="inner")
    cols1741 = [c for c in inv.columns if c != "case_id"]
    df = gt.merge(inv, on="case_id", how="inner").reset_index(drop=True)
    feats = {"f315": df[cols315].values.astype(np.float32),
             "f1741": df[cols1741].values.astype(np.float32)}
    Y = df[STATIONS].values.astype(np.float32)
    return df, feats, Y


def load_gap(df):
    enc = {}
    for e in ENCODERS:
        c = {}
        for k in range(5):
            z = np.load(HERE / "feats" / e / f"fold{k}.npz", allow_pickle=True)
            pos = {nm: i for i, nm in enumerate(z["names"])}
            idx = np.array([pos[x] for x in df.case_id])
            c[k] = (z["feat"][idx], z["feat_flip"][idx])
        enc[SHORT[e]] = c
    return enc


def member(name, fn):
    p = MEM / f"{name}.npy"
    if p.exists():
        return np.load(p)
    t0 = time.time()
    oof = fn()
    np.save(p, oof)
    print(f"  {name:28s} 生成 {time.time() - t0:.0f}s", flush=True)
    return oof


def build_members(df, feats, Y, device):
    n = len(df)
    enc = load_gap(df)
    out = {}
    # --- MLP: encoder x 特徴セット x seed
    for e, fk, seed in itertools.product(enc, ("f315", "f1741"), range(3)):
        INV = feats[fk]

        def fn(e=e, INV=INV, seed=seed):
            oof = np.zeros((n, len(STATIONS)))
            for k in range(5):
                tr, va = (df.fold != k).values, (df.fold == k).values
                gap, gap_f = enc[e][k]
                Xtr = np.vstack([np.hstack([gap[tr], INV[tr]]), np.hstack([gap_f[tr], INV[tr]])])
                Ytr = np.vstack([Y[tr], Y[tr]])
                Xva = np.hstack([gap[va], INV[va]])
                mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-6
                oof[va] = train_mlp((Xtr - mu) / sd, Ytr, (Xva - mu) / sd, seed=42 + seed, device=device)
            return oof
        out[f"mlp_{e}_{fk}_s{seed}"] = member(f"mlp_{e}_{fk}_s{seed}", fn)
    # --- LightGBM: 特徴セット x 6 seed（t3_ensemble と同じ random_state 系列）
    for fk in ("f315", "f1741"):
        INV = feats[fk]
        for seed, leaves in itertools.product(range(3), (7, 15)):
            rs = 42 + seed * 7 + leaves

            def fn(INV=INV, rs=rs):
                oof = np.zeros((n, len(STATIONS)))
                for k in range(5):
                    tr, va = (df.fold != k).values, (df.fold == k).values
                    for j in range(len(STATIONS)):
                        oof[va, j] = fit_lgb(INV[tr], Y[tr, j], INV[va], seed=rs)
                return oof
            out[f"lgb_{fk}_rs{rs}"] = member(f"lgb_{fk}_rs{rs}", fn)
    # --- XGBoost vector-leaf（gbdt_compare の n=1000 lr=0.02）: 1741 のみ, 3 seed
    import xgboost as xgb
    for seed in range(3):
        def fn(seed=seed):
            oof = np.zeros((n, len(STATIONS)))
            INV = feats["f1741"]
            for k in range(5):
                tr, va = (df.fold != k).values, (df.fold == k).values
                m = xgb.XGBClassifier(n_estimators=1000, learning_rate=0.02, max_depth=6,
                                      subsample=0.8, colsample_bytree=0.5, reg_lambda=1.0,
                                      tree_method="hist", device=device, verbosity=0,
                                      random_state=42 + seed, multi_strategy="multi_output_tree")
                m.fit(INV[tr], Y[tr])
                oof[va] = np.asarray(m.predict_proba(INV[va]), dtype=np.float32)
            return oof
        out[f"xgbv_f1741_s{seed}"] = member(f"xgbv_f1741_s{seed}", fn)
    return out


def fast_best_threshold(y: np.ndarray, p: np.ndarray) -> float:
    """fusion_compare.best_threshold と同じ候補集合・同じ tie-break（先頭優先）のベクトル版"""
    if y.sum() in (0, len(y)):
        return 0.5
    cands = np.unique(np.clip(p, 1e-4, 1 - 1e-4))
    if len(cands) > 200:
        cands = np.quantile(cands, np.linspace(0.01, 0.99, 200))
    pred = p[None, :] >= cands[:, None]
    yb = y.astype(bool)[None, :]
    tp = (pred & yb).sum(1); fp = (pred & ~yb).sum(1); fn = (~pred & yb).sum(1)
    den = 2 * tp + fp + fn
    f = np.where(den > 0, 2 * tp / np.maximum(den, 1), 0.0)
    return float(cands[int(np.argmax(f))])


class Scorer:
    def __init__(self, df, Y):
        self.df, self.Y = df, Y
        self.tmp = HERE / "_tmp_search.csv"

    def calib(self, P, lam):
        """nested: fold k の閾値は他 fold の OOF から推定。λ で 0.5 へ収縮（出荷と同じ）"""
        M = P.copy()
        for k in range(5):
            tr, va = (self.df.fold != k).values, (self.df.fold == k).values
            for j in range(len(STATIONS)):
                t = fast_best_threshold(self.Y[tr, j], P[tr, j])
                M[va, j] = remap(P[va, j], 0.5 + lam * (t - 0.5))
        return M

    def __call__(self, P, lam):
        M = self.calib(P, lam) if lam > 0 else P
        out = pd.DataFrame(M, columns=STATIONS)
        out.insert(0, "case_id", self.df.case_id.values)
        out.to_csv(self.tmp, index=False)
        r = official_eval(self.tmp)
        return round(r["final_f1"], 4), round(r["final_auroc"], 4)


def grid(df, Y, members, lams):
    sc = Scorer(df, Y)
    rows = []

    def add(name, P):
        rec = {"config": name}
        for lam in lams:
            f1, au = sc(P, lam)
            rec[f"f1@λ{lam}"], rec[f"auc@λ{lam}"] = f1, au
            rec[f"sum@λ{lam}"] = round(f1 + au, 4)
        rows.append(rec)
        print(f"{name:46s} " + "  ".join(f"λ{lam}: {rec[f'f1@λ{lam}']:.4f}/{rec[f'auc@λ{lam}']:.4f}" for lam in lams), flush=True)

    mean = lambda keys: np.mean([members[k] for k in keys], axis=0)  # noqa: E731
    print("\n=== 単体・枝ごとの平均 ===")
    for fk in ("f315", "f1741"):
        for e in "lde":
            add(f"MLP[{e}]_{fk}(3seed)", mean([k for k in members if k.startswith(f"mlp_{e}_{fk}_")]))
        add(f"LGB_{fk}(6)", mean([k for k in members if k.startswith(f"lgb_{fk}_")]))
    add("XGBv_f1741(3)", mean([k for k in members if k.startswith("xgbv_")]))
    add("LGB_f1741(6)+XGBv(3) 均等", 0.5 * mean([k for k in members if k.startswith("lgb_f1741_")])
        + 0.5 * mean([k for k in members if k.startswith("xgbv_")]))

    print("\n=== ブレンド総当たり ===")
    gbdt = {
        "LGB1741": mean([k for k in members if k.startswith("lgb_f1741_")]),
        "XGBv1741": mean([k for k in members if k.startswith("xgbv_")]),
        "LGB+XGB": 0.5 * mean([k for k in members if k.startswith("lgb_f1741_")])
        + 0.5 * mean([k for k in members if k.startswith("xgbv_")]),
    }
    enc_sets = ["l", "ld", "lde", "d", "e", "de", "le"]   # 先頭 3 つが priority 順の予算シナリオ
    for gname, G in gbdt.items():
        for fk in ("f315", "f1741"):
            for es in enc_sets:
                M = mean([k for k in members if k.startswith("mlp_") and k.split("_")[1] in es and f"_{fk}_" in k])
                for w in np.round(np.arange(0.3, 1.01, 0.1), 1):
                    add(f"{gname} w={w:.1f} + MLP[{es}]_{fk}", w * G + (1 - w) * M)
    res = pd.DataFrame(rows)
    res.to_csv(HERE / "t3_search_grid.csv", index=False)
    return res


def honest_w(df, Y, members, gname, es, fk, lam, ws=np.round(np.arange(0.3, 1.01, 0.1), 1)):
    """w を「他 4 fold の OOF で最良」→ 当該 fold に適用、で選んだときの成績（w 選択の楽観を除く）"""
    sc = Scorer(df, Y)
    mean = lambda keys: np.mean([members[k] for k in keys], axis=0)  # noqa: E731
    if gname == "LGB1741":
        G = mean([k for k in members if k.startswith("lgb_f1741_")])
    elif gname == "XGBv1741":
        G = mean([k for k in members if k.startswith("xgbv_")])
    else:
        G = 0.5 * mean([k for k in members if k.startswith("lgb_f1741_")]) + 0.5 * mean([k for k in members if k.startswith("xgbv_")])
    M = mean([k for k in members if k.startswith("mlp_") and k.split("_")[1] in es and f"_{fk}_" in k])
    from sklearn.metrics import f1_score, roc_auc_score
    P = np.zeros_like(G)
    chosen = []
    for k in range(5):
        tr, va = (df.fold != k).values, (df.fold == k).values
        best = None
        for w in ws:
            B = w * G + (1 - w) * M
            # 他 4 fold での F1@0.5(較正後) + AUROC の合計で選ぶ（ユーザー指定の選択基準）
            C = sc.calib(B, lam)
            f = np.mean([f1_score(Y[tr, j], (C[tr, j] >= 0.5).astype(int), zero_division=0)
                         for j in range(len(STATIONS))])
            f += np.mean([roc_auc_score(Y[tr, j], B[tr, j]) for j in range(len(STATIONS))
                          if 0 < Y[tr, j].sum() < tr.sum()])
            if best is None or f > best[0]:
                best = (f, w)
        chosen.append(best[1])
        P[va] = (best[1] * G + (1 - best[1]) * M)[va]
    f1, au = sc(P, lam)
    return f1, au, chosen


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["members", "grid", "honest"], default="grid")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--lams", type=float, nargs="*", default=[1.0, 0.5])
    args = ap.parse_args()
    df, feats, Y = load_df()
    print(f"{len(df)} 行, 315/1741 次元, fold 分布 {df.fold.value_counts().sort_index().tolist()}")
    if args.stage == "members":
        build_members(df, feats, Y, args.device)
        return
    members = {p.stem: np.load(p) for p in sorted(MEM.glob("*.npy"))}
    print(f"member {len(members)} 本")
    if args.stage == "grid":
        res = grid(df, Y, members, args.lams)
        for lam in args.lams:
            print(f"\n--- λ={lam} 上位 15 (F1+AUROC 順) ---")
            print(res.sort_values(f"sum@λ{lam}", ascending=False).head(15).to_string(index=False))
    else:
        out = {}
        for gname, es, fk in [("LGB1741", "lde", "f1741"),   # 出荷 v7
                              ("XGBv1741", "d", "f315"), ("XGBv1741", "de", "f315"),
                              ("XGBv1741", "ld", "f315"), ("XGBv1741", "lde", "f315"),
                              ("XGBv1741", "l", "f315"),
                              ("LGB+XGB", "d", "f315"), ("LGB+XGB", "lde", "f315")]:

            for lam in args.lams:
                f1, au, ws = honest_w(df, Y, members, gname, es, fk, lam)
                key = f"{gname} + MLP[{es}]_{fk} λ={lam}"
                out[key] = {"f1": f1, "auroc": au, "w_per_fold": ws}
                print(f"{key:44s} F1 {f1:.4f} AUROC {au:.4f}  w={ws}", flush=True)
        (HERE / "t3_search_honest.json").write_text(json.dumps(out, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()
