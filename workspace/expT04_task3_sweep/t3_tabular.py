"""expT04: 「解剖在庫」特徴だけで Task3 (14 station のマルチラベル) を解く.

expT03 (凍結 CNN + MLP, F1@0.5 0.7070 / AUROC 0.8867) とは **情報源が違う**:
こちらは画像ではなく **seg 予測マスクの構造在庫**しか見ない。
当たれば expT03 とのブレンドで上がるし、外れても「画像特徴の方が強い」ことが確定する。

fold は seg と同じ v2 (case グループ)。OOF 確率を公式 evaluate_cls で採点する。

Usage:
    python3 t3_tabular.py --model lgb           # LightGBM
    python3 t3_tabular.py --model lr            # ロジスティック回帰
    python3 t3_tabular.py --blend ../expT03_task3_noleak/results/oof_task3.csv --w 0.3
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO / "reference" / "tigersqai_challenge"))

STATIONS = ["6L", "6R", "7L", "7R", "8", "9", "10L", "10R", "11L", "11R", "12L", "12R", "13L", "13R"]


def official_eval(pred_csv: Path) -> dict:
    from metrics.classes_stations import CLASSES_STATIONS
    from metrics.evaluate_cls import evaluate
    return evaluate(gt_csv=REPO / "workspace/data_proc/task3_gt_wide.csv",
                    pred_csv=pred_csv, classes=CLASSES_STATIONS)


def fit_predict(model: str, Xtr, ytr, Xva, seed: int = 42) -> np.ndarray:
    if ytr.sum() == 0:
        return np.zeros(len(Xva))
    if ytr.sum() == len(ytr):
        return np.ones(len(Xva))
    if model == "lgb":
        import lightgbm as lgb
        clf = lgb.LGBMClassifier(
            n_estimators=300, learning_rate=0.05, num_leaves=7, min_child_samples=20,
            subsample=0.8, subsample_freq=1, colsample_bytree=0.5, reg_lambda=1.0,
            random_state=seed, verbose=-1)
        clf.fit(Xtr, ytr)
        return clf.predict_proba(Xva)[:, 1]
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    clf = make_pipeline(StandardScaler(),
                        LogisticRegression(C=0.05, max_iter=2000, random_state=seed))
    clf.fit(Xtr, ytr)
    return clf.predict_proba(Xva)[:, 1]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", default=str(HERE / "features_ens5.csv"))
    ap.add_argument("--model", choices=["lgb", "lr"], default="lgb")
    ap.add_argument("--blend", default=None, help="ブレンドする別の OOF 確率 CSV")
    ap.add_argument("--stack", default=None,
                    help="別モデルの OOF 確率を **特徴として** 加える (ブレンドより強いことが多い)")
    ap.add_argument("--w", type=float, default=0.5, help="blend 側の重み")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    feat = pd.read_csv(args.features)
    gt = pd.read_csv(REPO / "workspace/data_proc/task3_gt_wide.csv")
    folds = pd.read_csv(REPO / "workspace/fold/v2/folds.csv")
    stem2fold = {f.rsplit(".", 1)[0]: fo for f, fo in zip(folds.filename, folds.fold)}

    if args.stack:
        st = pd.read_csv(args.stack).rename(columns={s_: f"p_{s_}" for s_ in STATIONS})
        feat = feat.merge(st, on="case_id", how="left")
    df = gt.merge(feat, on="case_id", how="inner")
    df["fold"] = df.case_id.map(stem2fold)
    df["group"] = df.case_id.map(lambda s: re.match(r"center_\d+_case_\d+", s).group(0))
    assert df.fold.notna().all(), "fold 未割当の行がある"
    fcols = [c for c in feat.columns if c != "case_id"]
    print(f"{len(df)} 行 / {len(fcols)} 特徴 / {df.group.nunique()} case")

    oof = np.zeros((len(df), len(STATIONS)))
    for fold in sorted(df.fold.unique()):
        tr, va = df.fold != fold, df.fold == fold
        Xtr, Xva = df.loc[tr, fcols].values, df.loc[va, fcols].values
        for k, st in enumerate(STATIONS):
            oof[va.values, k] = fit_predict(args.model, Xtr, df.loc[tr, st].values, Xva)
        print(f"fold{fold} done ({tr.sum()} train / {va.sum()} val)", flush=True)

    out = pd.DataFrame(oof, columns=STATIONS)
    out.insert(0, "case_id", df.case_id.values)
    tag = args.model + ("_stack" if args.stack else "")
    if args.blend:
        other = pd.read_csv(args.blend).set_index("case_id").reindex(out.case_id)[STATIONS].values
        out[STATIONS] = (1 - args.w) * other + args.w * out[STATIONS].values
        tag += f"_blend{args.w}"
    path = Path(args.out) if args.out else HERE / f"oof_{tag}.csv"
    out.to_csv(path, index=False)
    res = official_eval(path)
    print(f"[{tag}] Weighted F1@0.5 = {res['final_f1']:.4f} / AUROC = {res['final_auroc']:.4f}"
          f"   (expT03 基準 0.7070 / 0.8867)")


if __name__ == "__main__":
    main()
