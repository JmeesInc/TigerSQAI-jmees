"""expT04: 提出用に Task3 の最終モデル一式を書き出す.

提出時の推論は次の順で行う（OOF の構成をそのまま再現する）:

  1. seg アンサンブルで task1/task2 マスクを予測
  2. `features_from_masks.mask_features` と同じ 235 次元の解剖在庫特徴を作る
  3. **fold ごとに学習した LGBM 5 本の平均**で station 確率を出す
  4. CNN ヘッド (expT03) の確率とブレンド (既定 w=0.6 が LGBM 側)
  5. `thresholds.json` の t_c で区分線形に較正して 0.5 基準に合わせる

fold ごとの 5 本を平均するのは、OOF で測った構成 (= 各行を「その case を見ていない
モデル」で予測) と学習量を揃えるため。全データ 1 本にすると OOF より強いモデルになるが、
較正閾値とブレンド重みを OOF で決めている以上、条件を揃えた方が安全。

Usage:
    python3 fit_final.py --w 0.6
出力: models/lgb_fold{0..4}_{station}.txt, models/meta.json
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
STATIONS = ["6L", "6R", "7L", "7R", "8", "9", "10L", "10R", "11L", "11R", "12L", "12R", "13L", "13R"]
LGB_PARAMS = dict(n_estimators=300, learning_rate=0.05, num_leaves=7, min_child_samples=20,
                  subsample=0.8, subsample_freq=1, colsample_bytree=0.5, reg_lambda=1.0,
                  random_state=42, verbose=-1)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", default=str(HERE / "features_ens5.csv"))
    ap.add_argument("--w", type=float, default=0.6, help="LGBM 側のブレンド重み")
    ap.add_argument("--thresholds", default=str(HERE / "thresholds_oof_blend0.6.json"))
    ap.add_argument("--out", default=str(HERE / "models"))
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(exist_ok=True)
    feat = pd.read_csv(args.features)
    gt = pd.read_csv(REPO / "workspace/data_proc/task3_gt_wide.csv")
    folds = pd.read_csv(REPO / "workspace/fold/v2/folds.csv")
    stem2fold = {f.rsplit(".", 1)[0]: fo for f, fo in zip(folds.filename, folds.fold)}
    df = gt.merge(feat, on="case_id", how="inner")
    df["fold"] = df.case_id.map(stem2fold)
    df["group"] = df.case_id.map(lambda s: re.match(r"center_\d+_case_\d+", s).group(0))
    fcols = [c for c in feat.columns if c != "case_id"]

    n_models = 0
    for fold in sorted(df.fold.unique()):
        tr = df[df.fold != fold]
        for st in STATIONS:
            y = tr[st].values
            path = out / f"lgb_fold{fold}_{st}.txt"
            if y.sum() in (0, len(y)):      # 定数クラスは学習できない → 定数を記録
                path.with_suffix(".const").write_text(str(float(y.mean())))
                continue
            clf = lgb.LGBMClassifier(**LGB_PARAMS)
            clf.fit(tr[fcols].values, y)
            clf.booster_.save_model(str(path))
            n_models += 1

    thr = json.loads(Path(args.thresholds).read_text()) if Path(args.thresholds).exists() else {}
    meta = {
        "features": fcols, "stations": STATIONS, "blend_w_lgb": args.w,
        "thresholds": thr, "n_models": n_models,
        "feature_spec": "features_from_masks.py / SUB=4 / fine31+coarse16 の area,has,cx,cy,ncomp,bw,bh",
        "note": "推論は fold0..4 の 5 本平均 -> CNN ヘッドと (1-w):w でブレンド -> thresholds で較正",
    }
    (out / "meta.json").write_text(json.dumps(meta, indent=2))
    print(f"{n_models} LGBM モデル + meta.json -> {out}")
    print(f"特徴 {len(fcols)} 次元 / 閾値 {len(thr)} クラス")


if __name__ == "__main__":
    main()
