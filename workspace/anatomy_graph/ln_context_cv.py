"""LN 成分の近傍コンテキストから station を当てられるか (データ駆動の検証).
最大成分の擬似正解 = ファイル名 station (multi-station フレームのみ)。case で GroupKFold。
予測は visible 集合 V 内の argmax に制限 (手書きルール表と同じ土俵)。"""
import re, sys
from pathlib import Path
import cv2, numpy as np, pandas as pd
from scipy import ndimage as ndi
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler
REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "workspace/anatomy_graph"))
from assign_ln_stations import STATIONS, LN, context_vector  # noqa: E402

df = pd.read_csv(REPO / "workspace/anatomy_graph/out/task3/ln_station_assignments.csv")
big = df.sort_values("area", ascending=False).groupby("filename").head(1)
big = big[big.n_visible > 1].reset_index(drop=True)
X, y, groups, Vs = [], [], [], []
for r in big.itertuples():
    lab = cv2.imread(str(REPO / "workspace/data_proc/labels_fine_1024" / r.filename), cv2.IMREAD_GRAYSCALE)
    cc, _ = ndi.label(lab == LN, structure=np.ones((3, 3), bool))
    comp = cc == r.comp_id
    ctx = context_vector(lab, comp)
    present = (np.bincount(lab.ravel(), minlength=31) > 0).astype(float)
    area_frac = np.bincount(lab.ravel(), minlength=31) / lab.size
    X.append(np.concatenate([ctx, present, area_frac, [r.cx, r.cy, np.log(r.area)]]))
    y.append(STATIONS.index(r.fname_station)); groups.append(r.case_id); Vs.append(r.visible.split())
X, y = np.array(X), np.array(y)
acc_rule = (big.station == big.fname_station).mean()
chance = (1 / big.n_visible).mean()
accs = []
for tr, te in GroupKFold(5).split(X, y, groups):
    sc = StandardScaler().fit(X[tr])
    clf = LogisticRegression(max_iter=2000, C=0.3).fit(sc.transform(X[tr]), y[tr])
    P = clf.predict_proba(sc.transform(X[te]))
    for i, idx in enumerate(te):
        cand = [STATIONS.index(s) for s in Vs[idx]]
        cls_idx = [list(clf.classes_).index(c) if c in clf.classes_ else None for c in cand]
        scores = [P[i, j] if j is not None else -1 for j in cls_idx]
        accs.append(cand[int(np.argmax(scores))] == y[idx])
print(f"n={len(y)}  chance={chance:.3f}  hand-rules={acc_rule:.3f}  logreg(GroupKFold, argmax in V)={np.mean(accs):.3f}")
# 特徴群別
for name, sl in [("ctx only", slice(0, 31)), ("frame presence+area only", slice(31, 93)), ("pos+size only", slice(93, 96))]:
    accs = []
    for tr, te in GroupKFold(5).split(X, y, groups):
        sc = StandardScaler().fit(X[tr][:, sl]); clf = LogisticRegression(max_iter=2000, C=0.3).fit(sc.transform(X[tr][:, sl]), y[tr])
        P = clf.predict_proba(sc.transform(X[te][:, sl]))
        for i, idx in enumerate(te):
            cand = [STATIONS.index(s) for s in Vs[idx]]
            scores = [P[i, list(clf.classes_).index(c)] if c in clf.classes_ else -1 for c in cand]
            accs.append(cand[int(np.argmax(scores))] == y[idx])
    print(f"  {name:28s} acc={np.mean(accs):.3f}")
