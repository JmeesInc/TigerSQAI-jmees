"""GT マスクから解剖グラフ統計を作る (Task1/2 ルール loss と Task3 station 割当ての共通基盤).

出力 (workspace/anatomy_graph/out/{fine|coarse}/):
  presence.npy        (C,)    クラスが出現した画像数
  cooc.npy            (C,C)   両クラスが同一画像に出現した画像数
  adj_img.npy         (C,C)   両クラスが「接している」画像数 (4近傍で境界画素対 >= min_pairs)
  adj_pairs.npy       (C,C)   境界画素対の総数 (4近傍, 対称)
  nb_dist.npy         (C,C)   行 a: クラス a の境界画素が接している相手クラスの分布 (背景含む, 行和=1)
  ncomp_hist.json     クラス別の連結成分数ヒストグラム (画像ごと, min_area 以上の成分)
  enclosure.npy       (C,C)   [a,b] = クラス a の成分の外周 >= enc_thr がクラス b だった成分数
  graph.json          人が読むまとめ (許容隣接ペア / 禁止ペア / 排他ペア / 期待成分数)

「禁止」= 学習全体 (指定 fold の train 側) で 1 度も観測されなかったペア。
fold を指定すると val 画像を除外して統計を取る (CV での情報リーク防止)。
"""
from __future__ import annotations

import argparse
import json
import logging
import re
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from scipy import ndimage as ndi

REPO = Path(__file__).resolve().parents[2]
log = logging.getLogger("build_graph")


def load_labelmap(task: str):
    lm = pd.read_csv(REPO / "data/labelmap.csv")
    if task == "fine":
        names = lm.set_index("fine_id").fine_name.to_dict()
        weights = lm.set_index("fine_id").weight.to_dict()
    else:
        g = lm.drop_duplicates("merged_id").set_index("merged_id")
        names = g.merged_name.to_dict()
        # merged の weight は公式 classes_merged.py の定義に従う (fine の max ではない)
        try:
            import sys
            sys.path.insert(0, str(REPO / "reference/tigersqai_challenge"))
            from metrics.classes_merged import CLASSES_MERGED
            weights = {c.id: c.weight for c in CLASSES_MERGED.values()} if isinstance(CLASSES_MERGED, dict) \
                else {c.id: c.weight for c in CLASSES_MERGED}
        except Exception:  # noqa: BLE001
            weights = lm.groupby("merged_id").weight.max().to_dict()
    n = max(names) + 1
    return n, [names.get(i, f"id{i}") for i in range(n)], [int(weights.get(i, 1)) for i in range(n)]


def image_stats(lab: np.ndarray, n: int, min_pairs: int, min_area: int, enc_thr: float):
    """1 画像分: 隣接画素対 (C,C), 成分数 (C,), 囲み (C,C), 存在 (C,)."""
    pairs = np.zeros((n, n), np.int64)
    for a, b in ((lab[:, :-1], lab[:, 1:]), (lab[:-1, :], lab[1:, :])):
        m = a != b
        idx = a[m].astype(np.int64) * n + b[m].astype(np.int64)
        pairs += np.bincount(idx, minlength=n * n).reshape(n, n)
    pairs = pairs + pairs.T
    present = np.bincount(lab.ravel(), minlength=n) > 0
    ncomp = np.zeros(n, np.int64)
    enc = np.zeros((n, n), np.int64)
    struct = np.ones((3, 3), bool)
    for c in np.flatnonzero(present):
        if c == 0:
            continue
        cc, k = ndi.label(lab == c, structure=struct)
        if k == 0:
            continue
        areas = np.bincount(cc.ravel(), minlength=k + 1)[1:]
        keep = np.flatnonzero(areas >= min_area) + 1
        ncomp[c] = len(keep)
        for ci in keep:
            comp = cc == ci
            ring = ndi.binary_dilation(comp, struct) & ~comp
            nb = lab[ring]
            if nb.size == 0:
                continue
            h = np.bincount(nb, minlength=n).astype(np.float64)
            h[c] = 0  # 同クラス (min_area 未満の破片) は除外
            if h.sum() == 0:
                continue
            frac = h / h.sum()
            b = int(frac.argmax())
            if frac[b] >= enc_thr:
                enc[c, b] += 1
    return pairs, present, ncomp, enc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", choices=["fine", "coarse"], default="fine")
    ap.add_argument("--labels-dir", default=None, help="既定: workspace/data_proc/labels_{task}_1024")
    ap.add_argument("--folds-csv", default="workspace/fold/v1/folds.csv", help="空文字なら labels-dir の全 png")
    ap.add_argument("--fold", type=int, default=-1, help=">=0 なら当該 fold の val 画像を除外")
    ap.add_argument("--exclude", default="center_7_case_2_13R.png", help="カンマ区切りの除外ファイル")
    ap.add_argument("--min-pairs", type=int, default=20, help="接触と見なす境界画素対の最小数")
    ap.add_argument("--min-area", type=int, default=64, help="成分としてカウントする最小面積 (px @576x1024)")
    ap.add_argument("--enc-thr", type=float, default=0.9)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")

    n, names, weights = load_labelmap(args.task)
    labels_dir = REPO / (args.labels_dir or f"workspace/data_proc/labels_{args.task}_1024")
    if args.folds_csv:
        folds = pd.read_csv(REPO / args.folds_csv)
        excl = set(x for x in args.exclude.split(",") if x)
        df = folds[~folds.filename.isin(excl)]
        if args.fold >= 0:
            df = df[df.fold != args.fold]
        files = sorted(df.filename)
    else:  # 合成データなど folds.csv が無いディレクトリ: 全 png
        files = sorted(p.name for p in labels_dir.glob("*.png"))
    log.info("task=%s n_classes=%d images=%d (fold=%d excluded)", args.task, n, len(files), args.fold)

    presence = np.zeros(n, np.int64)
    cooc = np.zeros((n, n), np.int64)
    adj_img = np.zeros((n, n), np.int64)
    adj_pairs = np.zeros((n, n), np.int64)
    enclosure = np.zeros((n, n), np.int64)
    ncomp_rows = []
    for i, f in enumerate(files):
        lab = cv2.imread(str(labels_dir / f), cv2.IMREAD_GRAYSCALE)
        assert lab is not None, f
        pairs, present, ncomp, enc = image_stats(lab, n, args.min_pairs, args.min_area, args.enc_thr)
        presence += present
        cooc += np.outer(present, present)
        adj_img += pairs >= args.min_pairs
        adj_pairs += pairs
        enclosure += enc
        ncomp_rows.append(ncomp)
        if (i + 1) % 100 == 0:
            log.info("%d/%d", i + 1, len(files))
    ncomp_arr = np.stack(ncomp_rows)  # (N, C)

    # 境界画素の相手分布 (行 a: a の境界画素が接する相手クラス, 背景含む)
    nb_dist = adj_pairs.astype(np.float64)
    nb_dist = nb_dist / np.clip(nb_dist.sum(1, keepdims=True), 1, None)

    out = REPO / (args.out or f"workspace/anatomy_graph/out/{args.task}" + (f"_fold{args.fold}" if args.fold >= 0 else ""))
    out.mkdir(parents=True, exist_ok=True)
    np.save(out / "presence.npy", presence)
    np.save(out / "cooc.npy", cooc)
    np.save(out / "adj_img.npy", adj_img)
    np.save(out / "adj_pairs.npy", adj_pairs)
    np.save(out / "nb_dist.npy", nb_dist)
    np.save(out / "enclosure.npy", enclosure)
    np.save(out / "ncomp.npy", ncomp_arr)

    # 人が読むまとめ
    ids = [c for c in range(1, n)]
    forbidden_adj, rare_adj, exclusive = [], [], []
    for a in ids:
        for b in ids:
            if b <= a:
                continue
            both = int(cooc[a, b])
            if both == 0 and presence[a] >= 5 and presence[b] >= 5:
                exclusive.append({"a": names[a], "b": names[b], "n_a": int(presence[a]), "n_b": int(presence[b])})
            if both > 0 and adj_img[a, b] == 0:
                forbidden_adj.append({"a": names[a], "b": names[b], "cooc_imgs": both})
            elif both > 0 and adj_img[a, b] / both < 0.05 and adj_img[a, b] <= 2:
                rare_adj.append({"a": names[a], "b": names[b], "cooc_imgs": both, "adj_imgs": int(adj_img[a, b])})
    ncomp_hist = {}
    expected_ncomp = {}
    for c in ids:
        v = ncomp_arr[:, c]
        v = v[v > 0]
        if len(v) == 0:
            continue
        h = np.bincount(v, minlength=6)
        ncomp_hist[names[c]] = {str(k): int(h[k]) for k in range(1, len(h)) if h[k] > 0}
        expected_ncomp[names[c]] = {"median": float(np.median(v)), "p90": float(np.percentile(v, 90)),
                                    "p_single": float((v == 1).mean()), "n_imgs": int(len(v))}
    enc_list = []
    for a in ids:
        for b in range(n):
            if enclosure[a, b] > 0 and a != b:
                enc_list.append({"inner": names[a], "outer": names[b], "n_components": int(enclosure[a, b])})
    enc_list.sort(key=lambda d: -d["n_components"])
    summary = {
        "task": args.task, "n_images": len(files), "fold_excluded": args.fold,
        "params": {"min_pairs": args.min_pairs, "min_area": args.min_area, "enc_thr": args.enc_thr},
        "class_names": names, "class_weights": weights,
        "presence": {names[c]: int(presence[c]) for c in ids},
        "adjacency_prob": {  # P(接する | 両方出現)
            names[a]: {names[b]: round(float(adj_img[a, b] / max(cooc[a, b], 1)), 3)
                       for b in ids if b != a and cooc[a, b] > 0}
            for a in ids
        },
        "forbidden_adjacent_pairs": forbidden_adj,
        "rare_adjacent_pairs": rare_adj,
        "mutually_exclusive_pairs": exclusive,
        "ncomp_hist": ncomp_hist,
        "expected_ncomp": expected_ncomp,
        "enclosure": enc_list,
    }
    (out / "graph.json").write_text(json.dumps(summary, indent=1, ensure_ascii=False))
    log.info("forbidden adjacent pairs: %d / rare: %d / exclusive: %d / enclosure rules: %d -> %s",
             len(forbidden_adj), len(rare_adj), len(exclusive), len(enc_list), out)


if __name__ == "__main__":
    main()
