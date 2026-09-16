"""build_graph.py の統計 (+ 3D アトラス) からルール行列を作る.

出力: workspace/anatomy_graph/out/rules_{task}[_fold{N}].npz
  W_adj  (C,C) float32  隣接ペナルティ [0,1]。1 = GT で「両方写っているのに接したことが無い」ペア
  E_excl (C,C) float32  排他ペナルティ {0,1}。1 = 同一画像に共起したことが無いペア
  K_max  (C,)  int64    1 画像あたりの連結成分数の上限 (GT p90) — 後処理用
  names  (C,)  str

ルールの根拠は全て学習側 GT の統計 (fold 指定時は val 画像を除外) なのでリークしない。
3D アトラス (atlas3d/graph3d.json) は「2D では未観測だが 3D では接触している」ペアの
ペナルティを弱める override にだけ使う (fine のみ。合成アトラスは不完全なので追加の禁止には使わない)。
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
OUT = REPO / "workspace/anatomy_graph/out"


# 隣接/排他ルールから外すクラス: 覆い被さる・偏在する組織は「何とでも接する」ので解剖学的な隣接制約を持たない
#   fine:   0 Background, 1 Instrument, 2 Other, 7 Fatty tissue esophagus, 10 Pleura, 20 Fatty tissue,
#           24 Pool of blood, 25 Resection area
#   coarse: 0 Background, 3 Pleura(+IPL), 12 Non-anatomical Other, 13 Fatty Tissue, 15 Anatomical Other
IGNORE_CLASSES = {"fine": (0, 1, 2, 7, 10, 20, 24, 25), "coarse": (0, 3, 12, 13, 15)}


def build_rules(task: str, fold: int = -1, p_lo: float = 0.02, p_hi: float = 0.2, min_cooc: int = 10,
                min_presence: int = 5, ignore_classes: tuple[int, ...] | None = None,
                atlas_override: bool = True, contact_mm: float = 5.0) -> dict:
    if ignore_classes is None:
        ignore_classes = IGNORE_CLASSES[task]
    d = OUT / (f"{task}_fold{fold}" if fold >= 0 else task)
    g = json.loads((d / "graph.json").read_text())
    names = g["class_names"]
    C = len(names)
    presence = np.load(d / "presence.npy")
    cooc = np.load(d / "cooc.npy")
    adj = np.load(d / "adj_img.npy")
    ncomp = np.load(d / "ncomp.npy")

    # --- 隣接ペナルティ: p_adj = P(接する | 両方出現) を Laplace 平滑化し, 対数線形で [0,1] に写す
    a0 = 0.5
    p_adj = (adj + a0) / (cooc + 2 * a0)
    W = np.log(p_hi / np.clip(p_adj, 1e-6, None)) / np.log(p_hi / p_lo)
    W = np.clip(W, 0.0, 1.0)
    W[cooc < min_cooc] = 0.0            # 証拠不足のペアは罰しない
    # --- 排他ペナルティ: 共起ゼロ (どちらも十分に出現するクラス同士)
    E = ((cooc == 0) & (presence[:, None] >= min_presence) & (presence[None, :] >= min_presence)).astype(np.float32)
    for c in ignore_classes:            # 背景 / 器具 / Other は何とでも接する
        W[c, :] = W[:, c] = 0.0
        E[c, :] = E[:, c] = 0.0
    np.fill_diagonal(W, 0.0)
    np.fill_diagonal(E, 0.0)
    W = np.maximum(W, W.T)
    E = np.maximum(E, E.T)

    overrides = []
    if atlas_override and task == "fine":
        p3 = OUT / "atlas3d/class_mindist_mm.npy"
        if p3.exists():
            d3 = np.load(p3)
            for a in range(C):
                for b in range(a + 1, C):
                    if W[a, b] > 0 and np.isfinite(d3[a, b]) and d3[a, b] <= contact_mm and cooc[a, b] < 3 * min_cooc:
                        overrides.append((names[a], names[b], float(W[a, b]), float(d3[a, b])))
                        W[a, b] = W[b, a] = W[a, b] * 0.5
    # --- 成分数上限 (GT の p90, 出現画像のみ)
    K = np.ones(C, np.int64) * 99
    for c in range(1, C):
        v = ncomp[:, c]
        v = v[v > 0]
        if len(v):
            K[c] = int(np.percentile(v, 90))
    return {"W_adj": W.astype(np.float32), "E_excl": E, "K_max": K, "names": np.array(names),
            "overrides": overrides, "p_adj": p_adj, "ignore_classes": np.array(ignore_classes)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", choices=["fine", "coarse"], default="fine")
    ap.add_argument("--fold", type=int, default=-1)
    ap.add_argument("--p-lo", type=float, default=0.02)
    ap.add_argument("--p-hi", type=float, default=0.2)
    ap.add_argument("--min-cooc", type=int, default=10)
    args = ap.parse_args()
    r = build_rules(args.task, args.fold, args.p_lo, args.p_hi, args.min_cooc)
    out = OUT / (f"rules_{args.task}" + (f"_fold{args.fold}" if args.fold >= 0 else "") + ".npz")
    np.savez(out, W_adj=r["W_adj"], E_excl=r["E_excl"], K_max=r["K_max"], names=r["names"],
             ignore_classes=r["ignore_classes"])
    W, E, names = r["W_adj"], r["E_excl"], r["names"]
    iu = np.triu_indices(len(names), 1)
    rows = [(names[a], names[b], round(float(W[a, b]), 2), round(float(r["p_adj"][a, b]), 3))
            for a, b in zip(*iu) if W[a, b] > 0]
    rows.sort(key=lambda t: -t[2])
    print(f"[{args.task} fold={args.fold}] adjacency-penalized pairs: {len(rows)} "
          f"(full=1.0: {sum(1 for t in rows if t[2] >= 1)}), exclusive pairs: {int(E[iu].sum())}, "
          f"atlas overrides: {len(r['overrides'])} -> {out.name}")
    for t in rows[:12]:
        print("   ", t)
    if r["overrides"]:
        print("   overrides:", r["overrides"])


if __name__ == "__main__":
    main()
