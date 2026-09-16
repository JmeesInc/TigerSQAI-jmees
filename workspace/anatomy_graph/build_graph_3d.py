"""expS01 の解剖アトラス (3D メッシュ, RAS mm) から 3D 隣接・相対位置グラフを作る.

2D GT 統計 (build_graph.py) は「術野に写ったときの隣接」しか分からず, 稀な構造対は
サンプル不足で「禁止」に見えてしまう。3D メッシュ間の最短距離は視点に依存しない
真の空間関係を与えるので, 2D 統計の裏付け / 補完に使う。

出力 (workspace/anatomy_graph/out/atlas3d/):
  class_mindist_mm.npy   (C,C) fine クラス間の表面最短距離 [mm] (自身は 0, 未収録は nan)
  class_centroid_ras.npy (C,3) クラス重心 (RAS mm; +x=右, +y=前, +z=頭側)
  graph3d.json           物体単位の距離表, リンパ節群ごとの近傍構造, 相対位置
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

REPO = Path(__file__).resolve().parents[2]


def surface_points(v: np.ndarray, f: np.ndarray, n_extra: int = 20000, seed: int = 0) -> np.ndarray:
    """頂点 + 面上のランダム点。粗いメッシュ (156 頂点など) でも距離が安定するよう面上点を足す。"""
    rng = np.random.default_rng(seed)
    if len(f) == 0:
        return v
    tri = v[f]  # (F,3,3)
    area = 0.5 * np.linalg.norm(np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0]), axis=1)
    if area.sum() <= 0:
        return v
    idx = rng.choice(len(f), size=n_extra, p=area / area.sum())
    r1, r2 = rng.random((2, n_extra))
    s = np.sqrt(r1)
    pts = (1 - s)[:, None] * tri[idx, 0] + (s * (1 - r2))[:, None] * tri[idx, 1] + (s * r2)[:, None] * tri[idx, 2]
    return np.concatenate([v, pts.astype(v.dtype)])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--work", default="workspace/expS01_atlas_synth/outputs/tier0_options2/work")
    ap.add_argument("--contact-mm", type=float, default=5.0, help="この距離以下を「接する」とみなす")
    ap.add_argument("--near-mm", type=float, default=20.0, help="この距離以下を「近傍」とみなす")
    ap.add_argument("--out", default="workspace/anatomy_graph/out/atlas3d")
    args = ap.parse_args()

    work = REPO / args.work
    meta = json.loads((work / "atlas.json").read_text())
    z = np.load(work / "atlas.npz")
    lm = pd.read_csv(REPO / "data/labelmap.csv")
    names = lm.set_index("fine_id").fine_name.to_dict()
    C = max(names) + 1

    objs = []
    for o in meta["objects"]:
        if o["role"] != "anatomy":
            continue
        v, f = z[o["key"] + "_v"].astype(np.float64), z[o["key"] + "_f"]
        c = v.mean(0)
        # 明らかな外れ (アトラスの配置ミス) は除外: 縦隔 z 範囲 [-200, 200] mm の外
        if not (-200 < c[2] < 200):
            print("skip (out of range):", o["name"], np.round(c, 1))
            continue
        pts = surface_points(v, f)
        objs.append({"name": o["name"], "fine_id": int(o["fine_id"]), "pts": pts, "centroid": c,
                     "tree": cKDTree(pts)})
    n = len(objs)
    D = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            d, _ = objs[j]["tree"].query(objs[i]["pts"], k=1)
            D[i, j] = D[j, i] = float(d.min())

    # クラス単位 (同 fine_id の物体は最小距離で束ねる)
    cls_d = np.full((C, C), np.nan)
    cls_c = np.full((C, 3), np.nan)
    ids = sorted({o["fine_id"] for o in objs})
    for a in ids:
        ia = [i for i, o in enumerate(objs) if o["fine_id"] == a]
        cls_c[a] = np.concatenate([objs[i]["pts"] for i in ia]).mean(0)
        for b in ids:
            ib = [i for i, o in enumerate(objs) if o["fine_id"] == b]
            cls_d[a, b] = 0.0 if a == b else min(D[i, j] for i in ia for j in ib)

    out = REPO / args.out
    out.mkdir(parents=True, exist_ok=True)
    np.save(out / "class_mindist_mm.npy", cls_d)
    np.save(out / "class_centroid_ras.npy", cls_c)

    obj_table = [{"name": o["name"], "fine_id": o["fine_id"], "class": names[o["fine_id"]],
                  "centroid_ras_mm": [round(float(x), 1) for x in o["centroid"]]} for o in objs]
    obj_dist = {o["name"]: {objs[j]["name"]: round(float(D[i, j]), 1) for j in range(n) if j != i}
                for i, o in enumerate(objs)}
    ln_neighbors = {}
    for i, o in enumerate(objs):
        if o["fine_id"] != 19:
            continue
        order = np.argsort(D[i])
        ln_neighbors[o["name"]] = [
            {"object": objs[j]["name"], "class": names[objs[j]["fine_id"]], "dist_mm": round(float(D[i, j]), 1),
             "delta_ras_mm": [round(float(x), 1) for x in (objs[j]["centroid"] - o["centroid"])]}
            for j in order if j != i and objs[j]["fine_id"] != 19 and D[i, j] <= 40
        ]
    contact = [{"a": names[a], "b": names[b], "dist_mm": round(float(cls_d[a, b]), 1)}
               for a in ids for b in ids if a < b and cls_d[a, b] <= args.contact_mm]
    near = [{"a": names[a], "b": names[b], "dist_mm": round(float(cls_d[a, b]), 1)}
            for a in ids for b in ids if a < b and args.contact_mm < cls_d[a, b] <= args.near_mm]
    far = [{"a": names[a], "b": names[b], "dist_mm": round(float(cls_d[a, b]), 1)}
           for a in ids for b in ids if a < b and cls_d[a, b] > args.near_mm]
    rel = {names[a]: {"x_right_mm": round(float(cls_c[a, 0]), 1), "y_ant_mm": round(float(cls_c[a, 1]), 1),
                      "z_sup_mm": round(float(cls_c[a, 2]), 1)} for a in ids}
    summary = {
        "source": str(work.relative_to(REPO)), "contact_mm": args.contact_mm, "near_mm": args.near_mm,
        "classes_covered": [names[a] for a in ids],
        "classes_missing": [names[c] for c in range(1, C) if c not in ids],
        "objects": obj_table, "class_centroid_ras": rel,
        "contact_pairs": contact, "near_pairs": near, "far_pairs": far,
        "lymph_node_groups_neighbors": ln_neighbors, "object_mindist_mm": obj_dist,
    }
    (out / "graph3d.json").write_text(json.dumps(summary, indent=1, ensure_ascii=False))
    print(f"objects={n} classes={len(ids)} contact={len(contact)} near={len(near)} far={len(far)} -> {out}")
    print("missing classes:", summary["classes_missing"])


if __name__ == "__main__":
    main()
