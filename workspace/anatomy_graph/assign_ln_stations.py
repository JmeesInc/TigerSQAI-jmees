"""Task3 用アノテーション: リンパ節 (fine id 19) の各連結成分を, 解剖グラフのルールで station に割り当てる.

入力:  labels_fine_1024 (GT), task3_gt_wide.csv (画像ごとの visible station 集合 V)
出力:  workspace/anatomy_graph/out/task3/ln_station_assignments.csv  (成分単位)
       .../station_maps/{filename}.png  (uint8: 0=なし, 1..14=station index+1  ※ LN 画素のみ)
       .../review_contact_sheet.png     (目視レビュー用)

割当てルール (survey/competition/tiger_ln_station_map.md の TIGER 定義から作った期待近傍表 STATION_CONTEXT):
  score(comp, s) = Σ_c STATION_CONTEXT[s][c] * near_c(comp)
    near_c = 成分の周囲 (接触リング ×2 + 近傍リング ×1) に占めるクラス c の割合 (解剖クラスのみで正規化)
  候補は画像の visible 集合 V に限定し argmax。V が 1 要素なら無条件でその station。
  左右 (10/11/12 の L/R, 6/7, 13) は画像の向きに依存しないよう「奇静脈・右迷走神経・SVC・右IPL = 右」
  「大動脈・左反回神経・左IPL = 左」という同一画像内の解剖学的手がかりで決める。
制約: ファイル名の station は割当てには使わない (QA 指標としてのみ参照)。
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
log = logging.getLogger("assign_ln")
STATIONS = ["6L", "6R", "7L", "7R", "8", "9", "10L", "10R", "11L", "11R", "12L", "12R", "13L", "13R"]

# fine_id
TRACHEA, RMB, LMB, ESO, FAT_ESO, R_IPL, L_IPL, PLEURA, PERI, IPV = 3, 4, 5, 6, 7, 8, 9, 10, 11, 12
R_SCA, R_VAGUS, AORTA, AZYGOS, SVC, LUNG, LN, FAT, L_SCA, R_BA, PA = 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23
BLOOD, RESECT, CONDUIT, R_RLN, L_RLN, OMENTUM, TD = 24, 25, 26, 27, 28, 29, 30
NON_ANATOMICAL = {0, 1, 2, FAT_ESO, PLEURA, FAT, BLOOD, RESECT, LN}   # 近傍分布の分母から外す (LN 自身も)

# station -> {fine_id: weight}. TIGER 定義に明記 = 2, 解剖学的推測で隣接 = 1, 左右の決め手 = 3
STATION_CONTEXT = {
    "6R": {R_VAGUS: 3, R_RLN: 3, TRACHEA: 2, R_SCA: 2, ESO: 1, SVC: 1},
    "6L": {L_RLN: 3, TRACHEA: 2, AORTA: 2, L_SCA: 2, ESO: 1},
    "7R": {TRACHEA: 2, RMB: 2, SVC: 3, AZYGOS: 2, R_VAGUS: 2},
    "7L": {TRACHEA: 2, LMB: 2, AORTA: 3, AZYGOS: 1, R_BA: 2, L_RLN: 2, PA: 1},
    "8":  {AORTA: 3, PA: 3, LMB: 2, L_RLN: 1},
    "9":  {TRACHEA: 2, RMB: 3, LMB: 3, ESO: 1, PERI: 1, TD: 1},
    "10R": {ESO: 2, TRACHEA: 1, R_VAGUS: 3, R_RLN: 2, AZYGOS: 2, RMB: 1},
    "10L": {ESO: 2, TRACHEA: 1, L_RLN: 3, AORTA: 2, LMB: 1},
    "11R": {ESO: 2, AZYGOS: 3, PERI: 1, IPV: 1, TD: 1, R_VAGUS: 1},
    "11L": {ESO: 2, AORTA: 3, PERI: 1, IPV: 1, TD: 1},
    "12R": {ESO: 2, IPV: 1, PERI: 1, AZYGOS: 2, CONDUIT: 1, R_IPL: 1},
    "12L": {ESO: 2, IPV: 1, PERI: 1, AORTA: 2, CONDUIT: 1, L_IPL: 1},
    "13R": {R_IPL: 4, LUNG: 1, IPV: 1, ESO: 1},
    "13L": {L_IPL: 4, LUNG: 1, IPV: 1, ESO: 1},
}
# 頭尾レベル (0=上縦隔 ... 3=下縦隔) — 位置手がかり無しの場合の同点解消に使う
LEVEL = {"6L": 0, "6R": 0, "7L": 1, "7R": 1, "8": 1, "9": 1, "10L": 1, "10R": 1,
         "11L": 2, "11R": 2, "12L": 3, "12R": 3, "13L": 3, "13R": 3}


def ring_hist(lab: np.ndarray, comp: np.ndarray, r: int, n: int = 31) -> np.ndarray:
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1))
    dil = cv2.dilate(comp.astype(np.uint8), k) > 0
    ring = dil & ~comp
    return np.bincount(lab[ring], minlength=n).astype(np.float64)


def context_vector(lab: np.ndarray, comp: np.ndarray, r_touch: int = 2, r_near: int = 25) -> np.ndarray:
    h = 2 * ring_hist(lab, comp, r_touch) + ring_hist(lab, comp, r_near)
    for c in NON_ANATOMICAL:
        h[c] = 0
    s = h.sum()
    return h / s if s > 0 else h


def score_stations(ctx: np.ndarray, present: np.ndarray) -> dict[str, float]:
    out = {}
    for s, table in STATION_CONTEXT.items():
        sc = sum(w * ctx[c] for c, w in table.items())
        # 画像内に存在するだけ (接していない) 手がかりにも弱い加点
        sc += 0.1 * sum(w * (present[c] > 0) for c, w in table.items()) / sum(table.values())
        out[s] = float(sc)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels-dir", default="workspace/data_proc/labels_fine_1024")
    ap.add_argument("--images-dir", default="workspace/data_proc/images_1024")
    ap.add_argument("--task3-csv", default="workspace/data_proc/task3_gt_wide.csv")
    ap.add_argument("--min-area", type=int, default=64)
    ap.add_argument("--out", default="workspace/anatomy_graph/out/task3")
    ap.add_argument("--n-review", type=int, default=24)
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    out = REPO / args.out
    (out / "station_maps").mkdir(parents=True, exist_ok=True)

    t3 = pd.read_csv(REPO / args.task3_csv)
    rows, review = [], []
    n_no_ln = 0
    for r in t3.itertuples():
        fn = f"{r.case_id}.png"
        lab = cv2.imread(str(REPO / args.labels_dir / fn), cv2.IMREAD_GRAYSCALE)
        if lab is None:
            log.warning("missing label: %s", fn); continue
        V = [s for s in STATIONS if int(getattr(r, f"_{STATIONS.index(s)+2}")) == 1] if False else \
            [s for s in STATIONS if int(t3.loc[r.Index, s]) == 1]
        present = np.bincount(lab.ravel(), minlength=31)
        cc, k = ndi.label(lab == LN, structure=np.ones((3, 3), bool))
        smap = np.zeros_like(lab)
        if k == 0:
            n_no_ln += 1
        case_id = re.match(r"center_\d+_case_\d+", r.case_id).group(0)
        fname_station = r.case_id[len(case_id) + 1:].split("_frame")[0]
        for i in range(1, k + 1):
            comp = cc == i
            area = int(comp.sum())
            if area < args.min_area:
                continue
            ys, xs = np.nonzero(comp)
            ctx = context_vector(lab, comp)
            sc = score_stations(ctx, present)
            cand = {s: sc[s] for s in V} if V else sc
            order = sorted(cand, key=lambda s: -cand[s])
            best = order[0] if order else ""
            second = cand[order[1]] if len(order) > 1 else 0.0
            margin = cand[best] - second if best else 0.0
            if len(V) == 1:
                best, margin = V[0], 9.0
            top_nb = np.argsort(ctx)[::-1][:3]
            smap[comp] = STATIONS.index(best) + 1 if best else 0
            rows.append(dict(filename=fn, case_id=case_id, comp_id=i, area=area,
                             cx=round(float(xs.mean()) / lab.shape[1], 3), cy=round(float(ys.mean()) / lab.shape[0], 3),
                             station=best, score=round(cand.get(best, 0.0), 3), margin=round(margin, 3),
                             n_visible=len(V), visible=" ".join(V), fname_station=fname_station,
                             neighbors=" ".join(f"{int(c)}:{ctx[c]:.2f}" for c in top_nb if ctx[c] > 0)))
        cv2.imwrite(str(out / "station_maps" / fn), smap)
    df = pd.DataFrame(rows)
    df.to_csv(out / "ln_station_assignments.csv", index=False)
    log.info("frames=%d (no LN: %d) components=%d", len(t3), n_no_ln, len(df))

    # --- 要約 / QA
    amb = df[df.n_visible > 1]
    log.info("components in multi-station frames: %d / %d; low-margin (<0.05): %d",
             len(amb), len(df), int((amb.margin < 0.05).sum()))
    log.info("assigned station counts:\n%s", df.station.value_counts().to_string())
    # QA: フレーム名の station は「そのフレームが選ばれた station」。最大成分がそれに割り当たる割合 (参考値)
    big = df.sort_values("area", ascending=False).groupby("filename").head(1)
    big_amb = big[big.n_visible > 1]
    agree = (big_amb.station == big_amb.fname_station).mean() if len(big_amb) else float("nan")
    chance = (1.0 / big_amb.n_visible).mean() if len(big_amb) else float("nan")
    log.info("QA (multi-station frames only, largest LN comp == filename station): %.3f  (chance %.3f, n=%d)",
             agree, chance, len(big_amb))
    per_station = df[df.n_visible > 1].groupby("station").margin.agg(["count", "mean"]).round(3)
    log.info("margin by station (multi-station frames):\n%s", per_station.to_string())
    json.dump({"n_frames": len(t3), "n_components": len(df), "qa_agree_largest": agree, "qa_chance": chance,
               "station_counts": df.station.value_counts().to_dict()}, open(out / "summary.json", "w"), indent=2)

    # --- 目視レビュー用コンタクトシート (多 station フレームからマージン低/高を混ぜて抽出)
    sel = pd.concat([amb.sort_values("margin").head(args.n_review // 2),
                     amb.sort_values("margin", ascending=False).head(args.n_review // 2)])
    tiles = []
    for r in sel.itertuples():
        img = cv2.imread(str(REPO / args.images_dir / r.filename))
        lab = cv2.imread(str(REPO / args.labels_dir / r.filename), cv2.IMREAD_GRAYSCALE)
        cc, _ = ndi.label(lab == LN, structure=np.ones((3, 3), bool))
        comp = (cc == r.comp_id).astype(np.uint8)
        cnts, _ = cv2.findContours(comp, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(img, cnts, -1, (0, 255, 255), 3)
        ys, xs = np.nonzero(comp)
        x0, x1 = max(0, xs.min() - 160), min(img.shape[1], xs.max() + 160)
        y0, y1 = max(0, ys.min() - 120), min(img.shape[0], ys.max() + 120)
        crop = cv2.resize(img[y0:y1, x0:x1], (320, 240))
        txt = f"{r.station} m={r.margin:.2f} V={r.visible}"
        cv2.putText(crop, txt, (4, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 3)
        cv2.putText(crop, txt, (4, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
        cv2.putText(crop, r.filename[:-4], (4, 232), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 0), 1)
        tiles.append(crop)
    if tiles:
        ncol = 4
        while len(tiles) % ncol:
            tiles.append(np.zeros_like(tiles[0]))
        sheet = np.vstack([np.hstack(tiles[i:i + ncol]) for i in range(0, len(tiles), ncol)])
        cv2.imwrite(str(out / "review_contact_sheet.png"), sheet)
        log.info("contact sheet -> %s", out / "review_contact_sheet.png")


if __name__ == "__main__":
    main()
