"""expA23: **5-fold CV の各 fold でアンサンブルを組み、公式 Dice + HD の 5fold 平均**で構成を決める.

これまでの候補選定は「全 OOF をまとめて 1 本のスコア」で見ていたが、提出構成は
fold ごとの平均で決める方針に合わせる。公式のランキング規則も Dice と HD を
別々に順位化して平均するので、ここでも両方を出す。

  * メンバー確率は save_probs.py が保存した OOF (288x1024/2 = 288x512, fp16)
  * 平均 → 原寸へ bilinear → argmax → 公式 weighted Dice / 正規化 HD
  * fold ごとに採点し、5 fold の平均で比較（case 集約は公式どおり画像→case→全体）

Usage:
    python3 ens_select_cv.py --greedy --max-size 10 --jobs 16
    python3 ens_select_cv.py --sets "q_cholec_dlv3,q_endovis18_dlv3" "k_dicedet_anat3d,l_dicedet"
"""

from __future__ import annotations

import argparse
import itertools
import json
import multiprocessing as mp
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(REPO / "reference" / "tigersqai_challenge"))

FOLDS = pd.read_csv(REPO / "workspace/fold/v2/folds.csv")


def available_arms() -> list[str]:
    out = []
    for d in sorted((HERE / "results").glob("expA23_*/probs/fine")):
        if len(list(d.glob("*.npy"))) >= 526:
            out.append(d.parts[-3].replace("expA23_", ""))
    return out


def _score_one(args):
    """1 画像: メンバー平均 → 原寸 → argmax → 公式 (dice, hd) を fine/coarse で返す。"""
    stem, arms = args
    import torch
    import torch.nn.functional as F
    from metrics import classes as C_fine
    from metrics import classes_merged as C_coarse
    from metrics.metrics import weighted_image_scores
    bgr = cv2.imread(str(REPO / "data/images" / f"{stem}.png"))
    oh, ow = bgr.shape[:2]
    out = {}
    for task, mod, gt_dir in (("fine", C_fine, "masks_fine"), ("coarse", C_coarse, "masks_coarse")):
        acc = None
        for a in arms:
            p = np.load(HERE / f"results/expA23_{a}/probs/{task}/{stem}.npy").astype(np.float32)
            acc = p if acc is None else acc + p
        t = torch.from_numpy(acc / len(arms))[None]
        up = F.interpolate(t, size=(oh, ow), mode="bilinear", align_corners=False)
        pred = up.argmax(1)[0].numpy().astype(np.uint8)
        g_rgb = cv2.cvtColor(cv2.imread(str(REPO / "data" / gt_dir / f"{stem}.png")), cv2.COLOR_BGR2RGB)
        g = mod.rgb_mask_to_label_mask(g_rgb)
        classes = mod.CLASSES if task == "fine" else mod.CLASSES_MERGED
        sc = weighted_image_scores(pred, g, classes, sum(c.weight for c in classes))
        out[task] = (float(sc["dice"]), float(sc["hd"]))
    return stem, out


def evaluate(arms: list[str], pool, stems_by_fold) -> dict:
    """fold ごとに採点し 5fold 平均を返す。"""
    per_fold = []
    for k in range(5):
        stems = stems_by_fold[k]
        res = pool.map(_score_one, [(s, arms) for s in stems], chunksize=2)
        per_case = {"fine": {}, "coarse": {}}
        for stem, o in res:
            case = stem.rsplit("_", 1)[0]
            for t, v in o.items():
                per_case[t].setdefault(case, []).append(v)
        row = {}
        for t in ("fine", "coarse"):
            cm = {c: (np.mean([x[0] for x in v]), np.mean([x[1] for x in v])) for c, v in per_case[t].items()}
            row[f"{t}_dice"] = float(np.mean([v[0] for v in cm.values()]))
            row[f"{t}_hd"] = float(np.mean([v[1] for v in cm.values()]))
        per_fold.append(row)
    mean = {k: float(np.mean([r[k] for r in per_fold])) for k in per_fold[0]}
    mean["per_fold"] = per_fold
    return mean


def fmt(m: dict) -> str:
    return (f"T1 Dice {m['coarse_dice']:.4f} HD {m['coarse_hd']:.4f} | "
            f"T2 Dice {m['fine_dice']:.4f} HD {m['fine_hd']:.4f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--greedy", action="store_true")
    ap.add_argument("--max-size", type=int, default=10)
    ap.add_argument("--top", type=int, default=None, help="単体上位 N 本だけを貪欲の候補にする")
    ap.add_argument("--jobs", type=int, default=12)
    ap.add_argument("--sets", nargs="*", default=[])
    ap.add_argument("--limit-per-fold", type=int, default=None, help="動作確認用")
    ap.add_argument("--out", default=str(HERE / "ens_select_cv.json"))
    args = ap.parse_args()

    arms = available_arms()
    print(f"候補 {len(arms)} レシピ: {', '.join(arms)}", flush=True)
    stems_by_fold = {}
    for k in range(5):
        s = [f.rsplit(".", 1)[0] for f in FOLDS[FOLDS.fold == k].filename]
        stems_by_fold[k] = s[: args.limit_per_fold] if args.limit_per_fold else s

    results = {}
    with mp.Pool(args.jobs) as pool:
        if args.sets:
            for spec in args.sets:
                a = [x.strip() for x in spec.split(",")]
                m = evaluate(a, pool, stems_by_fold)
                results[spec] = m
                print(f"{spec}\n  {fmt(m)}", flush=True)
        if args.greedy:
            # 単体の 5fold 平均でランキング
            singles = {}
            for a in arms:
                m = evaluate([a], pool, stems_by_fold)
                singles[a] = m
                print(f"[単体] {a:24s} {fmt(m)}", flush=True)
            results["singles"] = singles
            Path(args.out).write_text(json.dumps(results, indent=1, ensure_ascii=False))
            order = sorted(arms, key=lambda a: -(singles[a]["fine_dice"] + singles[a]["coarse_dice"]))
            if args.top:
                order = order[: args.top]
            chosen = [order[0]]
            best = evaluate(chosen, pool, stems_by_fold)
            print(f"\n[貪欲] 開始 {chosen} → {fmt(best)}", flush=True)
            hist = [{"members": list(chosen), **{k: v for k, v in best.items() if k != "per_fold"}}]
            while len(chosen) < args.max_size:
                cand, cand_m = None, None
                for a in order:
                    if a in chosen:
                        continue
                    m = evaluate(chosen + [a], pool, stems_by_fold)
                    gain = (m["fine_dice"] + m["coarse_dice"]) - (best["fine_dice"] + best["coarse_dice"])
                    if cand_m is None or gain > cand_m[1]:
                        cand, cand_m = a, (m, gain)
                if cand is None or cand_m[1] <= 0:
                    print(f"[貪欲] 改善なしで停止 ({len(chosen)} メンバー)", flush=True)
                    break
                chosen.append(cand); best = cand_m[0]
                print(f"[貪欲] +{cand:24s} → {fmt(best)}  (Δdice {cand_m[1]:+.4f})", flush=True)
                hist.append({"members": list(chosen), **{k: v for k, v in best.items() if k != "per_fold"}})
                results["greedy"] = hist
                Path(args.out).write_text(json.dumps(results, indent=1, ensure_ascii=False))
            results["greedy"] = hist
    Path(args.out).write_text(json.dumps(results, indent=1, ensure_ascii=False))
    print(f"\n-> {args.out}")


if __name__ == "__main__":
    main()
