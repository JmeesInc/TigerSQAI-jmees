"""expA23: 公式 Dice + 正規化 HD の **fold 平均**で貪欲前向き選択を回す（高速版）.

ens_select_cv.py の実測内訳（1080p 1 枚・fine のみ）:
    公式HD 867ms / 補間+argmax 299ms / npy読み 187ms / 公式Dice 112ms
    入力PNG読み 67ms / GT PNG読み 50ms / GT RGB->ID 43ms
HD は公式実装をそのまま使う（ここを近似すると選択の根拠が変わる）。
代わりに **毎回やり直していた前処理を全部キャッシュ**する:

  * GT のラベルマップ (RGB PNG デコード + LUT) -> uint8 npy に 1 回だけ変換
  * 原寸サイズ -> JSON に 1 回だけ記録（入力 PNG を開く必要が無くなる）

これで 1 枚あたり 160ms（約 10%）を削り、さらにマシンが空いている分だけ
ワーカを増やせる。選択の評価値は公式関数そのままなので変わらない。

Usage:
    python3 ens_greedy_fast.py --prepare                 # キャッシュ作成（初回のみ）
    python3 ens_greedy_fast.py --greedy --max-size 10 --jobs 64
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(REPO / "reference" / "tigersqai_challenge"))

CACHE = HERE / "_cvcache"
FOLDS = pd.read_csv(REPO / "workspace/fold/v2/folds.csv")


def available_arms() -> list[str]:
    out = []
    for d in sorted((HERE / "results").glob("expA23_*/probs/fine")):
        if len(list(d.glob("*.npy"))) >= 526:
            out.append(d.parts[-3].replace("expA23_", ""))
    return out


# ---------------------------------------------------------------- キャッシュ作成
def _prep_one(stem: str):
    from metrics import classes as C_fine
    from metrics import classes_merged as C_coarse
    out = {}
    for task, mod, gt_dir in (("fine", C_fine, "masks_fine"), ("coarse", C_coarse, "masks_coarse")):
        p = CACHE / f"gt_{task}" / f"{stem}.npy"
        if not p.exists():
            rgb = cv2.cvtColor(cv2.imread(str(REPO / "data" / gt_dir / f"{stem}.png")), cv2.COLOR_BGR2RGB)
            np.save(p, mod.rgb_mask_to_label_mask(rgb).astype(np.uint8))
        out[task] = p
    im = cv2.imread(str(REPO / "data/images" / f"{stem}.png"), cv2.IMREAD_REDUCED_COLOR_8)
    return stem, (im.shape[0] * 8, im.shape[1] * 8)


def prepare(stems, jobs):
    for t in ("fine", "coarse"):
        (CACHE / f"gt_{t}").mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    with mp.Pool(jobs) as pool:
        res = pool.map(_prep_one, stems, chunksize=4)
    sizes = {}
    for stem, hw in res:
        sizes[stem] = list(hw)
    (CACHE / "sizes.json").write_text(json.dumps(sizes))
    print(f"キャッシュ作成 {len(stems)} 枚 / {time.time() - t0:.0f} 秒 -> {CACHE}", flush=True)


# ---------------------------------------------------------------- 採点
_SIZES = None


def _score_one(args):
    stem, arms = args
    global _SIZES
    import torch
    torch.set_num_threads(1)      # 48 ワーカ x 多スレッドで 128 コアを溢れさせない
    import torch.nn.functional as F
    from metrics import classes as C_fine
    from metrics import classes_merged as C_coarse
    from metrics.metrics import weighted_image_scores
    if _SIZES is None:
        _SIZES = json.loads((CACHE / "sizes.json").read_text())
    oh, ow = _SIZES[stem]
    out = {}
    for task, mod in (("fine", C_fine), ("coarse", C_coarse)):
        acc = None
        for a in arms:
            p = np.load(HERE / f"results/expA23_{a}/probs/{task}/{stem}.npy").astype(np.float32)
            acc = p if acc is None else acc + p
        up = F.interpolate(torch.from_numpy(acc / len(arms))[None], size=(oh, ow),
                           mode="bilinear", align_corners=False)
        pred = up.argmax(1)[0].numpy().astype(np.uint8)
        g = np.load(CACHE / f"gt_{task}" / f"{stem}.npy")
        classes = mod.CLASSES if task == "fine" else mod.CLASSES_MERGED
        sc = weighted_image_scores(pred, g, classes, sum(c.weight for c in classes))
        out[task] = (float(sc["dice"]), float(sc["hd"]))
    return stem, out


def evaluate(arms, pool, stems_by_fold) -> dict:
    """5 fold を **1 回の map** で回す（fold ごとにバリアを置くと 4K の遅い 1 枚に全員が待たされる）。
    4K を先頭に並べて chunksize=1 にし、遅いタスクから配る。"""
    global _SIZES
    if _SIZES is None:
        _SIZES = json.loads((CACHE / "sizes.json").read_text())
    stem_fold = {s: k for k in range(5) for s in stems_by_fold[k]}
    order = sorted(stem_fold, key=lambda s: -_SIZES[s][0] * _SIZES[s][1])
    res_all = pool.map(_score_one, [(s, arms) for s in order], chunksize=1)
    by_fold = {k: [] for k in range(5)}
    for stem, o in res_all:
        by_fold[stem_fold[stem]].append((stem, o))
    per_fold = []
    for k in range(5):
        per_case = {"fine": {}, "coarse": {}}
        for stem, o in by_fold[k]:
            case = stem.rsplit("_", 1)[0]
            for t, v in o.items():
                per_case[t].setdefault(case, []).append(v)
        row = {}
        for t in ("fine", "coarse"):
            cm = {c: (np.mean([x[0] for x in v]), np.mean([x[1] for x in v])) for c, v in per_case[t].items()}
            row[f"{t}_dice"] = float(np.mean([v[0] for v in cm.values()]))
            row[f"{t}_hd"] = float(np.mean([v[1] for v in cm.values()]))
        per_fold.append(row)
    m = {k: float(np.mean([r[k] for r in per_fold])) for k in per_fold[0]}
    m["per_fold"] = per_fold
    return m


def fmt(m) -> str:
    return (f"T1 Dice {m['coarse_dice']:.4f} HD {m['coarse_hd']:.4f} | "
            f"T2 Dice {m['fine_dice']:.4f} HD {m['fine_hd']:.4f}")


def rank_obj(m) -> float:
    """公式は Dice 降順と HD 昇順の順位平均。選択では両方を同じ向きに足した値を使う。"""
    return (m["fine_dice"] + m["coarse_dice"]) - (m["fine_hd"] + m["coarse_hd"])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--prepare", action="store_true")
    ap.add_argument("--greedy", action="store_true")
    ap.add_argument("--max-size", type=int, default=10)
    ap.add_argument("--jobs", type=int, default=64)
    ap.add_argument("--arms", nargs="*", default=None)
    ap.add_argument("--sets", nargs="*", default=[])
    ap.add_argument("--skip-singles", action="store_true",
                    help="単体評価を飛ばし --arms の並び順を単体順位として使う（HDD 読み込みが律速のとき）")
    ap.add_argument("--out", default=str(HERE / "ens_greedy_fast.json"))
    args = ap.parse_args()

    stems_by_fold = {k: [f.rsplit(".", 1)[0] for f in FOLDS[FOLDS.fold == k].filename] for k in range(5)}
    all_stems = [s for k in range(5) for s in stems_by_fold[k]]
    if args.prepare or not (CACHE / "sizes.json").exists():
        prepare(all_stems, args.jobs)
        if args.prepare and not (args.greedy or args.sets):
            return

    arms = args.arms or available_arms()
    print(f"候補 {len(arms)} レシピ: {', '.join(arms)}", flush=True)
    results = {}
    with mp.Pool(args.jobs) as pool:
        t0 = time.time()
        for spec in args.sets:
            a = [x.strip() for x in spec.split(",")]
            m = evaluate(a, pool, stems_by_fold)
            results[spec] = m
            print(f"{spec}\n  {fmt(m)}", flush=True)
        if args.greedy:
            if args.skip_singles:
                order = list(arms)
                s = time.time()
                best = evaluate([order[0]], pool, stems_by_fold)
                print(f"[単体] {order[0]:24s} {fmt(best)}  ({time.time() - s:.0f}s)", flush=True)
                results["singles"] = {order[0]: best}
            else:
                singles = {}
                for a in arms:
                    s = time.time()
                    m = evaluate([a], pool, stems_by_fold)
                    singles[a] = m
                    print(f"[単体] {a:24s} {fmt(m)}  ({time.time() - s:.0f}s)", flush=True)
                results["singles"] = singles
                Path(args.out).write_text(json.dumps(results, indent=1, ensure_ascii=False))
                order = sorted(arms, key=lambda a: -rank_obj(singles[a]))
                best = singles[order[0]]
            chosen = [order[0]]
            print(f"\n[貪欲] 開始 {chosen} -> {fmt(best)}", flush=True)
            hist = [{"members": list(chosen), **{k: v for k, v in best.items() if k != "per_fold"}}]
            while len(chosen) < args.max_size:
                cand = None
                for a in order:
                    if a in chosen:
                        continue
                    m = evaluate(chosen + [a], pool, stems_by_fold)
                    g = rank_obj(m) - rank_obj(best)
                    print(f"    +{a:24s} {fmt(m)}  Δ{g:+.4f}", flush=True)
                    if cand is None or g > cand[2]:
                        cand = (a, m, g)
                if cand is None or cand[2] <= 0:
                    print(f"[貪欲] 改善なしで停止 ({len(chosen)} メンバー)", flush=True)
                    break
                chosen.append(cand[0]); best = cand[1]
                print(f"[貪欲] +{cand[0]:24s} -> {fmt(best)}  (Δ{cand[2]:+.4f})", flush=True)
                hist.append({"members": list(chosen), **{k: v for k, v in best.items() if k != "per_fold"}})
                results["greedy"] = hist
                Path(args.out).write_text(json.dumps(results, indent=1, ensure_ascii=False))
        print(f"\n総時間 {(time.time() - t0) / 60:.1f} 分", flush=True)
    Path(args.out).write_text(json.dumps(results, indent=1, ensure_ascii=False))
    print(f"-> {args.out}")


if __name__ == "__main__":
    main()
