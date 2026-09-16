"""expA23: 貪欲前向き選択（複数 GPU ワーカ版）.

1 画像の処理を丸ごと GPU ワーカで行う: メンバー npy 読み → 和 → 原寸 bilinear → argmax
→ GPU 指標（Dice 厳密 / HD 近似, gpu_metrics.py）。ワーカは spawn で作り、
worker i は GPU[i % n] に載る。Dice は公式と厳密一致、HD は最大誤差 3e-4（validate_gpu_metrics.py）。
"""
from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
CACHE = HERE / "_cvcache"
FOLDS = pd.read_csv(REPO / "workspace/fold/v2/folds.csv")

_G = {}


def _init(gpus: list[int], counter):
    import torch
    sys.path.insert(0, str(REPO / "reference" / "tigersqai_challenge")); sys.path.insert(0, str(HERE))
    with counter.get_lock():
        i = counter.value; counter.value += 1
    dev = f"cuda:{gpus[i % len(gpus)]}"
    torch.cuda.set_device(dev)
    torch.set_num_threads(1)
    from metrics import classes as CF, classes_merged as CC
    from gpu_metrics import weighted_scores_gpu
    _G.update(dev=dev, CF=CF.CLASSES, CC=CC.CLASSES_MERGED, score=weighted_scores_gpu,
              sizes=json.loads((CACHE / "sizes.json").read_text()), hd_long=int(os.environ.get("HD_LONG", "640")))


def _score_one(args):
    import torch, torch.nn.functional as F
    stem, arms = args
    dev = _G["dev"]; oh, ow = _G["sizes"][stem]
    out = {}
    with torch.no_grad():
        for task, cls in (("fine", _G["CF"]), ("coarse", _G["CC"])):
            # ARM_WEIGHTS="q_endovis18_dlv3:3,r_xl:2" : 指定メンバーの重み（既定 1）
            aw = dict((k, float(v)) for k, v in (x.split(":") for x in os.environ.get("ARM_WEIGHTS", "").split(",") if x))
            acc = None; denom = 0.0
            for a in arms:
                w = aw.get(a, 1.0)
                p = torch.from_numpy(np.load(HERE / f"results/expA23_{a}/probs/{task}/{stem}.npy")).to(dev).float() * w
                acc = p if acc is None else acc + p
                denom += w
            # FINE_EXTRA="ft_t2_fine:4" : fine 側にだけ重み付きで足す（コンテナの fine_only/weight と同じ扱い）
            fx = os.environ.get("FINE_EXTRA")
            if fx and task == "fine":
                name, w = fx.split(":"); w = float(w)
                acc = acc + w * torch.from_numpy(np.load(HERE / f"results/expA23_{name}/probs/fine/{stem}.npy")).to(dev).float()
                denom += w
            pred = F.interpolate(acc[None] / denom, size=(oh, ow), mode="bilinear",
                                 align_corners=False).argmax(1)[0].to(torch.uint8)
            gt = torch.from_numpy(np.load(CACHE / f"gt_{task}" / f"{stem}.npy")).to(dev)
            out[task] = _G["score"](pred, gt, cls, hd_long=_G["hd_long"])
    return stem, out


def evaluate(arms, pool, order, stem_fold) -> dict:
    res = pool.map(_score_one, [(s, arms) for s in order], chunksize=2)
    by_fold = {k: {"fine": {}, "coarse": {}} for k in range(5)}
    for stem, o in res:
        k = stem_fold[stem]; case = stem.rsplit("_", 1)[0]
        for t, v in o.items():
            by_fold[k][t].setdefault(case, []).append(v)
    per_fold = []
    for k in range(5):
        row = {}
        for t in ("fine", "coarse"):
            cm = {c: (np.mean([x[0] for x in v]), np.mean([x[1] for x in v])) for c, v in by_fold[k][t].items()}
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
    return (m["fine_dice"] + m["coarse_dice"]) - (m["fine_hd"] + m["coarse_hd"])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arms", nargs="+", required=True)
    ap.add_argument("--gpus", type=int, nargs="+", default=[0, 2])
    ap.add_argument("--per-gpu", type=int, default=3)
    ap.add_argument("--max-size", type=int, default=10)
    ap.add_argument("--sets", nargs="*", default=[])
    ap.add_argument("--out", default=str(HERE / "ens_greedy_mgpu.json"))
    args = ap.parse_args()
    sizes = json.loads((CACHE / "sizes.json").read_text())
    stem_fold = {f.rsplit(".", 1)[0]: int(k) for f, k in zip(FOLDS.filename, FOLDS.fold)}
    order = sorted(stem_fold, key=lambda s: -sizes[s][0] * sizes[s][1])
    ctx = mp.get_context("spawn")
    counter = ctx.Value("i", 0)
    results = {}
    with ctx.Pool(len(args.gpus) * args.per_gpu, initializer=_init, initargs=(args.gpus, counter)) as pool:
        t0 = time.time()
        for spec in args.sets:
            a = [x.strip() for x in spec.split(",")]; t = time.time()
            m = evaluate(a, pool, order, stem_fold); results[spec] = m
            print(f"{spec}\n  {fmt(m)}  ({time.time() - t:.0f}s)", flush=True)
        if args.max_size > 0 and not args.sets:
            chosen = [args.arms[0]]; t = time.time()
            best = evaluate(chosen, pool, order, stem_fold)
            print(f"[単体] {chosen[0]:24s} {fmt(best)}  ({time.time() - t:.0f}s)", flush=True)
            hist = [{"members": list(chosen), **{k: v for k, v in best.items() if k != "per_fold"}}]
            while len(chosen) < args.max_size:
                cb = None
                for a in args.arms:
                    if a in chosen:
                        continue
                    t = time.time(); m = evaluate(chosen + [a], pool, order, stem_fold)
                    g = rank_obj(m) - rank_obj(best)
                    print(f"    +{a:24s} {fmt(m)}  Δ{g:+.4f}  ({time.time() - t:.0f}s)", flush=True)
                    if cb is None or g > cb[2]:
                        cb = (a, m, g)
                if cb is None or cb[2] <= 0:
                    print(f"[貪欲] 改善なしで停止 ({len(chosen)} メンバー)", flush=True); break
                chosen.append(cb[0]); best = cb[1]
                print(f"[貪欲] +{cb[0]:24s} -> {fmt(best)}  (Δ{cb[2]:+.4f})  経過 {(time.time() - t0) / 60:.0f} 分", flush=True)
                hist.append({"members": list(chosen), **{k: v for k, v in best.items() if k != "per_fold"}})
                results["greedy"] = hist
                Path(args.out).write_text(json.dumps(results, indent=1, ensure_ascii=False))
            results["greedy"] = hist
        print(f"\n総時間 {(time.time() - t0) / 60:.1f} 分", flush=True)
    Path(args.out).write_text(json.dumps(results, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()
