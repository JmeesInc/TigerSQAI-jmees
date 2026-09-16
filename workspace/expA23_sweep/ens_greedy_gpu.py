"""expA23: 貪欲前向き選択（GPU 段 + CPU 公式採点の 2 段パイプライン版）.

ens_greedy_fast.py は CPU で 4K へ補間していて（1 枚 1GB のテンソル x 96 ワーカ）
メモリ帯域が飽和し、1 評価 246 秒 = 貪欲全体 6 時間超で締切に間に合わなかった。

  * GPU 段（主プロセス）: 選択済みメンバーの確率和 S を **全 526 枚ぶん GPU に常駐**
    （fp16 47ch x 288x512 x 526 = 7.3GB）。候補 c ごとに c の npy だけ読んで
    (S + c)/(n+1) → 原寸へ bilinear → argmax → uint8 を /dev/shm に書く。
  * CPU 段（Pool）: /dev/shm の予測ラベルとキャッシュ済み GT で **公式** weighted Dice + HD。

採点関数は公式実装そのままなので、選択の根拠は変わらない。
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
import torch
import torch.nn.functional as F

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(REPO / "reference" / "tigersqai_challenge"))
CACHE = HERE / "_cvcache"
SHM = Path("/dev/shm/ens_pred")
FOLDS = pd.read_csv(REPO / "workspace/fold/v2/folds.csv")
DEV = "cuda"


# ---------------------------------------------------------------- GPU 段
def load_probs(arm: str, stem: str, task: str) -> torch.Tensor:
    return torch.from_numpy(np.load(HERE / f"results/expA23_{arm}/probs/{task}/{stem}.npy")).to(DEV)


def build_base(chosen: list[str], stems: list[str]) -> dict:
    base = {}
    for s in stems:
        f = c = None
        for a in chosen:
            pf, pc = load_probs(a, s, "fine").float(), load_probs(a, s, "coarse").float()
            f = pf if f is None else f + pf
            c = pc if c is None else c + pc
        base[s] = (f.half(), c.half())          # 常駐は fp16
    return base


@torch.no_grad()
def gpu_predict(base: dict, n_base: int, cand: str | None, stems: list[str], sizes: dict) -> None:
    for s in stems:
        oh, ow = sizes[s]
        for task, i in (("fine", 0), ("coarse", 1)):
            acc = base[s][i].float()
            if cand is not None:
                acc = acc + load_probs(cand, s, task).float()
            up = F.interpolate(acc[None] / (n_base + (cand is not None)), size=(oh, ow),
                               mode="bilinear", align_corners=False)
            pred = up.argmax(1)[0].to(torch.uint8).cpu().numpy()
            np.save(SHM / f"{s}_{task}.npy", pred)
            del up


# ---------------------------------------------------------------- CPU 段
def _score_one(stem: str):
    from metrics import classes as C_fine
    from metrics import classes_merged as C_coarse
    from metrics.metrics import weighted_image_scores
    out = {}
    for task, mod in (("fine", C_fine), ("coarse", C_coarse)):
        pred = np.load(SHM / f"{stem}_{task}.npy")
        g = np.load(CACHE / f"gt_{task}" / f"{stem}.npy")
        classes = mod.CLASSES if task == "fine" else mod.CLASSES_MERGED
        sc = weighted_image_scores(pred, g, classes, sum(c.weight for c in classes))
        out[task] = (float(sc["dice"]), float(sc["hd"]))
    return stem, out


def cpu_score(pool, order: list[str], stem_fold: dict) -> dict:
    res = pool.map(_score_one, order, chunksize=1)
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


def evaluate(base, n_base, cand, order, stem_fold, sizes, pool):
    t = time.time()
    gpu_predict(base, n_base, cand, order, sizes)
    t_gpu = time.time() - t
    m = cpu_score(pool, order, stem_fold)
    m["_t"] = (t_gpu, time.time() - t - t_gpu)
    return m


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arms", nargs="+", required=True, help="単体順位の順（先頭が開始点）")
    ap.add_argument("--max-size", type=int, default=10)
    ap.add_argument("--jobs", type=int, default=96)
    ap.add_argument("--out", default=str(HERE / "ens_greedy_gpu.json"))
    args = ap.parse_args()
    torch.set_num_threads(4)
    SHM.mkdir(parents=True, exist_ok=True)
    sizes = json.loads((CACHE / "sizes.json").read_text())
    stem_fold = {f.rsplit(".", 1)[0]: int(k) for f, k in zip(FOLDS.filename, FOLDS.fold)}
    order = sorted(stem_fold, key=lambda s: -sizes[s][0] * sizes[s][1])   # 4K を先に配る

    results = {}
    with mp.Pool(args.jobs) as pool:
        t0 = time.time()
        chosen = [args.arms[0]]
        base = build_base(chosen, order)
        best = evaluate(base, 1, None, order, stem_fold, sizes, pool)
        print(f"[単体] {chosen[0]:24s} {fmt(best)}  (gpu {best['_t'][0]:.0f}s / cpu {best['_t'][1]:.0f}s)", flush=True)
        hist = [{"members": list(chosen), **{k: v for k, v in best.items() if k not in ("per_fold", "_t")}}]
        while len(chosen) < args.max_size:
            cand_best = None
            for a in args.arms:
                if a in chosen:
                    continue
                m = evaluate(base, len(chosen), a, order, stem_fold, sizes, pool)
                g = rank_obj(m) - rank_obj(best)
                print(f"    +{a:24s} {fmt(m)}  Δ{g:+.4f}  (gpu {m['_t'][0]:.0f}s / cpu {m['_t'][1]:.0f}s)", flush=True)
                if cand_best is None or g > cand_best[2]:
                    cand_best = (a, m, g)
            if cand_best is None or cand_best[2] <= 0:
                print(f"[貪欲] 改善なしで停止 ({len(chosen)} メンバー)", flush=True)
                break
            chosen.append(cand_best[0]); best = cand_best[1]
            print(f"[貪欲] +{cand_best[0]:24s} -> {fmt(best)}  (Δ{cand_best[2]:+.4f})  経過 {(time.time() - t0) / 60:.0f} 分", flush=True)
            hist.append({"members": list(chosen), **{k: v for k, v in best.items() if k not in ("per_fold", "_t")}})
            results["greedy"] = hist
            Path(args.out).write_text(json.dumps(results, indent=1, ensure_ascii=False))
            base = build_base(chosen, order)
            torch.cuda.empty_cache()
        print(f"\n総時間 {(time.time() - t0) / 60:.1f} 分", flush=True)
    results["greedy"] = hist
    Path(args.out).write_text(json.dumps(results, indent=1, ensure_ascii=False))
    print(f"-> {args.out}")


if __name__ == "__main__":
    main()
