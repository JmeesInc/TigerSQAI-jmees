"""解剖グラフのルールによる後処理を OOF 確率 (ens5) で評価する (学習なし).

ルール (fold 別 rules npz, val 画像を含まない GT 統計から生成):
  excl : 排他ペア (共起ゼロ) が両方出ていたら, 確信度 (自クラス確率の総和) が低い方を次点クラスに塗り替える
  adj  : 禁止隣接 (W >= w_thr) で接している成分対のうち, 自クラス平均確率が低い方の成分を次点クラスに塗り替える
  count: 成分数が K_max を超えるクラスは (面積×平均確率) 上位 K_max 個だけ残し, 残りを次点に塗り替える
採点は search_postproc.py と同じ (公式規約の weighted Dice を確率解像度で, case→全体の階層平均)。
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from scipy import ndimage as ndi

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "reference/tigersqai_challenge"))
sys.path.insert(0, str(REPO / "workspace/analysis"))
from search_postproc import load_task, weighted_dice  # noqa: E402

log = logging.getLogger("rule_postproc")
STRUCT = np.ones((3, 3), bool)


def runner_up(probs: np.ndarray, mask: np.ndarray, exclude: list[int]) -> np.ndarray:
    p = probs[:, mask].copy()
    p[exclude] = -1
    return p.argmax(0)


def apply_rules(probs: np.ndarray, R: dict, steps: tuple[str, ...], min_area: int, w_thr: float,
                log_events: list | None = None) -> np.ndarray:
    W, E, K = R["W_adj"], R["E_excl"], R["K_max"]
    C = W.shape[0]
    pred = probs.argmax(0)

    if "excl" in steps:
        changed = True
        while changed:
            changed = False
            areas = np.bincount(pred.ravel(), minlength=C)
            present = np.flatnonzero(areas >= min_area)
            for a in present:
                for b in present:
                    if b <= a or E[a, b] == 0:
                        continue
                    conf_a = probs[a][pred == a].sum()
                    conf_b = probs[b][pred == b].sum()
                    loser = a if conf_a < conf_b else b
                    m = pred == loser
                    pred[m] = runner_up(probs, m, [loser])
                    if log_events is not None:
                        log_events.append(("excl", int(a), int(b), int(loser)))
                    changed = True
                    break
                if changed:
                    break

    if "adj" in steps:
        for _ in range(3):
            # 成分ラベル付け
            comp = np.zeros_like(pred, dtype=np.int32)
            comp_cls, comp_conf, comp_area = [0], [0.0], [0]
            k0 = 1
            for c in range(1, C):
                if (W[c] > 0).any() is False:
                    continue
                cc, k = ndi.label(pred == c, structure=STRUCT)
                if k == 0:
                    continue
                comp[cc > 0] = cc[cc > 0] + k0 - 1
                for i in range(1, k + 1):
                    m = cc == i
                    comp_cls.append(c); comp_area.append(int(m.sum())); comp_conf.append(float(probs[c][m].mean()))
                k0 += k
            comp_cls = np.array(comp_cls); comp_conf = np.array(comp_conf); comp_area = np.array(comp_area)
            # 隣接成分対の境界長
            pairs = {}
            for A, B in ((comp[:, :-1], comp[:, 1:]), (comp[:-1, :], comp[1:, :])):
                m = (A != B) & (A > 0) & (B > 0)
                for i, j in zip(A[m], B[m]):
                    key = (min(i, j), max(i, j))
                    pairs[key] = pairs.get(key, 0) + 1
            viol = [(n, i, j) for (i, j), n in pairs.items() if W[comp_cls[i], comp_cls[j]] >= w_thr]
            if not viol:
                break
            viol.sort(reverse=True)
            removed = set()
            for n, i, j in viol:
                if i in removed or j in removed:
                    continue
                if comp_area[i] < min_area and comp_area[j] < min_area:
                    continue
                # 確信度が低い方を消す (小さすぎる成分は確信度に関わらず消す)
                loser = i if (comp_conf[i] < comp_conf[j]) else j
                if comp_area[i] < min_area <= comp_area[j]:
                    loser = i
                elif comp_area[j] < min_area <= comp_area[i]:
                    loser = j
                m = comp == loser
                pred[m] = runner_up(probs, m, [int(comp_cls[loser])])
                removed.add(loser)
                if log_events is not None:
                    log_events.append(("adj", int(comp_cls[i]), int(comp_cls[j]), int(comp_cls[loser]), int(n)))

    if "count" in steps:
        for c in range(1, C):
            if K[c] >= 99:
                continue
            cc, k = ndi.label(pred == c, structure=STRUCT)
            if k <= K[c]:
                continue
            score = []
            for i in range(1, k + 1):
                m = cc == i
                score.append(m.sum() * probs[c][m].mean())
            order = np.argsort(score)[::-1]
            for i in order[K[c]:]:
                m = cc == (i + 1)
                pred[m] = runner_up(probs, m, [c])
                if log_events is not None:
                    log_events.append(("count", c, int(k), int(K[c])))
    return pred


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", choices=["fine", "coarse"], required=True)
    ap.add_argument("--probs", default="workspace/analysis/oof_probs")
    ap.add_argument("--folds-csv", default="workspace/fold/v1/folds.csv")
    ap.add_argument("--rules-dir", default="workspace/anatomy_graph/out")
    ap.add_argument("--min-area", type=int, default=30, help="確率解像度 (288x512) での最小成分面積")
    ap.add_argument("--w-thr", type=float, default=0.8)
    ap.add_argument("--out", default="workspace/anatomy_graph/out/postproc")
    args = ap.parse_args()
    out = REPO / args.out; out.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s",
                        handlers=[logging.StreamHandler(), logging.FileHandler(out / f"rule_postproc_{args.task}.log")])
    CLS, WT, dec, gtdir = load_task(args.task)
    ids = np.array([c.label_id for c in CLS]); weights = np.array([c.weight for c in CLS], dtype=np.float64)
    n_lab = int(ids.max()) + 1
    folds = pd.read_csv(REPO / args.folds_csv)
    rules = {f: dict(np.load(REPO / args.rules_dir / f"rules_{args.task}_fold{f}.npz", allow_pickle=True))
             for f in sorted(folds.fold.unique())}
    names = [str(x) for x in rules[0]["names"]]

    data = []
    for r in folds.itertuples():
        p = REPO / args.probs / args.task / f"{r.filename[:-4]}.npy"
        if not p.exists():
            continue
        probs = np.load(p).astype(np.float32)
        h, w = probs.shape[1:]
        gt = dec(np.asarray(Image.open(REPO / f"data/{gtdir}/{r.filename}").convert("RGB").resize((w, h), Image.NEAREST)))
        data.append((r.case_id, int(r.fold), r.filename, probs, gt))
    log.info("task=%s images=%d", args.task, len(data))

    configs = [("none", ()), ("excl", ("excl",)), ("adj", ("adj",)), ("count", ("count",)),
               ("excl+adj", ("excl", "adj")), ("excl+adj+count", ("excl", "adj", "count"))]
    results = {}
    for name, steps in configs:
        per_case, per_img, events = {}, [], []
        for case, f, fn, probs, gt in data:
            pred = apply_rules(probs, rules[f], steps, args.min_area, args.w_thr, events) if steps else probs.argmax(0)
            d = weighted_dice(pred.astype(np.int64), gt, ids, weights, n_lab)
            per_case.setdefault(case, []).append(d); per_img.append((fn, d))
        score = float(np.mean([np.mean(v) for v in per_case.values()]))
        results[name] = {"score": score, "per_img": dict(per_img), "n_events": len(events)}
        ev = pd.Series([e[0] for e in events]).value_counts().to_dict() if events else {}
        log.info("%-16s weighted Dice (case 平均) = %.4f  events=%s", name, score, ev)
        if name == "adj":
            top = pd.Series([f"{names[e[1]]}|{names[e[2]]}->drop {names[e[3]]}" for e in events]).value_counts().head(8)
            log.info("  adj top relabels:\n%s", top.to_string())
    base = results["none"]["score"]
    for k, v in results.items():
        log.info("%-16s %.4f (%+.4f)", k, v["score"], v["score"] - base)
    json.dump({k: {"score": v["score"], "n_events": v["n_events"]} for k, v in results.items()},
              open(out / f"rule_postproc_{args.task}.json", "w"), indent=2)
    # 画像単位の差分 (adj) を保存 → 目視レビュー用
    diff = pd.DataFrame({"filename": list(results["none"]["per_img"]),
                         "base": list(results["none"]["per_img"].values()),
                         "excl_adj": [results["excl+adj"]["per_img"][k] for k in results["none"]["per_img"]]})
    diff["delta"] = diff.excl_adj - diff.base
    diff.sort_values("delta").to_csv(out / f"per_image_{args.task}.csv", index=False)
    log.info("per-image delta: mean %+.4f, improved %d / worsened %d",
             diff.delta.mean(), (diff.delta > 1e-6).sum(), (diff.delta < -1e-6).sum())


if __name__ == "__main__":
    main()
