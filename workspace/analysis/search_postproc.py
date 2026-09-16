"""OOF 確率に対してクラス別スケーリング係数を最適化する (学習なしの後処理).

背景:
  公式 weighted Dice はクラス単位で「両方空=1.0 / 片方だけ空=0.0」という
  全か無かの規約を持つ。argmax はこの構造を考慮しないため,
    - GT にあるクラスを 1px も出さない -> 0.0  (Task2 fine で損失 0.150)
    - GT に無いクラスを出してしまう    -> 0.0  (Task1 coarse で損失 0.106)
  という取りこぼしが出る。argmax 前に p_c <- alpha_c * p_c を掛けて調整する。

過学習を避けるため fold 単位の cross-fitting を行う:
  fold f の係数は「f 以外の 4 fold の OOF」で最適化し, f に適用して集計する。
  これは本番 (全 OOF で最適化 -> テストに適用) と同じ手続き。
"""
from __future__ import annotations
import argparse, json, logging, sys
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "reference/tigersqai_challenge"))
log = logging.getLogger("postproc")


def load_task(task):
    if task == "fine":
        from metrics.classes import CLASSES as C, WEIGHT_TOTAL as W, rgb_mask_to_label_mask as dec
        return C, W, dec, "masks_fine"
    from metrics.classes_merged import (CLASSES_MERGED as C, WEIGHT_TOTAL_MERGED as W,
                                        rgb_mask_to_label_mask as dec)
    return C, W, dec, "masks_coarse"


def weighted_dice(pred, gt, ids, weights, n):
    """公式規約 (両方空=1.0, 片方だけ空=0.0) の weighted Dice。pred/gt は (H,W) int。"""
    # gt は uint8 なので必ず int64 に上げる。fine (n=31) では gt=27 のとき 27*31=837 が
    # uint8 を溢れて 69 に巻き戻り, クラスが入れ替わる (coarse は n=16 で偶然溢れない)。
    g = gt.ravel().astype(np.int64)
    pr = pred.ravel().astype(np.int64)
    cm = np.bincount(g * n + pr, minlength=n * n).reshape(n, n)
    tp = np.diag(cm).astype(np.float64)
    n_gt, n_pred = cm.sum(1).astype(np.float64), cm.sum(0).astype(np.float64)
    both0 = (n_gt == 0) & (n_pred == 0)
    d = np.where(both0, 1.0, np.where((n_gt == 0) | (n_pred == 0), 0.0,
                                      2 * tp / np.maximum(n_gt + n_pred, 1)))
    return float((d[ids] * weights).sum() / weights.sum())


def score_set(data, alpha, ids, weights, n):
    """case -> 画像平均 -> 全体平均 (公式の階層集計)。

    alpha は確率チャンネル順 (= label_id 順) のベクトル。argmax の結果はそのまま
    label_id になる。metrics.classes.CLASSES は weight 順に並んでおり label_id 順
    ではないため, ここで ids を添字に使うとクラスが入れ替わる。
    """
    per_case = {}
    for case, probs, gt in data:
        pred = (probs * alpha[:, None, None]).argmax(0).astype(np.int64)
        per_case.setdefault(case, []).append(weighted_dice(pred, gt, ids, weights, n))
    return float(np.mean([np.mean(v) for v in per_case.values()]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", choices=["fine", "coarse"], required=True)
    ap.add_argument("--probs", default="workspace/analysis/oof_probs")
    ap.add_argument("--folds-csv", default="workspace/fold/v3/folds.csv")
    ap.add_argument("--out", default="workspace/analysis/postproc")
    ap.add_argument("--grid", nargs="+", type=float,
                    default=[0.4, 0.6, 0.8, 1.0, 1.3, 1.7, 2.2, 3.0, 4.0])
    ap.add_argument("--passes", type=int, default=2)
    args = ap.parse_args()

    out = REPO / args.out; out.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s",
                        handlers=[logging.StreamHandler(),
                                  logging.FileHandler(out / f"search_{args.task}.log")])

    CLS, WT, dec, gtdir = load_task(args.task)
    ids = np.array([c.label_id for c in CLS])
    weights = np.array([c.weight for c in CLS], dtype=np.float64)
    names = [c.name for c in CLS]
    n_lab = int(ids.max()) + 1
    folds = pd.read_csv(REPO / args.folds_csv)

    # 読み込み (確率と同じ解像度に GT を最近傍で落とす)
    by_fold = {f: [] for f in sorted(folds.fold.unique())}
    for r in folds.itertuples():
        p = REPO / args.probs / args.task / f"{r.filename[:-4]}.npy"
        if not p.exists():
            continue
        probs = np.load(p).astype(np.float32)
        h, w = probs.shape[1:]
        gt = dec(np.asarray(Image.open(REPO / f"data/{gtdir}/{r.filename}")
                            .convert("RGB").resize((w, h), Image.NEAREST)))
        by_fold[int(r.fold)].append((r.case_id, probs, gt))
    total = sum(len(v) for v in by_fold.values())
    log.info("task=%s  読み込み %d 枚 / %d クラス", args.task, total, len(CLS))
    if total == 0:
        log.error("確率ファイルが無い"); return

    n_ch = int(max(n_lab, int(ids.max()) + 1))
    base_all = score_set([x for v in by_fold.values() for x in v],
                         np.ones(n_ch), ids, weights, n_lab)
    log.info("係数なし (argmax) の全体スコア: %.4f", base_all)

    # fold 単位 cross-fitting
    applied, alphas = [], {}
    for f in sorted(by_fold):
        fit = [x for g, v in by_fold.items() if g != f for x in v]
        a = np.ones(n_ch)
        cur = score_set(fit, a, ids, weights, n_lab)
        for p in range(args.passes):
            for ci in ids:          # 採点対象クラスの label_id を直接動かす
                best_v, best_s = a[ci], cur
                for v in args.grid:
                    if v == a[ci]:
                        continue
                    a[ci] = v
                    s = score_set(fit, a, ids, weights, n_lab)
                    if s > best_s:
                        best_v, best_s = v, s
                a[ci], cur = best_v, best_s
            log.info("  fold %d pass %d: fit スコア %.4f", f, p, cur)
        applied += [(c, (pr * a[:, None, None]), g) for c, pr, g in by_fold[f]]
        alphas[f] = a.tolist()
        log.info("fold %d 完了 (fit %.4f)", f, cur)

    # 適用後の全体スコア (各 fold は未使用データで最適化された係数で採点されている)
    per_case = {}
    for case, probs, gt in applied:
        pred = ids[probs.argmax(0)]
        per_case.setdefault(case, []).append(weighted_dice(pred, gt, ids, weights, n_lab))
    after = float(np.mean([np.mean(v) for v in per_case.values()]))
    log.info("cross-fit 適用後: %.4f  (係数なし %.4f, 差 %+.4f)", after, base_all, after - base_all)

    mean_a = np.mean([alphas[f] for f in alphas], axis=0)
    df = pd.DataFrame(dict(cls=names, weight=weights.astype(int),
                           alpha_mean=mean_a[ids].round(3)))
    log.info("\n-- 係数が 1.0 から離れたクラス --\n%s",
             df.reindex(df.alpha_mean.sub(1).abs().sort_values(ascending=False).index)
               .head(12).to_string(index=False))
    json.dump(dict(task=args.task, base=base_all, after=after,
                   alpha_per_fold=alphas, alpha_mean=mean_a.tolist(), names=names),
              open(out / f"alpha_{args.task}.json", "w"), indent=2)
    df.to_csv(out / f"alpha_{args.task}.csv", index=False)


if __name__ == "__main__":
    main()
