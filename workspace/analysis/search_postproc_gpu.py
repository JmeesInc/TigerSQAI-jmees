"""クラス別スケーリング係数の探索 (GPU ベクトル化版).

CPU 版は 1 評価あたり全画像を Python ループで回すため数秒かかり, 座標上昇法が
非現実的だった (coarse で 1 fold 6 時間)。全画像の確率を GPU に常駐させ,
argmax と混同行列を一括計算することで 1 評価を数十ミリ秒にする。

過学習を避けるため fold 単位の cross-fitting を行う:
  fold f の係数は f 以外の 4 fold で最適化し, f に適用して集計する。
  これは本番 (全 OOF で最適化 -> テストに適用) と同じ手続き。
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "reference/tigersqai_challenge"))
log = logging.getLogger("pp")


def load_task(task: str):
    if task == "fine":
        from metrics.classes import CLASSES as C, rgb_mask_to_label_mask as dec
        return C, dec, "masks_fine"
    from metrics.classes_merged import CLASSES_MERGED as C, rgb_mask_to_label_mask as dec
    return C, dec, "masks_coarse"


class Scorer:
    """全画像を GPU に常駐させ, alpha を変えた argmax -> weighted Dice を一括評価する。"""

    def __init__(self, probs, gt, case_idx, ids, weights, n_lab, device, chunk=48):
        self.p = probs                       # (N,C,H,W) fp16 on GPU
        self.gt = gt.long()                  # (N,H,W)
        self.case_idx = case_idx             # (N,) long
        self.n_case = int(case_idx.max()) + 1
        self.ids, self.w, self.n = ids, weights, n_lab
        self.dev, self.chunk = device, chunk

    def __call__(self, alpha):
        n_img = self.p.shape[0]
        per_img = torch.empty(n_img, device=self.dev)
        a = alpha.view(1, -1, 1, 1)
        for s in range(0, n_img, self.chunk):
            e = min(s + self.chunk, n_img)
            pred = (self.p[s:e].float() * a).argmax(1)
            b = pred.shape[0]
            flat = (self.gt[s:e] * self.n + pred).view(b, -1)
            cm = torch.zeros(b, self.n * self.n, device=self.dev)
            cm.scatter_add_(1, flat, torch.ones_like(flat, dtype=torch.float32))
            cm = cm.view(b, self.n, self.n)
            tp = torch.diagonal(cm, dim1=1, dim2=2)
            n_gt, n_pred = cm.sum(2), cm.sum(1)
            both0 = (n_gt == 0) & (n_pred == 0)
            one0 = ((n_gt == 0) | (n_pred == 0)) & ~both0
            d = 2 * tp / torch.clamp(n_gt + n_pred, min=1)
            d = torch.where(both0, torch.ones_like(d), torch.where(one0, torch.zeros_like(d), d))
            per_img[s:e] = (d[:, self.ids] * self.w).sum(1) / self.w.sum()
        cs = torch.zeros(self.n_case, device=self.dev)
        cn = torch.zeros(self.n_case, device=self.dev)
        cs.scatter_add_(0, self.case_idx, per_img)
        cn.scatter_add_(0, self.case_idx, torch.ones_like(per_img))
        return float((cs / torch.clamp(cn, min=1)).mean())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", choices=["fine", "coarse"], required=True)
    ap.add_argument("--probs", default="workspace/analysis/oof_probs")
    ap.add_argument("--folds-csv", default="workspace/fold/v3/folds.csv")
    ap.add_argument("--out", default="workspace/analysis/postproc")
    ap.add_argument("--grid", nargs="+", type=float,
                    default=[0.3, 0.5, 0.7, 0.85, 1.0, 1.2, 1.5, 2.0, 2.8, 4.0])
    ap.add_argument("--passes", type=int, default=3)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    out = REPO / args.out
    out.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s",
                        handlers=[logging.StreamHandler(),
                                  logging.FileHandler(out / f"gpu_{args.task}.log")])

    CLS, dec, gtdir = load_task(args.task)
    ids_np = np.array([c.label_id for c in CLS])
    n_lab = int(ids_np.max()) + 1
    dev = args.device
    ids = torch.as_tensor(ids_np, device=dev)
    w = torch.as_tensor([c.weight for c in CLS], dtype=torch.float32, device=dev)

    folds = pd.read_csv(REPO / args.folds_csv)
    P, G, C, FD, cases = [], [], [], [], {}
    for r in folds.itertuples():
        f = REPO / args.probs / args.task / f"{r.filename[:-4]}.npy"
        if not f.exists():
            continue
        pr = np.load(f)
        h, wd = pr.shape[1:]
        g = dec(np.asarray(Image.open(REPO / f"data/{gtdir}/{r.filename}")
                           .convert("RGB").resize((wd, h), Image.NEAREST)))
        P.append(torch.from_numpy(pr))
        G.append(torch.from_numpy(g.astype(np.int16)))
        C.append(cases.setdefault(r.case_id, len(cases)))
        FD.append(int(r.fold))

    probs = torch.stack(P).to(dev)
    gts = torch.stack(G).to(dev)
    cidx = torch.as_tensor(C, device=dev)
    fld = np.array(FD)
    n_ch = probs.shape[1]
    log.info("task=%s  %d 枚 / %d ch / %d case  (GPU %.1f GB)", args.task, probs.shape[0],
             n_ch, len(cases), probs.numel() * probs.element_size() / 1e9)

    ones = torch.ones(n_ch, device=dev)
    base = Scorer(probs, gts, cidx, ids, w, n_lab, dev)(ones)
    log.info("係数なし (argmax): %.4f", base)

    alphas = {}
    for f in sorted(set(FD)):
        fit_m = torch.as_tensor(fld != f, device=dev)
        idx = torch.nonzero(fit_m).squeeze(1)
        fit = Scorer(probs[idx], gts[idx], torch.unique(cidx[idx], return_inverse=True)[1],
                     ids, w, n_lab, dev)
        a = torch.ones(n_ch, device=dev)
        cur = fit(a)
        for p in range(args.passes):
            improved = 0
            for ci in ids_np:
                old = float(a[ci])
                best_v, best_s = old, cur
                for v in args.grid:
                    if abs(v - old) < 1e-9:
                        continue
                    a[ci] = v
                    s = fit(a)
                    if s > best_s + 1e-6:
                        best_v, best_s = v, s
                a[ci] = best_v
                if best_s > cur + 1e-6:
                    cur, improved = best_s, improved + 1
            log.info("  fold %d pass %d: fit %.4f (更新 %d クラス)", f, p, cur, improved)
            if improved == 0:
                break
        alphas[f] = a.cpu().numpy().tolist()
        ev = torch.nonzero(~fit_m).squeeze(1)
        ev_sc = Scorer(probs[ev], gts[ev], torch.unique(cidx[ev], return_inverse=True)[1],
                       ids, w, n_lab, dev)
        b4, af = ev_sc(ones), ev_sc(a)
        log.info("fold %d: 適用前 %.4f -> 適用後 %.4f (%+.4f)", f, b4, af, af - b4)

    alpha_map = torch.ones(probs.shape[0], n_ch, device=dev)
    for f, a in alphas.items():
        m = torch.as_tensor(fld == f, device=dev)
        alpha_map[m] = torch.as_tensor(a, dtype=torch.float32, device=dev)
    scaled = (probs.float() * alpha_map[:, :, None, None]).half()
    after = Scorer(scaled, gts, cidx, ids, w, n_lab, dev)(ones)
    log.info("=== cross-fit 適用後 全体: %.4f  (係数なし %.4f, %+.4f) ===", after, base, after - base)

    mean_a = np.mean([alphas[f] for f in alphas], axis=0)
    df = pd.DataFrame(dict(cls=[c.name for c in CLS], weight=[c.weight for c in CLS],
                           alpha_mean=mean_a[ids_np].round(3)))
    log.info("\n-- 1.0 から離れたクラス --\n%s",
             df.reindex(df.alpha_mean.sub(1).abs().sort_values(ascending=False).index)
               .head(12).to_string(index=False))
    json.dump(dict(task=args.task, base=base, after=after, alpha_per_fold=alphas,
                   alpha_mean=mean_a.tolist(), names=[c.name for c in CLS]),
              open(out / f"alpha_{args.task}.json", "w"), indent=2)
    df.to_csv(out / f"alpha_{args.task}.csv", index=False)


if __name__ == "__main__":
    main()
