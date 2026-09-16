"""GT と OOF 予測 (ens5) のルール違反率を測る診断. ルール loss / 後処理の伸びしろを見る.

adj_rate  = 異クラス境界画素対のうち禁止 (W 重み付き) の割合
excl_pairs = 1 画像あたりの排他ペア共起数
per-class: 禁止境界に関与した画素対の多いクラス対 top-N
"""
from __future__ import annotations
import argparse, sys
from pathlib import Path
import cv2, numpy as np, pandas as pd, torch
REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "workspace/expA22_anatomy_rules"))
from anatomy_rules import AnatomyRuleLoss  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", default="fine")
    ap.add_argument("--probs", default="workspace/analysis/oof_probs")
    ap.add_argument("--rules", default=None)
    ap.add_argument("--folds-csv", default="workspace/fold/v1/folds.csv")
    ap.add_argument("--top", type=int, default=15)
    args = ap.parse_args()
    rules = args.rules or f"workspace/anatomy_graph/out/rules_{args.task}.npz"
    R = AnatomyRuleLoss(REPO / rules)
    C = R.W.shape[0]
    folds = pd.read_csv(REPO / args.folds_csv)
    lab_dir = REPO / f"workspace/data_proc/labels_{args.task}_1024"
    stats = {"gt": [], "pred": []}
    pair_mass = {"gt": torch.zeros(C, C), "pred": torch.zeros(C, C)}
    n = 0
    for r in folds.itertuples():
        p = REPO / args.probs / args.task / f"{r.filename[:-4]}.npy"
        if not p.exists():
            continue
        probs = np.load(p)
        h, w = probs.shape[1:]
        gt = cv2.resize(cv2.imread(str(lab_dir / r.filename), cv2.IMREAD_GRAYSCALE), (w, h), interpolation=cv2.INTER_NEAREST)
        pred = probs.argmax(0)
        for k, lab in (("gt", gt), ("pred", pred)):
            t = torch.from_numpy(lab.astype(np.int64))[None]
            stats[k].append(R.hard_violations(t))
            oh = torch.nn.functional.one_hot(t, C).permute(0, 3, 1, 2).float()
            for A, B in R._pairs(oh):
                pair_mass[k] += torch.einsum("bchw,bdhw->cd", A, B) * R.W
        n += 1
    print(f"task={args.task} rules={rules} images={n}")
    for k in ("gt", "pred"):
        a = np.mean([s["adj_rate"] for s in stats[k]]); e = np.mean([s["excl_pairs"] for s in stats[k]])
        frac_e = np.mean([s["excl_pairs"] > 0 for s in stats[k]])
        print(f"  {k:4s}: adj_rate={a:.4f}  excl_pairs/img={e:.3f}  imgs_with_excl={frac_e:.3f}")
    M = pair_mass["pred"] + pair_mass["pred"].T
    iu = np.triu_indices(C, 1)
    order = np.argsort(-M[iu].numpy())[: args.top]
    print("  top forbidden boundary pairs in PRED (weighted pixel pairs, total over OOF):")
    for i in order:
        a, b = iu[0][i], iu[1][i]
        if M[a, b] > 0:
            print(f"    {R.names[a]:34s} | {R.names[b]:34s} | {M[a,b]:9.0f}  (gt {pair_mass['gt'][a,b]+pair_mass['gt'][b,a]:.0f})")


if __name__ == "__main__":
    main()
