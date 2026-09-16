"""TTA と アンサンブル本数 のどちらが推論パスあたり有利かを OOF で直接比較する.

OOF では各画像を「その画像を学習に含まない fold のモデル 5 本」で予測する。
同じ推論パス数で
    モデルを増やす  vs  TTA を増やす
のどちらが得かを, 実際のスコアで測る。
"""
from __future__ import annotations
import argparse, json, logging, sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from PIL import Image

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "workspace/expE01_ensemble"))
sys.path.insert(0, str(REPO / "reference/tigersqai_challenge"))
sys.path.insert(0, str(REPO / "workspace/analysis"))
from dataset import IMAGENET_MEAN, IMAGENET_STD            # noqa: E402
from predict_ens import MEMBERS, load_member               # noqa: E402
from search_postproc import weighted_dice                  # 公式と一致を検証済み  # noqa: E402

log = logging.getLogger("tta")


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--folds-csv", default="workspace/fold/v3/folds.csv")
    ap.add_argument("--folds", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    ap.add_argument("--out", default="workspace/analysis/tta")
    ap.add_argument("--img-h", type=int, default=576)
    ap.add_argument("--img-w", type=int, default=1024)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    out = REPO / args.out; out.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s",
                        handlers=[logging.StreamHandler(),
                                  logging.FileHandler(out / f"tta_{args.folds[0]}.log")])

    from metrics.classes import CLASSES, rgb_mask_to_label_mask as r2f
    from metrics.classes_merged import CLASSES_MERGED, rgb_mask_to_label_mask as r2c
    spec = {"fine": (CLASSES, r2f, "masks_fine"), "coarse": (CLASSES_MERGED, r2c, "masks_coarse")}
    meta = {t: (np.array([c.label_id for c in C]),
                np.array([c.weight for c in C], float),
                int(max(c.label_id for c in C)) + 1) for t, (C, _, _) in spec.items()}

    folds = pd.read_csv(REPO / args.folds_csv)
    mean = torch.tensor(IMAGENET_MEAN, device=args.device).view(1, 3, 1, 1)
    std = torch.tensor(IMAGENET_STD, device=args.device).view(1, 3, 1, 1)
    names = list(MEMBERS)
    # 構成: (使用モデル数, TTA 種別) -> 推論パス数 = n_model * n_tta
    configs = [(n, tta) for n in (1, 2, 3, 5) for tta in ("plain", "hflip")]
    rows = []

    for fold in args.folds:
        val = folds[folds.fold == fold]
        models = [load_member(m, fold, args.device).half() for m in names]
        log.info("fold %d: %d 枚 / モデル %s", fold, len(val), names)
        for row in val.itertuples():
            bgr = cv2.imread(str(REPO / "workspace/data_proc/images_1024" / row.filename))
            if bgr is None:
                continue
            rgb = cv2.cvtColor(cv2.resize(bgr, (args.img_w, args.img_h),
                                          interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2RGB)
            x = torch.from_numpy(rgb).permute(2, 0, 1)[None].float().to(args.device) / 255.0
            x = ((x - mean) / std).half()
            xf = torch.flip(x, dims=[3])
            # per-model x per-view の確率を貯める
            pv = {"fine": {}, "coarse": {}}
            for mi, m in enumerate(models):
                lf, lc = m(x)
                lff, lcf = m(xf)
                for t, (a, b) in (("fine", (lf, lff)), ("coarse", (lc, lcf))):
                    pv[t][(mi, "id")] = a.float().softmax(1)
                    pv[t][(mi, "fl")] = torch.flip(b.float().softmax(1), dims=[3])
            gts = {}
            for t, (C, dec, gd) in spec.items():
                g = dec(np.asarray(Image.open(REPO / f"data/{gd}/{row.filename}")
                                   .convert("RGB").resize((args.img_w, args.img_h), Image.NEAREST)))
                gts[t] = g
            for n_model, tta in configs:
                views = ["id"] if tta == "plain" else ["id", "fl"]
                for t in spec:
                    acc = None
                    for mi in range(n_model):
                        for v in views:
                            p = pv[t][(mi, v)]
                            acc = p if acc is None else acc + p
                    pred = (acc / (n_model * len(views)))[0].argmax(0).cpu().numpy()
                    ids, w, n = meta[t]
                    rows.append(dict(fold=fold, case=row.case_id, file=row.filename, task=t,
                                     n_model=n_model, tta=tta, passes=n_model * len(views),
                                     dice=weighted_dice(pred, gts[t], ids, w, n)))
        del models
        torch.cuda.empty_cache()
        pd.DataFrame(rows).to_csv(out / f"scores_fold{args.folds[0]}.csv", index=False)
        log.info("fold %d 完了", fold)

    d = pd.DataFrame(rows)
    agg = (d.groupby(["task", "n_model", "tta", "passes", "case"]).dice.mean()
             .groupby(["task", "n_model", "tta", "passes"]).mean().reset_index())
    log.info("\n%s", agg.pivot_table(index=["task", "n_model", "passes"],
                                     columns="tta", values="dice").round(4).to_string())


if __name__ == "__main__":
    main()
