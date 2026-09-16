"""生成クリップに ens5 (25 モデル) を当てて擬似ラベルを作り, 生成フレームの
ドメインシフトを測る.

擬似ラベルは「端点フレームを学習に含む fold のモデル」で作る = そのクリップの case が
val になっている fold を除外し, 残り 4 fold x 5 メンバー = 20 モデルの softmax 平均.
端点で GT を再現できるモデルなら, 中間フレームの予測も端点に繋がるはずという発想.

診断として次を測る:
  (a) 実 first.png への予測 vs 生成 f000.png への予測  → 生成フレームのドメインシフト
  (b) 生成 f000.png への予測 vs 実 GT_A               → 擬似ラベルのアンカー精度
  (c) 実 first.png への予測 vs 実 GT_A                → 上限の参照 (学習済みなので楽観的)
"""
from __future__ import annotations
import argparse, glob, json, logging, os, sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import yaml

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "workspace/expE01_ensemble"))
sys.path.insert(0, str(REPO / "reference/tigersqai_challenge"))
from dataset import IMAGENET_MEAN, IMAGENET_STD  # noqa: E402
from model import DualHeadUnetPP  # noqa: E402
from predict_ens import MEMBERS, FOLDS_CSV, load_member, build_id2rgb  # noqa: E402

log = logging.getLogger("pseudo")


def preprocess(bgr, img_h, img_w, device, mean, std):
    rgb = cv2.cvtColor(cv2.resize(bgr, (img_w, img_h), interpolation=cv2.INTER_AREA),
                       cv2.COLOR_BGR2RGB)
    x = torch.from_numpy(rgb).permute(2, 0, 1)[None].float().to(device) / 255.0
    return (x - mean) / std


@torch.no_grad()
def predict(models, bgr, img_h, img_w, out_hw, device, mean, std):
    x = preprocess(bgr, img_h, img_w, device, mean, std)
    pf = pc = None
    for m in models:
        with torch.autocast("cuda", dtype=torch.float16):
            lf, lc = m(x)
        sf, sc = lf.float().softmax(1), lc.float().softmax(1)
        pf = sf if pf is None else pf + sf
        pc = sc if pc is None else pc + sc
    out = []
    for p in (pf, pc):
        up = F.interpolate(p / len(models), size=out_hw, mode="bilinear", align_corners=False)
        out.append(up[0].cpu().numpy())
    return out  # (31,H,W), (16,H,W) の確率


def rgb_to_id(path, rgb_map, size):
    from PIL import Image
    a = np.asarray(Image.open(path).convert("RGB").resize(size, Image.NEAREST))
    out = np.zeros(a.shape[:2], np.uint8)
    for i, c in rgb_map.items():
        out[(a == np.array(c, np.uint8)).all(-1)] = i
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--clips", default="workspace/expS02_flf2v/outputs/clips")
    ap.add_argument("--out", default="workspace/expS02_flf2v/outputs/pseudo_ens")
    ap.add_argument("--members", nargs="+", default=list(MEMBERS))
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--img-h", type=int, default=576)
    ap.add_argument("--img-w", type=int, default=1024)
    ap.add_argument("--save-probs", action="store_true")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s",
                        handlers=[logging.StreamHandler(),
                                  logging.FileHandler(os.path.join(args.out, "pseudo_ens.log"))])

    lm = pd.read_csv(REPO / "data/labelmap.csv")
    fine_rgb = {int(r.fine_id): (int(r.fine_r), int(r.fine_g), int(r.fine_b))
                for r in lm.itertuples() if not pd.isna(r.fine_id)}
    id2rgb_fine, id2rgb_coarse = build_id2rgb(lm)
    folds = pd.read_csv(REPO / FOLDS_CSV)
    case_fold = {}
    for r in folds.itertuples():
        case = "_".join(r.filename[:-4].split("_")[:4])
        case_fold[case] = int(r.fold)

    from metrics.metrics import weighted_image_scores
    from metrics.classes import CLASSES, WEIGHT_TOTAL

    mean = torch.tensor(IMAGENET_MEAN, device=args.device).view(1, 3, 1, 1)
    std = torch.tensor(IMAGENET_STD, device=args.device).view(1, 3, 1, 1)

    dirs = sorted(d for d in glob.glob(os.path.join(args.clips, "*"))
                  if os.path.isdir(d) and os.path.exists(os.path.join(d, "f000.png")))
    log.info("clips=%d members=%s", len(dirs), args.members)

    rows, cache = [], {}
    for d in dirs:
        tag = os.path.basename(d)
        case = "_".join(tag.split("_")[:4])
        sa, sb = tag[len(case) + 1:].split("_to_")
        val_fold = case_fold.get(case)
        use_folds = [f for f in range(5) if f != val_fold]   # 端点を学習に含む fold のみ
        key = tuple(use_folds)
        if key not in cache:
            for k in list(cache):     # GPU を空けるため 1 セットだけ保持
                del cache[k]
            torch.cuda.empty_cache()
            cache[key] = [load_member(m, f, args.device) for f in use_folds for m in args.members]
            log.info("loaded %d models (folds %s, case %s val_fold=%s)",
                     len(cache[key]), use_folds, case, val_fold)
        models = cache[key]

        frames = sorted(glob.glob(os.path.join(d, "f[0-9][0-9][0-9].png")))
        h, w = cv2.imread(frames[0]).shape[:2]
        gt_a = rgb_to_id(REPO / f"data/masks_fine/{case}_{sa}.png", fine_rgb, (w, h))
        gt_b = rgb_to_id(REPO / f"data/masks_fine/{case}_{sb}.png", fine_rgb, (w, h))

        md = os.path.join(args.out, tag); os.makedirs(md, exist_ok=True)
        labs = []
        for i, p in enumerate(frames):
            pf, pc = predict(models, cv2.imread(p), args.img_h, args.img_w, (h, w),
                             args.device, mean, std)
            lab_f, lab_c = pf.argmax(0).astype(np.uint8), pc.argmax(0).astype(np.uint8)
            conf = pf.max(0)
            labs.append(lab_f)
            cv2.imwrite(f"{md}/ens_fine_f{i:03d}.png",
                        cv2.cvtColor(id2rgb_fine[lab_f], cv2.COLOR_RGB2BGR))
            cv2.imwrite(f"{md}/ens_coarse_f{i:03d}.png",
                        cv2.cvtColor(id2rgb_coarse[lab_c], cv2.COLOR_RGB2BGR))
            np.save(f"{md}/conf_f{i:03d}.npy", conf.astype(np.float16))

        # 診断
        pf_real_a, _ = predict(models, cv2.imread(os.path.join(d, "first.png")),
                               args.img_h, args.img_w, (h, w), args.device, mean, std)
        pf_real_b, _ = predict(models, cv2.imread(os.path.join(d, "last.png")),
                               args.img_h, args.img_w, (h, w), args.device, mean, std)
        real_a, real_b = pf_real_a.argmax(0).astype(np.uint8), pf_real_b.argmax(0).astype(np.uint8)
        gen_a, gen_b = labs[0], labs[-1]
        row = dict(
            clip=tag, n_frames=len(frames), val_fold=val_fold, n_models=len(models),
            shift_a=float((real_a == gen_a).mean()),          # (a) 実 vs 生成 の予測一致
            shift_b=float((real_b == gen_b).mean()),
            gen_vs_gt_a=weighted_image_scores(gen_a, gt_a, CLASSES, WEIGHT_TOTAL)["dice"],   # (b)
            gen_vs_gt_b=weighted_image_scores(gen_b, gt_b, CLASSES, WEIGHT_TOTAL)["dice"],
            real_vs_gt_a=weighted_image_scores(real_a, gt_a, CLASSES, WEIGHT_TOTAL)["dice"], # (c)
            real_vs_gt_b=weighted_image_scores(real_b, gt_b, CLASSES, WEIGHT_TOTAL)["dice"],
        )
        rows.append(row)
        log.info("[%s] domain shift a/b=%.3f/%.3f | ens on generated vs GT %.4f/%.4f | "
                 "ens on real vs GT %.4f/%.4f", tag, row["shift_a"], row["shift_b"],
                 row["gen_vs_gt_a"], row["gen_vs_gt_b"], row["real_vs_gt_a"], row["real_vs_gt_b"])

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(args.out, "diagnostics.csv"), index=False)
    log.info("\n%s", df.round(4).to_string(index=False))
    log.info("MEAN shift=%.3f | ens@generated vs GT=%.4f | ens@real vs GT=%.4f",
             df[["shift_a", "shift_b"]].mean().mean(),
             df[["gen_vs_gt_a", "gen_vs_gt_b"]].mean().mean(),
             df[["real_vs_gt_a", "real_vs_gt_b"]].mean().mean())
    json.dump(rows, open(os.path.join(args.out, "diagnostics.json"), "w"), indent=2)


if __name__ == "__main__":
    main()
