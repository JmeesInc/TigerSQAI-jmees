"""ens5 予測と SAM3 双方向伝播を突き合わせて擬似ラベルを作る.

役割分担:
  ens5   … 31 クラスすべてを出せる (途中で現れる構造も扱える) が実 GT に固定されていない
  SAM3   … 両端の実 GT に幾何が固定されるが, 種フレームに無いクラスは出せない
両者が一致した画素だけを擬似ラベルとして採用し, 残りは ignore(255) に落とす.
不一致は生成側の幻覚か追跡の失敗のどちらかなので, そのまま品質フィルタになる.
"""
from __future__ import annotations
import argparse, glob, json, logging, os, sys
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "reference/tigersqai_challenge"))
log = logging.getLogger("combine")
IGNORE = 255


def load_luts():
    lm = pd.read_csv(REPO / "data/labelmap.csv")
    fine_rgb, f2m = {}, {}
    for r in lm.itertuples():
        if not pd.isna(r.fine_id):
            fine_rgb[int(r.fine_id)] = (int(r.fine_r), int(r.fine_g), int(r.fine_b))
            f2m[int(r.fine_id)] = int(r.merged_id)
    return fine_rgb, f2m


def rgb_to_id(path, rgb_map, size=None):
    im = Image.open(path).convert("RGB")
    if size:
        im = im.resize(size, Image.NEAREST)
    a = np.asarray(im)
    out = np.zeros(a.shape[:2], np.uint8)
    for i, c in rgb_map.items():
        out[(a == np.array(c, np.uint8)).all(-1)] = i
    return out


def id_to_rgb(lab, rgb_map):
    out = np.zeros((*lab.shape, 3), np.uint8)
    for i, c in rgb_map.items():
        out[lab == i] = c
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--clips", default="workspace/expS02_flf2v/outputs/clips")
    ap.add_argument("--sam", default="workspace/expS02_flf2v/outputs/pseudo")
    ap.add_argument("--ens", default="workspace/expS02_flf2v/outputs/pseudo_ens")
    ap.add_argument("--out", default="workspace/expS02_flf2v/outputs/pseudo_combined")
    ap.add_argument("--min-conf", type=float, default=0.0,
                    help="ens の最大確率がこれ未満の画素も ignore にする")
    ap.add_argument("--trust-sam-only", action="store_true",
                    help="fwd==bwd だが ens と不一致の画素も SAM3 側で採用する")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s",
                        handlers=[logging.StreamHandler(),
                                  logging.FileHandler(os.path.join(args.out, "combine.log"))])
    fine_rgb, f2m = load_luts()
    from metrics.metrics import weighted_image_scores
    from metrics.classes import CLASSES, WEIGHT_TOTAL

    rows, per_frame = [], []
    for d in sorted(glob.glob(os.path.join(args.clips, "*"))):
        tag = os.path.basename(d)
        sam_d, ens_d = os.path.join(args.sam, tag), os.path.join(args.ens, tag)
        if not (os.path.isdir(sam_d) and os.path.isdir(ens_d)):
            continue
        case = "_".join(tag.split("_")[:4])
        sa, sb = tag[len(case) + 1:].split("_to_")
        n = len(glob.glob(os.path.join(ens_d, "ens_fine_f*.png")))
        md = os.path.join(args.out, tag); os.makedirs(md, exist_ok=True)

        cov, agree_sam = [], []
        for i in range(n):
            E = rgb_to_id(f"{ens_d}/ens_fine_f{i:03d}.png", fine_rgb)
            F = rgb_to_id(f"{sam_d}/fwd_f{i:03d}.png", fine_rgb)
            B = rgb_to_id(f"{sam_d}/bwd_f{i:03d}.png", fine_rgb)
            ok = (E == F) & (E == B)
            lab = np.full(E.shape, IGNORE, np.uint8)
            lab[ok] = E[ok]
            if args.trust_sam_only:
                sam_ok = (F == B) & ~ok
                lab[sam_ok] = F[sam_ok]
            if args.min_conf > 0:
                cf = np.load(f"{ens_d}/conf_f{i:03d}.npy").astype(np.float32)
                lab[cf < args.min_conf] = IGNORE
            Image.fromarray(id_to_rgb(np.where(lab == IGNORE, 0, lab), fine_rgb)).save(
                f"{md}/pseudo_f{i:03d}.png")
            Image.fromarray(((lab != IGNORE) * 255).astype(np.uint8)).save(
                f"{md}/valid_f{i:03d}.png")
            cov.append(float((lab != IGNORE).mean()))
            agree_sam.append(float((F == B).mean()))
            per_frame.append(dict(clip=tag, frame=i, coverage=cov[-1], sam_cycle=agree_sam[-1]))

        # 端点で実 GT と比べ, 統合ルール自体の妥当性を確認する
        h, w = E.shape
        ends = {}
        for idx, st in ((0, sa), (n - 1, sb)):
            gt = rgb_to_id(REPO / f"data/masks_fine/{case}_{st}.png", fine_rgb, (w, h))
            lab = rgb_to_id(f"{md}/pseudo_f{idx:03d}.png", fine_rgb)
            valid = np.asarray(Image.open(f"{md}/valid_f{idx:03d}.png")) > 0
            pred = np.where(valid, lab, gt)   # ignore 画素は採点対象外 = GT で埋める
            ends[st] = weighted_image_scores(pred, gt, CLASSES, WEIGHT_TOTAL)["dice"]

        rows.append(dict(clip=tag, n_frames=n, coverage_mean=float(np.mean(cov)),
                         coverage_first=cov[0], coverage_mid=cov[n // 2], coverage_last=cov[-1],
                         sam_cycle_mean=float(np.mean(agree_sam)),
                         end_dice_a=ends[sa], end_dice_b=ends[sb]))
        log.info("[%s] coverage mean=%.3f (first %.3f / mid %.3f / last %.3f) | "
                 "SAM cycle=%.3f | end Dice %.4f/%.4f", tag, rows[-1]["coverage_mean"],
                 cov[0], cov[n // 2], cov[-1], rows[-1]["sam_cycle_mean"],
                 ends[sa], ends[sb])

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(args.out, "summary.csv"), index=False)
    pd.DataFrame(per_frame).to_csv(os.path.join(args.out, "per_frame.csv"), index=False)
    if len(df):
        log.info("\n%s", df.round(4).to_string(index=False))
        log.info("MEAN coverage=%.3f  SAM cycle=%.3f  end Dice=%.4f",
                 df.coverage_mean.mean(), df.sam_cycle_mean.mean(),
                 df[["end_dice_a", "end_dice_b"]].mean().mean())


if __name__ == "__main__":
    main()
