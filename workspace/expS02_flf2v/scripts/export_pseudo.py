"""擬似ラベル付きクリップを学習キャッシュ形式 (1024x576) に書き出す.

出力は workspace/data_proc/*_pseudo_1024/ と pseudo_index.csv.
ラベルは背景 0..30 の ID マップで, 採用できなかった画素は 255 (ignore).

重要: 擬似ラベルは「その case を学習に含む ens5」と「その case の実 GT」から作られている.
      したがって case X 由来の擬似フレームは, X が val になる fold の学習に入れてはならない.
      pseudo_index.csv の case_id 列でそれを担保する.
"""
from __future__ import annotations
import argparse, glob, json, logging, os
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from PIL import Image

LOG = logging.getLogger(__name__)
REPO = Path(__file__).resolve().parents[3]
IGNORE = 255


def luts():
    lm = pd.read_csv(REPO / "data/labelmap.csv")
    rgb2fine = {}
    f2m = np.full(256, IGNORE, np.uint8)
    for r in lm.itertuples():
        if pd.isna(r.fine_id):
            continue
        rgb2fine[(int(r.fine_r), int(r.fine_g), int(r.fine_b))] = int(r.fine_id)
        f2m[int(r.fine_id)] = int(r.merged_id)
    return rgb2fine, f2m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--clips", default="workspace/expS02_flf2v/outputs/clips")
    ap.add_argument("--pseudo", default="workspace/expS02_flf2v/outputs/pseudo_combined")
    ap.add_argument("--out", default="workspace/data_proc")
    ap.add_argument("--img-h", type=int, default=576)
    ap.add_argument("--img-w", type=int, default=1024)
    ap.add_argument("--stride", type=int, default=3, help="何フレームおきに採用するか")
    ap.add_argument("--skip-ends", type=int, default=1,
                    help="両端 N フレームは実画像とほぼ同一なので除外する")
    ap.add_argument("--min-coverage", type=float, default=0.70,
                    help="有効画素率がこれ未満のフレームは捨てる")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")

    rgb2fine, f2m = luts()
    img_dir = REPO / args.out / "images_pseudo_1024"
    fine_dir = REPO / args.out / "labels_fine_pseudo_1024"
    coarse_dir = REPO / args.out / "labels_coarse_pseudo_1024"
    for d in (img_dir, fine_dir, coarse_dir):
        d.mkdir(parents=True, exist_ok=True)

    rows, n_clip, n_drop = [], 0, 0
    for md in sorted(glob.glob(os.path.join(args.pseudo, "*"))):
        if not os.path.exists(os.path.join(md, "DONE")):
            continue
        tag = os.path.basename(md)
        case = "_".join(tag.split("_")[:4])
        cd = os.path.join(args.clips, tag)
        n = len(glob.glob(os.path.join(md, "pseudo_f*.png")))
        if n == 0:
            continue
        n_clip += 1
        stats = {}
        sp = os.path.join(md, "stats.json")
        if os.path.exists(sp):
            stats = json.load(open(sp))
        for i in range(args.skip_ends, n - args.skip_ends, args.stride):
            valid = np.asarray(Image.open(f"{md}/valid_f{i:03d}.png")) > 0
            cov = float(valid.mean())
            if cov < args.min_coverage:
                n_drop += 1
                continue
            name = f"{tag}_f{i:03d}.png"
            bgr = cv2.imread(f"{cd}/f{i:03d}.png")
            if bgr is None:
                continue
            cv2.imwrite(str(img_dir / name),
                        cv2.resize(bgr, (args.img_w, args.img_h), interpolation=cv2.INTER_CUBIC))

            rgb = np.asarray(Image.open(f"{md}/pseudo_f{i:03d}.png").convert("RGB"))
            fine = np.zeros(rgb.shape[:2], np.uint8)
            for c, fid in rgb2fine.items():
                fine[(rgb == np.array(c, np.uint8)).all(-1)] = fid
            fine[~valid] = IGNORE
            fine = cv2.resize(fine, (args.img_w, args.img_h), interpolation=cv2.INTER_NEAREST)
            coarse = f2m[fine]
            coarse[fine == IGNORE] = IGNORE
            cv2.imwrite(str(fine_dir / name), fine)
            cv2.imwrite(str(coarse_dir / name), coarse)

            rows.append(dict(filename=name, case_id=case, clip=tag, frame=i,
                             coverage=round(cov, 4),
                             clip_cycle=round(stats.get("cycle", float("nan")), 4),
                             clip_fwd_dice=round(stats.get("fwd_dice", float("nan")), 4),
                             clip_base_dice=round(stats.get("base_dice", float("nan")), 4)))

    df = pd.DataFrame(rows)
    idx = REPO / args.out / "pseudo_index.csv"
    df.to_csv(idx, index=False)
    LOG.info("clips=%d  exported frames=%d  dropped(low coverage)=%d", n_clip, len(df), n_drop)
    if len(df):
        LOG.info("cases=%d  coverage mean=%.3f min=%.3f", df.case_id.nunique(),
                 df.coverage.mean(), df.coverage.min())
        LOG.info("1 case あたりの擬似フレーム数: median=%.0f min=%d max=%d",
                 df.groupby("case_id").size().median(), df.groupby("case_id").size().min(),
                 df.groupby("case_id").size().max())
    LOG.info("wrote %s", idx)


if __name__ == "__main__":
    main()
