"""生成済みクリップに対して SAM3 伝播 + ens5 予測 + 統合を一気に行うワーカー.

モデルのロードが重いので 1 プロセスで全部持ち, クリップを順に処理する.
生成ワーカーと並走させる前提で, 未完成のクリップは飛ばして次のループで拾う.
"""
from __future__ import annotations
import argparse, glob, json, logging, os, sys, time
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from PIL import Image

import torchvision.transforms.v2.functional as _tvF
if not hasattr(_tvF, "grayscale_to_rgb"):     # torchvision 0.16 対策 (共有 venv は触らない)
    def _g2r(v):
        return v.repeat_interleave(3, dim=-3) if v.shape[-3] == 1 else v
    _tvF.grayscale_to_rgb = _g2r

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "workspace/expE01_ensemble"))
sys.path.insert(0, str(REPO / "reference/tigersqai_challenge"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from dataset import IMAGENET_MEAN, IMAGENET_STD          # noqa: E402
from predict_ens import MEMBERS, FOLDS_CSV, load_member, build_id2rgb   # noqa: E402
from propagate_sam3 import propagate, compose            # noqa: E402

log = logging.getLogger("stageB")
IGNORE = 255


def load_luts():
    lm = pd.read_csv(REPO / "data/labelmap.csv")
    fine_rgb = {int(r.fine_id): (int(r.fine_r), int(r.fine_g), int(r.fine_b))
                for r in lm.itertuples() if not pd.isna(r.fine_id)}
    f2m = {int(r.fine_id): int(r.merged_id) for r in lm.itertuples() if not pd.isna(r.fine_id)}
    return lm, fine_rgb, f2m


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


@torch.no_grad()
def ens_predict(models, bgr, img_h, img_w, out_hw, device, mean, std):
    rgb = cv2.cvtColor(cv2.resize(bgr, (img_w, img_h), interpolation=cv2.INTER_AREA),
                       cv2.COLOR_BGR2RGB)
    x = torch.from_numpy(rgb).permute(2, 0, 1)[None].float().to(device) / 255.0
    x = ((x - mean) / std).half()      # 重みが fp16 なので入力も揃える
    pf = None
    for m in models:
        lf, _ = m(x)
        sf = lf.float().softmax(1)
        pf = sf if pf is None else pf + sf
    up = F.interpolate(pf / len(models), size=out_hw, mode="bilinear", align_corners=False)
    p = up[0].cpu().numpy()
    return p.argmax(0).astype(np.uint8), p.max(0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--clips", default="workspace/expS02_flf2v/outputs/clips")
    ap.add_argument("--out", default="workspace/expS02_flf2v/outputs/pseudo_combined")
    ap.add_argument("--queue", default="workspace/expS02_flf2v/outputs/queue.csv")
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--num-shards", type=int, default=1)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--img-h", type=int, default=576)
    ap.add_argument("--img-w", type=int, default=1024)
    ap.add_argument("--min-area", type=float, default=0.001)
    ap.add_argument("--watch", type=int, default=0, help="秒. >0 なら未完成クリップを待って繰り返す")
    ap.add_argument("--save-debug", action="store_true")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s",
                        handlers=[logging.StreamHandler(),
                                  logging.FileHandler(f"{args.out}/stage_b_{args.shard}.log")])

    lm, fine_rgb, f2m = load_luts()
    id2rgb_fine, _ = build_id2rgb(lm)
    folds = pd.read_csv(REPO / FOLDS_CSV)
    case_fold = {"_".join(r.filename[:-4].split("_")[:4]): int(r.fold) for r in folds.itertuples()}

    from transformers import Sam3TrackerVideoModel, Sam3TrackerVideoProcessor
    log.info("loading SAM3")
    sam = Sam3TrackerVideoModel.from_pretrained("facebook/sam3", dtype=torch.float32).to(args.device).eval()
    sam_proc = Sam3TrackerVideoProcessor.from_pretrained("facebook/sam3")
    mean = torch.tensor(IMAGENET_MEAN, device=args.device).view(1, 3, 1, 1)
    std = torch.tensor(IMAGENET_STD, device=args.device).view(1, 3, 1, 1)

    q = pd.read_csv(args.queue)
    tags = [f'{r.case}_{r.station_a}_to_{r.station_b}' for r in q.itertuples()]
    tags = tags[args.shard :: args.num_shards]
    # キューは case ラウンドロビン順なので, そのまま処理すると case ごとに fold が変わり
    # 20 本の seg モデルを毎クリップ読み直すことになる. fold でまとめて並べ替える.
    tags.sort(key=lambda t: case_fold.get("_".join(t.split("_")[:4]), 99))
    log.info("shard %d/%d -> %d clips", args.shard, args.num_shards, len(tags))

    seg_cache, done, rows = {}, set(), []
    while True:
        progressed = False
        for tag in tags:
            if tag in done:
                continue
            d = os.path.join(args.clips, tag)
            md = os.path.join(args.out, tag)
            if os.path.exists(os.path.join(md, "DONE")):
                done.add(tag); continue
            frames = sorted(glob.glob(os.path.join(d, "f[0-9][0-9][0-9].png")))
            if not frames or not os.path.exists(os.path.join(d, "meta.txt")):
                continue                                   # まだ生成中
            progressed = True
            t0 = time.time()
            try:
                case = "_".join(tag.split("_")[:4])
                sa, sb = tag[len(case) + 1:].split("_to_")
                imgs = [np.asarray(Image.open(p).convert("RGB")) for p in frames]
                h, w = imgs[0].shape[:2]
                n = len(imgs)
                gt_a = rgb_to_id(REPO / f"data/masks_fine/{case}_{sa}.png", fine_rgb, (w, h))
                gt_b = rgb_to_id(REPO / f"data/masks_fine/{case}_{sb}.png", fine_rgb, (w, h))

                with torch.inference_mode():
                    fwd, ids_a = propagate(sam, sam_proc, imgs, gt_a, 0, args.device, args.min_area)
                    bwd, ids_b = propagate(sam, sam_proc, imgs, gt_b, n - 1, args.device, args.min_area)
                if fwd is None or bwd is None:
                    log.warning("[%s] no seed objects, skip", tag); done.add(tag); continue

                val_fold = case_fold.get(case)
                key = tuple(f for f in range(5) if f != val_fold)
                if key not in seg_cache:
                    models = None          # 直前の 20 本への参照を切らないと解放されず OOM する
                    for k in list(seg_cache):
                        del seg_cache[k]
                    torch.cuda.empty_cache()
                    seg_cache[key] = [load_member(m, f, args.device).half()
                                  for f in key for m in MEMBERS]
                    log.info("loaded %d seg models (folds %s)", len(seg_cache[key]), key)
                models = seg_cache[key]

                os.makedirs(md, exist_ok=True)
                cov = []
                for i, p in enumerate(frames):
                    E, conf = ens_predict(models, cv2.imread(p), args.img_h, args.img_w,
                                          (h, w), args.device, mean, std)
                    ok = (E == fwd[i]) & (E == bwd[i])
                    lab = np.full(E.shape, IGNORE, np.uint8)
                    lab[ok] = E[ok]
                    Image.fromarray(id_to_rgb(np.where(lab == IGNORE, 0, lab), fine_rgb)).save(
                        f"{md}/pseudo_f{i:03d}.png")
                    Image.fromarray(((lab != IGNORE) * 255).astype(np.uint8)).save(
                        f"{md}/valid_f{i:03d}.png")
                    cov.append(float(ok.mean()))
                    if args.save_debug:
                        Image.fromarray(id_to_rgb(fwd[i], fine_rgb)).save(f"{md}/fwd_f{i:03d}.png")
                        Image.fromarray(id_to_rgb(bwd[i], fine_rgb)).save(f"{md}/bwd_f{i:03d}.png")
                        Image.fromarray(id_to_rgb(E, fine_rgb)).save(f"{md}/ens_f{i:03d}.png")

                from metrics.metrics import weighted_image_scores
                from metrics.classes import CLASSES, WEIGHT_TOTAL
                row = dict(clip=tag, case=case, station_a=sa, station_b=sb, n_frames=n,
                           n_obj_a=len(ids_a), n_obj_b=len(ids_b),
                           fwd_dice=weighted_image_scores(fwd[n - 1], gt_b, CLASSES, WEIGHT_TOTAL)["dice"],
                           bwd_dice=weighted_image_scores(bwd[0], gt_a, CLASSES, WEIGHT_TOTAL)["dice"],
                           base_dice=weighted_image_scores(gt_a, gt_b, CLASSES, WEIGHT_TOTAL)["dice"],
                           cycle=float(np.mean([(fwd[i] == bwd[i]).mean() for i in range(n)])),
                           coverage=float(np.mean(cov)), coverage_mid=cov[n // 2],
                           secs=round(time.time() - t0, 1))
                rows.append(row)
                json.dump(row, open(f"{md}/stats.json", "w"), indent=2)
                open(f"{md}/DONE", "w").write("ok\n")
                done.add(tag)
                log.info("[%s] cov=%.3f fwd=%.3f bwd=%.3f base=%.3f cycle=%.3f (%.0fs) %d/%d",
                         tag, row["coverage"], row["fwd_dice"], row["bwd_dice"], row["base_dice"],
                         row["cycle"], row["secs"], len(done), len(tags))
                pd.DataFrame(rows).to_csv(f"{args.out}/stats_shard{args.shard}.csv", index=False)
            except torch.cuda.OutOfMemoryError as e:
                log.error("[%s] CUDA OOM, skip: %s", tag, e)
                models = None
                seg_cache.clear()
                torch.cuda.empty_cache()
            except Exception:
                log.exception("[%s] 失敗, skip", tag)
                done.add(tag)

        if len(done) >= len(tags) or not args.watch:
            break
        if not progressed:
            time.sleep(args.watch)
    log.info("shard %d finished: %d clips", args.shard, len(done))


if __name__ == "__main__":
    main()
