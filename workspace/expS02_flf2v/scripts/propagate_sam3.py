"""生成クリップ上で SAM3 tracker により GT マスクを伝播させ, 擬似ラベルの質を測る.

両端 (first / last) に実 GT があるので, 片端から伝播した結果を反対端の実 GT と
公式指標で比較すれば, 「生成 + 伝播」が使い物になるかを推測なしで判定できる.
対照として「伝播せず first の GT をそのまま last に置く」ベースラインも測る.
"""
import argparse, glob, json, logging, os
import numpy as np
import pandas as pd
import torch
from PIL import Image

# torchvision 0.16 (user site) には grayscale_to_rgb が無く transformers 5.x の
# video processor が落ちる. 共有 venv を書き換えると走行中の学習に影響するので,
# ここで等価な実装を注入して回避する.
import torchvision.transforms.v2.functional as _tvF
if not hasattr(_tvF, "grayscale_to_rgb"):
    def _grayscale_to_rgb(video):
        if video.shape[-3] == 1:
            return video.repeat_interleave(3, dim=-3)
        return video
    _tvF.grayscale_to_rgb = _grayscale_to_rgb

LOG = logging.getLogger(__name__)
MODEL_ID = "facebook/sam3"


def load_labelmap(path="data/labelmap.csv"):
    lm = pd.read_csv(path)
    fine_rgb, fine2merged, merged_rgb = {}, {}, {}
    for r in lm.itertuples():
        if not pd.isna(r.fine_id):
            fine_rgb[int(r.fine_id)] = (int(r.fine_r), int(r.fine_g), int(r.fine_b))
            fine2merged[int(r.fine_id)] = int(r.merged_id)
        if not pd.isna(r.merged_id) and int(r.merged_id) not in merged_rgb:
            merged_rgb[int(r.merged_id)] = (int(r.merged_r), int(r.merged_g), int(r.merged_b))
    return fine_rgb, fine2merged, merged_rgb


def rgb_to_id(path, rgb_map, size):
    a = Image.open(path).convert("RGB").resize(size, Image.NEAREST)
    a = np.asarray(a)
    out = np.zeros(a.shape[:2], np.uint8)
    for i, c in rgb_map.items():
        out[(a == np.array(c, np.uint8)).all(-1)] = i
    return out


def id_to_rgb(lab, rgb_map):
    out = np.zeros((*lab.shape, 3), np.uint8)
    for i, c in rgb_map.items():
        out[lab == i] = c
    return out


def compose(masks_by_obj, shape):
    """obj_id -> logit map を 1 枚のセマンティックラベルに合成 (logit 最大のクラスを採用)."""
    lab = np.zeros(shape, np.uint8)
    best = np.zeros(shape, np.float32)
    for oid, logit in masks_by_obj.items():
        hit = (logit > 0) & (logit > best)
        lab[hit] = oid
        best[hit] = logit[hit]
    return lab


def official_scores(pred_fine, gt_fine, f2m):
    from metrics.metrics import weighted_image_scores
    from metrics.classes import CLASSES, WEIGHT_TOTAL
    from metrics.classes_merged import CLASSES_MERGED, WEIGHT_TOTAL_MERGED
    to_m = np.zeros(256, np.uint8)
    for f, m in f2m.items():
        to_m[f] = m
    t2 = weighted_image_scores(pred_fine, gt_fine, CLASSES, WEIGHT_TOTAL)
    t1 = weighted_image_scores(to_m[pred_fine], to_m[gt_fine], CLASSES_MERGED, WEIGHT_TOTAL_MERGED)
    return dict(t1_dice=t1["dice"], t1_hd=t1["hd"], t2_dice=t2["dice"], t2_hd=t2["hd"])


def propagate(model, processor, frames, seed_lab, seed_idx, device, min_area):
    """seed_idx フレームの seed_lab を種に全フレームへ伝播し, フレームごとの合成ラベルを返す."""
    h, w = seed_lab.shape
    obj_ids, masks = [], []
    for cid in np.unique(seed_lab):
        if cid == 0:
            continue
        m = seed_lab == cid
        if m.mean() < min_area:
            continue
        obj_ids.append(int(cid)); masks.append(m)
    kept = list(obj_ids)  # processor が obj_ids を破壊的に扱うため控えを取る
    if not obj_ids:
        return None, []
    sess = processor.init_video_session(video=frames, inference_device=device, dtype=torch.float32)
    processor.add_inputs_to_inference_session(
        inference_session=sess, frame_idx=seed_idx, obj_ids=obj_ids,
        input_masks=[m.astype(np.uint8) for m in masks], original_size=(h, w))
    out = {}
    for o in model.propagate_in_video_iterator(sess, start_frame_idx=seed_idx):
        logits = processor.post_process_masks([o.pred_masks], original_sizes=[[h, w]],
                                              binarize=False)[0]
        logits = logits.float().cpu().numpy()
        by_obj = {oid: logits[k].squeeze() for k, oid in enumerate(sess.obj_ids)}
        out[o.frame_idx] = compose(by_obj, (h, w))
    if seed_idx > 0:  # 逆方向も回して全フレーム埋める
        for o in model.propagate_in_video_iterator(sess, start_frame_idx=seed_idx, reverse=True):
            logits = processor.post_process_masks([o.pred_masks], original_sizes=[[h, w]],
                                                  binarize=False)[0].float().cpu().numpy()
            by_obj = {oid: logits[k].squeeze() for k, oid in enumerate(sess.obj_ids)}
            out[o.frame_idx] = compose(by_obj, (h, w))
    del sess
    torch.cuda.empty_cache()
    return out, list(kept)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--clips", default="workspace/expS02_flf2v/outputs/clips")
    ap.add_argument("--out", default="workspace/expS02_flf2v/outputs/pseudo")
    ap.add_argument("--min-area", type=float, default=0.001)
    ap.add_argument("--save-masks", action="store_true")
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s",
                        handlers=[logging.StreamHandler(),
                                  logging.FileHandler(os.path.join(args.out, "propagate.log"))])

    fine_rgb, f2m, merged_rgb = load_labelmap()
    from transformers import Sam3TrackerVideoModel, Sam3TrackerVideoProcessor
    device = "cuda"
    LOG.info("loading %s", MODEL_ID)
    model = Sam3TrackerVideoModel.from_pretrained(MODEL_ID, dtype=torch.float32).to(device).eval()
    processor = Sam3TrackerVideoProcessor.from_pretrained(MODEL_ID)

    dirs = sorted(d for d in glob.glob(os.path.join(args.clips, "*"))
                  if os.path.isdir(d) and os.path.exists(os.path.join(d, "f000.png")))
    if args.limit:
        dirs = dirs[: args.limit]
    LOG.info("clips=%d", len(dirs))

    results = []
    for d in dirs:
        tag = os.path.basename(d)
        case = "_".join(tag.split("_")[:4])
        sa, sb = tag[len(case) + 1:].split("_to_")
        paths = sorted(glob.glob(os.path.join(d, "f[0-9][0-9][0-9].png")))  # first.png を拾わない
        frames = [np.asarray(Image.open(p).convert("RGB")) for p in paths]
        h, w = frames[0].shape[:2]
        gt_a = rgb_to_id(f"data/masks_fine/{case}_{sa}.png", fine_rgb, (w, h))
        gt_b = rgb_to_id(f"data/masks_fine/{case}_{sb}.png", fine_rgb, (w, h))
        n = len(frames)
        LOG.info("[%s] %d frames %dx%d  seed classes A=%d B=%d", tag, n, w, h,
                 len(np.unique(gt_a)) - 1, len(np.unique(gt_b)) - 1)

        with torch.inference_mode():
            fwd, ids_a = propagate(model, processor, frames, gt_a, 0, device, args.min_area)
            bwd, ids_b = propagate(model, processor, frames, gt_b, n - 1, device, args.min_area)
        if fwd is None or bwd is None:
            LOG.warning("[%s] no seed objects, skip", tag); continue

        seed_ok_f = float((fwd[0] == gt_a).mean())
        seed_ok_b = float((bwd[n - 1] == gt_b).mean())
        LOG.info("[%s] seed reproduction: fwd@0 vs GT_A=%.4f  bwd@%d vs GT_B=%.4f  (objs %s / %s)",
                 tag, seed_ok_f, n - 1, seed_ok_b, ids_a, ids_b)
        s_fwd = official_scores(fwd[n - 1], gt_b, f2m)        # 伝播した結果 vs 反対端の実 GT
        s_bwd = official_scores(bwd[0], gt_a, f2m)
        s_base_f = official_scores(gt_a, gt_b, f2m)           # 対照: 伝播せずコピー
        s_base_b = official_scores(gt_b, gt_a, f2m)
        cyc = [float((fwd[i] == bwd[i]).mean()) for i in range(n)]

        row = dict(clip=tag, n_frames=n, n_obj_a=len(ids_a), n_obj_b=len(ids_b),
                   fwd_t1_dice=s_fwd["t1_dice"], fwd_t2_dice=s_fwd["t2_dice"],
                   bwd_t1_dice=s_bwd["t1_dice"], bwd_t2_dice=s_bwd["t2_dice"],
                   base_t1_dice=(s_base_f["t1_dice"] + s_base_b["t1_dice"]) / 2,
                   base_t2_dice=(s_base_f["t2_dice"] + s_base_b["t2_dice"]) / 2,
                   fwd_t2_hd=s_fwd["t2_hd"], base_t2_hd=s_base_f["t2_hd"],
                   cycle_mean=float(np.mean(cyc)), cycle_min=float(np.min(cyc)),
                   seed_repro_fwd=seed_ok_f, seed_repro_bwd=seed_ok_b)
        results.append(row)
        LOG.info("[%s] propagated T2 Dice fwd=%.4f bwd=%.4f | copy-baseline %.4f | cycle %.3f",
                 tag, row["fwd_t2_dice"], row["bwd_t2_dice"], row["base_t2_dice"], row["cycle_mean"])

        if args.save_masks:
            md = os.path.join(args.out, tag); os.makedirs(md, exist_ok=True)
            for i in range(n):
                Image.fromarray(id_to_rgb(fwd[i], fine_rgb)).save(f"{md}/fwd_f{i:03d}.png")
                Image.fromarray(id_to_rgb(bwd[i], fine_rgb)).save(f"{md}/bwd_f{i:03d}.png")

    df = pd.DataFrame(results)
    df.to_csv(os.path.join(args.out, "scores.csv"), index=False)
    if len(df):
        LOG.info("\n%s", df.round(4).to_string(index=False))
        LOG.info("MEAN  propagated T2 Dice fwd=%.4f bwd=%.4f | copy-baseline=%.4f | cycle=%.3f",
                 df.fwd_t2_dice.mean(), df.bwd_t2_dice.mean(), df.base_t2_dice.mean(),
                 df.cycle_mean.mean())
    json.dump(results, open(os.path.join(args.out, "scores.json"), "w"), indent=2)


if __name__ == "__main__":
    main()
