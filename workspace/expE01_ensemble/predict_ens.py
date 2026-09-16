"""expE01: 複数レシピ × 5fold の softmax 平均アンサンブルで OOF を作り、公式評価する.

背景: 単一 5-fold 走行の公式 case 平均 Dice は run 間分散が ±0.012 程度あり
(config 完全同一の expA06 vs expA11 が +0.0120, p=0.0005 という帰無比較で判明)、
個別レシピの微差 (0.002〜0.014) は検出できない。異なる loss 構成のモデルを混ぜて
誤差を非相関化し、run 間分散そのものも平均で潰す。

- 全メンバーが同一アーキ (DualHeadUnetPP / MaxViT-Base tf_512 + dual Unet++)
- fold N の val ケースは全メンバーの fold N モデルが未学習 (A06 は v1 学習だが、
  v2 の追加 2 枚は v1 に存在しないため未見。case→fold 割当は v1/v2 で完全一致)
- 1024x576 で forward → softmax → メンバー平均 → 元解像度へ bilinear → argmax

Usage: python3 predict_ens.py --folds 0 1 --device cuda:0 [--members A06 A09 A10]
       python3 predict_ens.py --eval-only
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import yaml

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent))

from dataset import IMAGENET_MEAN, IMAGENET_STD  # noqa: E402
from model import DualHeadUnetPP  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
log = logging.getLogger("ens")

# メンバー: name -> (config パス, results 配下の experiment 名)
MEMBERS = {
    "A06": ("expA06_f2c_loss", "expA06_f2c_loss"),
    "A09": ("expA09_lymph_aux", "expA09_lymph_aux"),
    "A10": ("expA10_toolmask", "expA10_toolmask"),
    "A11": ("expA11_v2data", "expA11_v2data"),
    "A05": ("expA05_maxvit", "expA05_maxvit"),   # 同一アーキ (f2c loss なし)
}
FOLDS_CSV = "workspace/fold/v2/folds.csv"   # 526 枚。case→fold は v1 と同一
OUT_ROOT = REPO / "workspace/expE01_ensemble/results"


def load_member(member: str, fold: int, device: str) -> DualHeadUnetPP:
    exp_dir, exp_name = MEMBERS[member]
    cfg = yaml.safe_load((REPO / "workspace" / exp_dir / "config.yaml").read_text())
    m = cfg["model"]
    model = DualHeadUnetPP(
        encoder_name=m["encoder_name"], encoder_weights=None,
        decoder_channels=tuple(m["decoder_channels"]),
        num_classes_fine=m["num_classes_fine"], num_classes_coarse=m["num_classes_coarse"],
        img_size=(cfg["data"]["img_h"], cfg["data"]["img_w"]),
    )
    ckpt = REPO / cfg["paths"]["results_root"] / exp_name / f"fold{fold}" / "best.ckpt"
    assert ckpt.exists(), f"missing {ckpt}"
    state = torch.load(ckpt, map_location="cpu", weights_only=False)["state_dict"]
    # "model." 配下のみ = 共通アーキ。A09 の lymph_head 等の補助枝はここで落ちる
    state = {k.removeprefix("model."): v for k, v in state.items() if k.startswith("model.")}
    model.load_state_dict(state, strict=True)
    return model.to(device).eval()


def build_id2rgb(labelmap: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    fine = np.zeros((256, 3), dtype=np.uint8)
    coarse = np.zeros((256, 3), dtype=np.uint8)
    for _, r in labelmap.iterrows():
        fine[int(r.fine_id)] = (r.fine_r, r.fine_g, r.fine_b)
        coarse[int(r.merged_id)] = (r.merged_r, r.merged_g, r.merged_b)
    return fine, coarse


@torch.no_grad()
def predict_fold(members: list[str], fold: int, out1: Path, out2: Path, device: str,
                 img_h: int, img_w: int, limit: int | None = None) -> None:
    models = [load_member(m, fold, device) for m in members]
    folds = pd.read_csv(REPO / FOLDS_CSV)
    val_df = folds[folds.fold == fold]
    if limit is not None:
        val_df = val_df.head(limit)
    id2rgb_fine, id2rgb_coarse = build_id2rgb(pd.read_csv(REPO / "data/labelmap.csv"))
    mean = torch.tensor(IMAGENET_MEAN, device=device).view(1, 3, 1, 1)
    std = torch.tensor(IMAGENET_STD, device=device).view(1, 3, 1, 1)

    for _, row in val_df.iterrows():
        bgr = cv2.imread(str(REPO / "data/images" / row.filename))
        oh, ow = bgr.shape[:2]
        rgb = cv2.cvtColor(cv2.resize(bgr, (img_w, img_h), interpolation=cv2.INTER_AREA),
                           cv2.COLOR_BGR2RGB)
        x = torch.from_numpy(rgb).permute(2, 0, 1)[None].float().to(device) / 255.0
        x = (x - mean) / std
        pf = pc = None
        for model in models:
            with torch.autocast("cuda", dtype=torch.float16):
                lf, lc = model(x)
            sf, sc = lf.float().softmax(1), lc.float().softmax(1)
            pf = sf if pf is None else pf + sf
            pc = sc if pc is None else pc + sc
        for probs, id2rgb, out_dir in [(pf, id2rgb_fine, out1), (pc, id2rgb_coarse, out2)]:
            up = F.interpolate(probs / len(models), size=(oh, ow), mode="bilinear",
                               align_corners=False)
            ids = up.argmax(1)[0].cpu().numpy().astype(np.uint8)
            cv2.imwrite(str(out_dir / row.filename), cv2.cvtColor(id2rgb[ids], cv2.COLOR_RGB2BGR))
    log.info("fold %d: predicted %d images (%d models)", fold, len(val_df), len(models))
    del models
    torch.cuda.empty_cache()


def evaluate_oof(out1: Path, out2: Path, oof_dir: Path) -> None:
    sys.path.insert(0, str(REPO / "reference" / "tigersqai_challenge"))
    from metrics.classes import CLASSES, rgb_mask_to_label_mask as rgb2id_fine
    from metrics.classes_merged import CLASSES_MERGED, rgb_mask_to_label_mask as rgb2id_coarse
    from metrics.evaluate_seg import evaluate

    from fast_hd import patch_official_metrics
    patch_official_metrics()
    log.info("HD implementation patched to EDT-based equivalent (verified against official)")

    results = {}
    for task, pred_dir, gt_src, classes, rgb_fn in [
        ("task1", out1, "masks_fine", CLASSES, rgb2id_fine),
        ("task2", out2, "masks_coarse", CLASSES_MERGED, rgb2id_coarse),
    ]:
        pred_names = sorted(p.name for p in pred_dir.glob("*.png"))
        if not pred_names:
            log.warning("%s: no predictions, skip", task)
            continue
        gt_dir = oof_dir / f"gt_{task}"
        gt_dir.mkdir(exist_ok=True)
        for old in gt_dir.glob("*.png"):
            old.unlink()
        for n in pred_names:
            (gt_dir / n).symlink_to(REPO / "data" / gt_src / n)
        log.info("%s: evaluating %d images (official code)...", task, len(pred_names))
        res = evaluate(gt_dir, pred_dir, classes=classes, rgb_to_label_fn=rgb_fn)
        results[task] = {
            "final_dice": res["final_dice"], "final_hd": res["final_hd"],
            "n_images": len(pred_names), "n_cases": len(res["case_ids"]),
            "case_dice": res["case_dice"], "case_hd": res["case_hd"],
        }
        log.info("%s | Dice=%.4f HD=%.4f (%d cases)", task, res["final_dice"], res["final_hd"],
                 len(res["case_ids"]))
    (oof_dir / "oof_metrics.json").write_text(json.dumps(results, indent=2))
    log.info("saved %s", oof_dir / "oof_metrics.json")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--members", nargs="+", default=["A06", "A09", "A10"])
    ap.add_argument("--folds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--tag", default=None, help="出力名 (既定: メンバー連結)")
    ap.add_argument("--eval-only", action="store_true")
    ap.add_argument("--no-eval", action="store_true")
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    for m in args.members:
        assert m in MEMBERS, f"unknown member {m}"
    tag = args.tag or "ens_" + "".join(args.members)
    oof_dir = OUT_ROOT / tag
    out1, out2 = oof_dir / "task1", oof_dir / "task2"
    out1.mkdir(parents=True, exist_ok=True)
    out2.mkdir(parents=True, exist_ok=True)
    log.info("members=%s folds=%s -> %s", args.members, args.folds, oof_dir)

    cfg0 = yaml.safe_load((REPO / "workspace" / MEMBERS[args.members[0]][0] / "config.yaml").read_text())
    if not args.eval_only:
        for fold in args.folds:
            predict_fold(args.members, fold, out1, out2, args.device,
                         cfg0["data"]["img_h"], cfg0["data"]["img_w"], limit=args.limit)
    if not args.no_eval:
        evaluate_oof(out1, out2, oof_dir)


if __name__ == "__main__":
    main()
