"""expA03: STIR の器具セグメンテーションモデルで Instrument クラスを上書きする後処理の ablation.

モデル: ../STIR/reference/stitch_track/weights/convnext-unet-best.pth
        (smp.Unet, encoder=tu-convnext_base.dinov3_lvd1689m, classes=1, sigmoid)
        前処理は stitch_track/tracker.py と同一: /255 → resize 512x512 → imagenet norm
        出力を元解像度に bilinear アップサンプル → >thr で binary tool mask

処理: expA00 の OOF 予測 PNG (task1/task2) の tool mask 画素を
      Instrument 色 (184,61,245) に置換 (fine=Instrument / coarse=Non_Anatomical_Other, 同色)
      → 公式コードで採点し、ベースラインと比較する。学習は行わない。

Usage:
    python3 override_eval.py --fold 0 [--thr 0.5]
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

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent))

STIR_CKPT = REPO.parent / "STIR" / "reference" / "stitch_track" / "weights" / "convnext-unet-best.pth"
SRC_OOF = REPO / "workspace" / "expA00_task12_baseline" / "results" / "expA00_task12_baseline" / "oof"
OUT_ROOT = REPO / "workspace" / "expA03_instr_delegate" / "results"
INSTRUMENT_RGB = np.array([184, 61, 245], dtype=np.uint8)
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
log = logging.getLogger("expA03")


def load_stir_model(device: str):
    import segmentation_models_pytorch as smp

    model = smp.Unet(
        encoder_name="tu-convnext_base.dinov3_lvd1689m",
        encoder_weights=None,
        in_channels=3,
        classes=1,
        activation="sigmoid",
    ).to(device)
    state = torch.load(STIR_CKPT, map_location=device, weights_only=False)
    if isinstance(state, dict) and "model_state_dict" in state:
        state = state["model_state_dict"]
    model.load_state_dict(state, strict=True)
    return model.eval()


@torch.no_grad()
def tool_mask(model, bgr: np.ndarray, device: str, thr: float) -> np.ndarray:
    h, w = bgr.shape[:2]
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    x = torch.from_numpy(rgb).permute(2, 0, 1)[None].float().to(device) / 255.0
    x = F.interpolate(x, size=(512, 512), mode="bilinear", align_corners=False)
    mean = torch.tensor(IMAGENET_MEAN, device=device).view(1, 3, 1, 1)
    std = torch.tensor(IMAGENET_STD, device=device).view(1, 3, 1, 1)
    x = (x - mean) / std
    with torch.autocast("cuda", dtype=torch.float16):
        prob = model(x.half())
    prob = F.interpolate(prob.float(), size=(h, w), mode="bilinear", align_corners=False)
    return (prob[0, 0].cpu().numpy() > thr)


def evaluate_dirs(pred_root: Path, out_json: Path) -> dict:
    sys.path.insert(0, str(REPO / "reference" / "tigersqai_challenge"))
    from fast_hd import patch_official_metrics
    patch_official_metrics()
    from metrics.classes import CLASSES, rgb_mask_to_label_mask as rgb_fine
    from metrics.classes_merged import CLASSES_MERGED, rgb_mask_to_label_mask as rgb_coarse
    from metrics.evaluate_seg import evaluate

    results = {}
    for task, gt_src, classes, fn in [
        ("task1", "masks_fine", CLASSES, rgb_fine),
        ("task2", "masks_coarse", CLASSES_MERGED, rgb_coarse),
    ]:
        pred_dir = pred_root / task
        names = sorted(p.name for p in pred_dir.glob("*.png"))
        gt_dir = pred_root / f"gt_{task}"
        gt_dir.mkdir(exist_ok=True)
        for old in gt_dir.glob("*.png"):
            old.unlink()
        for n in names:
            (gt_dir / n).symlink_to(REPO / "data" / gt_src / n)
        res = evaluate(gt_dir, pred_dir, classes=classes, rgb_to_label_fn=fn)
        results[task] = {"final_dice": res["final_dice"], "final_hd": res["final_hd"],
                         "n_images": len(names), "case_dice": res["case_dice"], "case_hd": res["case_hd"]}
        log.info("%s | Dice=%.4f HD=%.4f", task, res["final_dice"], res["final_hd"])
    out_json.write_text(json.dumps(results, indent=2))
    return results


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fold", type=int, default=0)
    ap.add_argument("--thr", type=float, default=0.5)
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()

    folds = pd.read_csv(REPO / "workspace" / "fold" / "v1" / "folds.csv")
    names = folds[folds.fold == args.fold].filename.tolist()

    out = OUT_ROOT / f"fold{args.fold}_thr{args.thr}"
    (out / "task1").mkdir(parents=True, exist_ok=True)
    (out / "task2").mkdir(parents=True, exist_ok=True)

    model = load_stir_model(args.device)
    n_px = 0
    for name in names:
        bgr = cv2.imread(str(REPO / "data" / "images" / name))
        m = tool_mask(model, bgr, args.device, args.thr)
        n_px += int(m.sum())
        for task in ("task1", "task2"):
            pred = cv2.imread(str(SRC_OOF / task / name))  # BGR
            pred[m] = INSTRUMENT_RGB[::-1]  # BGR で書く
            cv2.imwrite(str(out / task / name), pred)
    log.info("fold %d: %d images, overridden px total=%d", args.fold, len(names), n_px)

    results = evaluate_dirs(out, out / "metrics.json")
    log.info("baseline (expA00 fold0 公式): task1 Dice 0.5973/HD 0.3358, task2 Dice 0.5731/HD 0.3515")
    for t, r in results.items():
        log.info("override %s: Dice %.4f / HD %.4f", t, r["final_dice"], r["final_hd"])


if __name__ == "__main__":
    main()
