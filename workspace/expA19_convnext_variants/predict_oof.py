"""OOF 推論 + 公式評価.

各 fold の best.ckpt で val ケースを元解像度で推論し、提出と同じ RGB PNG を出力。
全 fold 分が揃った pred ディレクトリを公式 metrics.evaluate_seg.evaluate で採点する。

Usage:
    python3 predict_oof.py                 # 全 fold 推論 + 評価
    python3 predict_oof.py --folds 0       # fold0 のみ推論 + (揃っている分だけ) 評価
    python3 predict_oof.py --eval-only     # 推論スキップ、評価のみ

出力: {results_root}/{name}/oof/task1/*.png, task2/*.png, oof_metrics.json
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
log = logging.getLogger("oof")


def load_model(cfg: dict, ckpt_path: Path, device: str) -> DualHeadUnetPP:
    m = cfg["model"]
    # 重みは ckpt から入るので事前学習は読まない (cholec_fold=None)
    model = DualHeadUnetPP(
        encoder_name=m["encoder_name"],
        encoder_weights=None,
        decoder_channels=tuple(m["decoder_channels"]),
        num_classes_fine=m["num_classes_fine"],
        num_classes_coarse=m["num_classes_coarse"],
        img_size=(cfg["data"]["img_h"], cfg["data"]["img_w"]),
    )
    state = torch.load(ckpt_path, map_location="cpu", weights_only=False)["state_dict"]
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
def predict_fold(cfg: dict, fold: int, out1: Path, out2: Path, device: str,
                 limit: int | None = None) -> None:
    name = cfg["experiment"]["name"]
    fold_dir = REPO / cfg["paths"]["results_root"] / name / f"fold{fold}"
    ckpt = fold_dir / "best.ckpt"
    assert ckpt.exists(), f"missing {ckpt}"
    model = load_model(cfg, ckpt, device)

    folds = pd.read_csv(REPO / cfg["cv"]["folds_csv"])
    val_df = folds[folds.fold == fold]
    if limit is not None:
        val_df = val_df.head(limit)
    labelmap = pd.read_csv(REPO / cfg["paths"]["labelmap_csv"])
    id2rgb_fine, id2rgb_coarse = build_id2rgb(labelmap)

    h, w = cfg["data"]["img_h"], cfg["data"]["img_w"]
    mean = torch.tensor(IMAGENET_MEAN, device=device).view(1, 3, 1, 1)
    std = torch.tensor(IMAGENET_STD, device=device).view(1, 3, 1, 1)

    for _, row in val_df.iterrows():
        bgr = cv2.imread(str(REPO / cfg["paths"]["orig_images_dir"] / row.filename))
        oh, ow = bgr.shape[:2]
        rgb = cv2.cvtColor(cv2.resize(bgr, (w, h), interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2RGB)
        x = torch.from_numpy(rgb).permute(2, 0, 1)[None].float().to(device) / 255.0
        x = (x - mean) / std
        with torch.autocast("cuda", dtype=torch.float16):
            lf, lc = model(x)
        for logits, id2rgb, out_dir in [(lf, id2rgb_fine, out1), (lc, id2rgb_coarse, out2)]:
            up = F.interpolate(logits.float(), size=(oh, ow), mode="bilinear", align_corners=False)
            ids = up.argmax(1)[0].cpu().numpy().astype(np.uint8)
            cv2.imwrite(str(out_dir / row.filename), cv2.cvtColor(id2rgb[ids], cv2.COLOR_RGB2BGR))
    log.info("fold %d: predicted %d images", fold, len(val_df))
    del model
    torch.cuda.empty_cache()


def evaluate_oof(cfg: dict, out1: Path, out2: Path, oof_dir: Path) -> None:
    sys.path.insert(0, str(REPO / "reference" / "tigersqai_challenge"))
    from metrics.classes import CLASSES, rgb_mask_to_label_mask as rgb2id_fine
    from metrics.classes_merged import CLASSES_MERGED, rgb_mask_to_label_mask as rgb2id_coarse
    from metrics.evaluate_seg import evaluate

    # 公式の総当たり HD は 4K で計算不能なため、等価な EDT 実装に差し替える
    # (パッチ前に公式実装との一致を assert。fast_hd.py 参照)
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
        # gt は予測が存在するファイルのみ (fold 部分評価にも対応) — symlink で用意
        gt_dir = oof_dir / f"gt_{task}"
        gt_dir.mkdir(exist_ok=True)
        for old in gt_dir.glob("*.png"):
            old.unlink()
        for n in pred_names:
            (gt_dir / n).symlink_to(REPO / "data" / gt_src / n)
        log.info("%s: evaluating %d images (official code)...", task, len(pred_names))
        res = evaluate(gt_dir, pred_dir, classes=classes, rgb_to_label_fn=rgb_fn)
        results[task] = {
            "final_dice": res["final_dice"],
            "final_hd": res["final_hd"],
            "n_images": len(pred_names),
            "n_cases": len(res["case_ids"]),
            "case_dice": res["case_dice"],
            "case_hd": res["case_hd"],
        }
        log.info("%s | Dice=%.4f HD=%.4f (%d cases)", task, res["final_dice"], res["final_hd"],
                 len(res["case_ids"]))

    (oof_dir / "oof_metrics.json").write_text(json.dumps(results, indent=2))
    log.info("saved %s", oof_dir / "oof_metrics.json")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--folds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    ap.add_argument("--config", default=str(Path(__file__).parent / "config.yaml"))
    ap.add_argument("--eval-only", action="store_true")
    ap.add_argument("--no-eval", action="store_true", help="推論のみ (評価は後で一括)")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--exp-name", default=None, help="experiment 名の上書き (smoke 検証用)")
    ap.add_argument("--limit", type=int, default=None, help="fold あたりの推論枚数上限 (smoke 検証用)")
    args = ap.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text())
    if args.exp_name:
        cfg["experiment"]["name"] = args.exp_name
    oof_dir = REPO / cfg["paths"]["results_root"] / cfg["experiment"]["name"] / "oof"
    out1, out2 = oof_dir / "task1", oof_dir / "task2"
    out1.mkdir(parents=True, exist_ok=True)
    out2.mkdir(parents=True, exist_ok=True)

    if not args.eval_only:
        for fold in args.folds:
            predict_fold(cfg, fold, out1, out2, args.device, limit=args.limit)
    if not args.no_eval:
        evaluate_oof(cfg, out1, out2, oof_dir)


if __name__ == "__main__":
    main()
