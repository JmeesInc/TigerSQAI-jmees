"""expT02 の分類ヘッドで Task3 OOF 確率を出力する.

Usage: python3 predict_t3.py [--folds 0] [--out results_t3_oof.csv]
出力 CSV は公式 evaluate_cls 互換 (case_id + 14 列確率)。
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
import yaml

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent))
from dataset import STATIONS, station_onehot  # noqa: E402
from model import DualHeadUnetPP  # noqa: E402

MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--folds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    ap.add_argument("--out", default="results_t3_oof.csv")
    args = ap.parse_args()

    cfg = yaml.safe_load((Path(__file__).parent / "config.yaml").read_text())
    gt = pd.read_csv(REPO / "workspace/data_proc/task3_gt_wide.csv")
    folds = pd.read_csv(REPO / cfg["cv"]["folds_csv"])
    stem2fold = {f.rsplit(".", 1)[0]: fo for f, fo in zip(folds.filename, folds.fold)}
    gt["fold"] = gt.case_id.map(stem2fold)
    device = "cuda"
    img_dir = REPO / "workspace/data_proc/images_1024"

    rows = []
    for fold in args.folds:
        ckpt = REPO / cfg["paths"]["results_root"] / cfg["experiment"]["name"] / f"fold{fold}/best.ckpt"
        m = DualHeadUnetPP(
            encoder_name=cfg["model"]["encoder_name"], encoder_weights=None,
            img_size=(cfg["data"]["img_h"], cfg["data"]["img_w"]),
        )
        state = torch.load(ckpt, map_location="cpu", weights_only=False)["state_dict"]
        state = {k.removeprefix("model."): v for k, v in state.items() if k.startswith("model.")}
        m.load_state_dict(state, strict=True)
        m = m.to(device).eval()

        va = gt[gt.fold == fold]
        for cid in va.case_id:
            img = cv2.cvtColor(cv2.imread(str(img_dir / f"{cid}.png")), cv2.COLOR_BGR2RGB)
            x = (img.astype(np.float32) / 255.0 - MEAN) / STD
            x = torch.from_numpy(x.transpose(2, 0, 1))[None].to(device)
            oh = torch.from_numpy(station_onehot(cid))[None].to(device)
            with torch.autocast("cuda", torch.float16):
                _, _, lt3 = m(x, oh)
            p = torch.sigmoid(lt3.float())[0].cpu().numpy()
            rows.append({"case_id": cid, **{s: float(p[i]) for i, s in enumerate(STATIONS)}})
        print(f"fold{fold}: {len(va)} rows")
        del m
        torch.cuda.empty_cache()

    pd.DataFrame(rows).to_csv(Path(__file__).parent / args.out, index=False)
    print("saved", args.out)


if __name__ == "__main__":
    main()
