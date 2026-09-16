"""expA13: CholecSeg8k (公開データ) で encoder + dual Unet++ decoder を事前学習する.

狙い: ImageNet 事前学習は encoder しか初期化できない。CholecSeg8k は 8,080 枚の
ピクセルラベル付き腹腔鏡データなので、**decoder ごと外科ドメインで事前学習**できる。
本コンペの学習データが 526 枚しかないことを考えると、ここが最も効く可能性がある。

- データ: data_ext/CholecSeg8k（公開データを配置）
  13 クラス。TMAM の LABEL2CH でグレースケール watershed 値 -> クラス ID
- 連続フレームは冗長なので **--stride で間引く** (既定 4 -> 約 2,020 枚)
- 解像度は下流と同一の 1024x576 (CholecSeg8k は 854x480 でアスペクト比が一致)
- モデルは expA06 と完全同一の DualHeadUnetPP。head だけクラス数が違うので
  転移時に捨てる。encoder + decoder_fine + decoder_coarse を持ち出す

Usage: python3 pretrain.py [--epochs 20] [--stride 4] [--device cuda:0]
出力: results/cholec_pretrain/{best.pt, latest.pt, transfer_weights.pt, training_log.json}
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
from monai.losses import DiceLoss
from torch.utils.data import DataLoader, Dataset

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent))

from dataset import IMAGENET_MEAN, IMAGENET_STD, build_transforms  # noqa: E402
from model import DualHeadUnetPP  # noqa: E402

log = logging.getLogger("cholec")

CHOLEC_ROOT = Path("data_ext/CholecSeg8k")
# TMAM train_cholec.py の CFG.LABEL2CH (13 クラス)
LABEL2CH = {0: 0, 50: 0, 255: 0, 5: 1, 11: 2, 12: 3, 13: 4,
            21: 5, 22: 6, 23: 7, 24: 8, 25: 9, 31: 10, 32: 11, 33: 12}
N_CLASSES = 13
IMG_H, IMG_W = 576, 1024


def build_lut() -> np.ndarray:
    lut = np.zeros(256, dtype=np.uint8)
    for k, v in LABEL2CH.items():
        lut[k] = v
    return lut


class CholecDataset(Dataset):
    def __init__(self, files: list[Path], train: bool):
        self.files = files
        self.tf = build_transforms(IMG_H, IMG_W, train)
        self.lut = build_lut()

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, i: int):
        mp = self.files[i]
        ip = mp.parent / mp.name.replace("_watershed_mask", "")
        img = cv2.cvtColor(cv2.imread(str(ip)), cv2.COLOR_BGR2RGB)
        mask = self.lut[cv2.imread(str(mp), cv2.IMREAD_GRAYSCALE)]
        out = self.tf(image=img, mask=mask)
        return out["image"], out["mask"].long()


def collect_files(stride: int) -> tuple[list[Path], list[Path]]:
    """video 単位で train/val 分割 (最後の 2 video を val)。連続フレームは stride で間引く."""
    videos = sorted(p for p in CHOLEC_ROOT.iterdir() if p.is_dir())
    assert videos, f"no videos under {CHOLEC_ROOT}"
    train_v, val_v = videos[:-2], videos[-2:]

    def gather(vs: list[Path]) -> list[Path]:
        out = []
        for v in vs:
            for clip in sorted(p for p in v.iterdir() if p.is_dir()):
                fs = sorted(clip.glob("*_watershed_mask.png"))
                out += fs[::stride]
        return out

    return gather(train_v), gather(val_v)


@torch.no_grad()
def validate(model, dl, device) -> float:
    """クラス平均 Dice (背景除く) を返す."""
    model.eval()
    inter = torch.zeros(N_CLASSES, device=device)
    denom = torch.zeros(N_CLASSES, device=device)
    for x, y in dl:
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        with torch.autocast("cuda", torch.float16):
            lf, _ = model(x)
        p = lf.float().argmax(1)
        for c in range(N_CLASSES):
            pc, yc = (p == c), (y == c)
            inter[c] += 2 * (pc & yc).sum()
            denom[c] += pc.sum() + yc.sum()
    dice = torch.where(denom > 0, inter / denom.clamp(min=1), torch.ones_like(inter))
    return float(dice[1:].mean())


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--stride", type=int, default=4)
    ap.add_argument("--batch", type=int, default=2)
    ap.add_argument("--accum", type=int, default=4)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--out", default="results/cholec_pretrain")
    args = ap.parse_args()

    out_dir = Path(__file__).parent / args.out
    out_dir.mkdir(parents=True, exist_ok=True)
    fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    root = logging.getLogger(); root.setLevel(logging.DEBUG)
    ch = logging.StreamHandler(); ch.setLevel(logging.INFO); ch.setFormatter(fmt)
    fh = logging.FileHandler(out_dir / f"pretrain_{datetime.now():%Y%m%d_%H%M%S}.log")
    fh.setLevel(logging.DEBUG); fh.setFormatter(fmt)
    root.addHandler(ch); root.addHandler(fh)
    (out_dir / "args.json").write_text(json.dumps(vars(args), indent=2))

    torch.manual_seed(42); np.random.seed(42)
    device = args.device
    tr_files, va_files = collect_files(args.stride)
    log.info("CholecSeg8k: train %d / val %d frames (stride=%d)", len(tr_files), len(va_files), args.stride)

    dl_tr = DataLoader(CholecDataset(tr_files, True), batch_size=args.batch, shuffle=True,
                       num_workers=args.workers, pin_memory=True, drop_last=True,
                       persistent_workers=args.workers > 0)
    dl_va = DataLoader(CholecDataset(va_files, False), batch_size=args.batch, shuffle=False,
                       num_workers=args.workers, pin_memory=True,
                       persistent_workers=args.workers > 0)

    model = DualHeadUnetPP(
        encoder_name="tu-maxvit_base_tf_512.in21k_ft_in1k", encoder_weights="imagenet",
        decoder_channels=(256, 128, 64, 32, 16),
        num_classes_fine=N_CLASSES, num_classes_coarse=N_CLASSES,
        img_size=(IMG_H, IMG_W),
    ).to(device)

    crit = DiceLoss(softmax=True, to_onehot_y=True, include_background=True)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    warmup = 2
    sched = torch.optim.lr_scheduler.SequentialLR(
        opt,
        [torch.optim.lr_scheduler.LinearLR(opt, start_factor=0.01, total_iters=warmup),
         torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(1, args.epochs - warmup))],
        milestones=[warmup],
    )
    scaler = torch.amp.GradScaler("cuda")

    start_ep, best, history = 0, -1.0, []
    latest = out_dir / "latest.pt"
    if latest.exists():  # レジューム (CLAUDE.md: rolling checkpoint 必須)
        ck = torch.load(latest, map_location="cpu", weights_only=False)
        model.load_state_dict(ck["model"]); opt.load_state_dict(ck["opt"])
        sched.load_state_dict(ck["sched"]); scaler.load_state_dict(ck["scaler"])
        start_ep, best, history = ck["epoch"] + 1, ck["best"], ck["history"]
        log.info("resumed from %s (epoch %d, best %.4f)", latest, start_ep, best)

    for ep in range(start_ep, args.epochs):
        model.train()
        tot, n = 0.0, 0
        opt.zero_grad(set_to_none=True)
        for i, (x, y) in enumerate(dl_tr):
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            with torch.autocast("cuda", torch.float16):
                lf, lc = model(x)
                yl = y.unsqueeze(1)
                loss = 0.5 * crit(lf, yl) + 0.5 * crit(lc, yl)
            scaler.scale(loss / args.accum).backward()
            if (i + 1) % args.accum == 0:
                scaler.step(opt); scaler.update(); opt.zero_grad(set_to_none=True)
            tot += loss.item(); n += 1
            if (i + 1) % 200 == 0:
                log.debug("ep%d %d/%d loss=%.4f", ep, i + 1, len(dl_tr), tot / n)
        sched.step()
        dice = validate(model, dl_va, device)
        history.append({"epoch": ep, "train_loss": tot / max(n, 1), "val_dice": dice})
        log.info("epoch %d | loss=%.4f val_dice(fg)=%.4f%s", ep, tot / max(n, 1), dice,
                 "  <- best" if dice > best else "")
        ck = {"model": model.state_dict(), "opt": opt.state_dict(), "sched": sched.state_dict(),
              "scaler": scaler.state_dict(), "epoch": ep, "best": max(best, dice), "history": history}
        torch.save(ck, latest)
        if dice > best:
            best = dice
            torch.save(ck, out_dir / "best.pt")
            # 転移用: head は クラス数が違うので捨て、encoder + 両 decoder のみ
            sd = model.state_dict()
            torch.save({k: v for k, v in sd.items() if not k.startswith("head_")},
                       out_dir / "transfer_weights.pt")
        (out_dir / "training_log.json").write_text(json.dumps(history, indent=2))

    log.info("done. best val_dice(fg)=%.4f -> %s", best, out_dir / "transfer_weights.pt")


if __name__ == "__main__":
    main()
