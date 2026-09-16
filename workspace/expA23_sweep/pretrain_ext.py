"""expA23: 外科ドメインの公開データで encoder+decoder ごと事前学習する.

ImageNet 事前学習は encoder しか初期化できない。CholecSeg8k (8,080 枚) / EndoVis2018
(2,235 枚) は腹腔鏡・内視鏡のピクセルラベル付き公開データなので、**decoder ごと**
外科ドメインに寄せられる。本コンペの学習データが 526 枚しかないので効く可能性がある。

- モデルは expA23 の `build_model` をそのまま使い、head のクラス数だけ外部データに合わせる。
  転移時は head 以外（encoder + decoder）を strict=False で読む（train.py の
  `model.init_from_partial`）。
- 入力解像度は下流と同じ 576x1024。
- 連続フレームは冗長なので `--stride` で間引く。

Usage:
    python3 pretrain_ext.py --data cholec --arch unetplusplus --device cuda:0
    python3 pretrain_ext.py --data endovis18 --arch deeplabv3plus --device cuda:1
出力: results/pre_<data>_<arch>/weights.pt  (head を除いた state_dict)
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(HERE))
from model import build_model  # noqa: E402
from dataset import build_transforms  # noqa: E402

log = logging.getLogger("pre")
IMG_H, IMG_W = 576, 1024
CHOLEC = Path(os.environ.get("CHOLECSEG8K_DIR", "data_ext/CholecSeg8k"))
ENDOVIS = Path(os.environ.get("ENDOVIS2018_DIR", "data_ext/EndoVis2018"))
# CholecSeg8k: watershed のグレースケール値 -> 13 クラス (TMAM train_cholec.py の LABEL2CH)
CHOLEC_LUT = {0: 0, 50: 0, 255: 0, 5: 1, 11: 2, 12: 3, 13: 4,
              21: 5, 22: 6, 23: 7, 24: 8, 25: 9, 31: 10, 32: 11, 33: 12}


def cholec_files(stride: int):
    vids = sorted(p for p in CHOLEC.iterdir() if p.is_dir())
    assert vids, CHOLEC

    def gather(vs):
        out = []
        for v in vs:
            for clip in sorted(p for p in v.iterdir() if p.is_dir()):
                out += sorted(clip.glob("*_watershed_mask.png"))[::stride]
        return out
    return gather(vids[:-2]), gather(vids[-2:])


def endovis_files(stride: int):
    seqs = sorted(p for p in (ENDOVIS / "train").iterdir() if p.is_dir())
    assert seqs, ENDOVIS

    def gather(vs):
        out = []
        for v in vs:
            out += sorted((v / "labels").glob("*.png"))[::stride]
        return out
    return gather(seqs[:-2]), gather(seqs[-2:])


class ExtDataset(Dataset):
    def __init__(self, files, train, data, color_lut=None):
        self.files, self.data, self.color_lut = files, data, color_lut
        self.mixed = data == "both"
        self.tf = build_transforms(IMG_H, IMG_W, train) if callable(build_transforms) else None
        if self.data in ("cholec", "both"):
            self.lut = np.zeros(256, np.uint8)
            for k, v in CHOLEC_LUT.items():
                self.lut[k] = v

    def __len__(self):
        return len(self.files)

    def __getitem__(self, i):
        mp = self.files[i]
        kind = self.data
        if self.mixed:
            kind = "cholec" if "_watershed_mask" in mp.name else "endovis18"
        if kind == "cholec":
            ip = mp.parent / mp.name.replace("_watershed_mask", "")
            mask = self.lut[cv2.imread(str(mp), cv2.IMREAD_GRAYSCALE)]
        else:
            pass
        if kind != "cholec":
            ip = mp.parent.parent / "left_frames" / mp.name
            bgr = cv2.imread(str(mp), cv2.IMREAD_COLOR)
            rgb = bgr[:, :, ::-1].astype(np.uint32)
            mask = self.color_lut[rgb[:, :, 0] * 65536 + rgb[:, :, 1] * 256 + rgb[:, :, 2]]
            if self.mixed:
                mask = np.where(mask > 0, mask.astype(np.int32) + 13, 0).astype(np.uint8)
        img = cv2.cvtColor(cv2.imread(str(ip)), cv2.COLOR_BGR2RGB)
        out = self.tf(image=img, mask=mask)
        return out["image"], out["mask"].long()


def build_color_lut():
    cls = json.load(open(ENDOVIS / "labels.json"))["classes"]
    lut = np.zeros(256 ** 3, np.uint8)
    for c in cls:
        r, g, b = c["color"][:3]
        lut[r * 65536 + g * 256 + b] = int(c["classid"])
    return lut, len(cls)


def dice_loss(logits, target, n):
    p = logits.float().softmax(1)
    t = F.one_hot(target.clamp(min=0), n).permute(0, 3, 1, 2).float()
    inter = (p * t).sum((2, 3))
    return (1 - (2 * inter + 1e-5) / (p.sum((2, 3)) + t.sum((2, 3)) + 1e-5)).mean()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", choices=["cholec", "endovis18", "both"], required=True)
    ap.add_argument("--arch", default="unetplusplus")
    ap.add_argument("--encoder", default="tu-convnext_large.fb_in22k_ft_in1k_384")
    ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--stride", type=int, default=6)
    ap.add_argument("--batch", type=int, default=2)
    ap.add_argument("--accum", type=int, default=4)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--tag", default="", help="出力ディレクトリ名の接尾辞（既存の重みを上書きしないため）")
    args = ap.parse_args()

    out = HERE / "results" / f"pre_{args.data}_{args.arch}{args.tag}"
    out.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s",
                        handlers=[logging.StreamHandler(),
                                  logging.FileHandler(out / "pretrain.log")])
    color_lut = None
    if args.data == "cholec":
        tr_f, va_f = cholec_files(args.stride); n_cls = 13
    elif args.data == "endovis18":
        tr_f, va_f = endovis_files(args.stride); color_lut, n_cls = build_color_lut()
    else:
        # 両方を 1 モデルで学習する。ラベル空間は連結 (cholec 13 + endovis 12 = 25)
        tc, vc = cholec_files(args.stride); te, ve = endovis_files(max(args.stride // 3, 1))
        color_lut, n_e = build_color_lut()
        tr_f, va_f = tc + te, vc + ve
        n_cls = 13 + n_e
    log.info("%s: train %d / val %d 枚, %d クラス", args.data, len(tr_f), len(va_f), n_cls)

    m = {"source": "timm", "arch": args.arch, "encoder_name": args.encoder,
         "encoder_weights": "imagenet", "decoder_channels": [256, 128, 64, 32, 16],
         "num_classes_fine": n_cls, "num_classes_coarse": n_cls}
    net = build_model(m, (IMG_H, IMG_W)).to(args.device)
    tr = DataLoader(ExtDataset(tr_f, True, args.data, color_lut), batch_size=args.batch,
                    shuffle=True, num_workers=args.workers, drop_last=True, pin_memory=True)
    va = DataLoader(ExtDataset(va_f, False, args.data, color_lut), batch_size=args.batch,
                    shuffle=False, num_workers=4)
    opt = torch.optim.AdamW(net.parameters(), lr=args.lr, weight_decay=1e-2)
    sch = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=args.lr, pct_start=0.1,
                                              total_steps=args.epochs * (len(tr) // args.accum + 1))
    scaler = torch.amp.GradScaler("cuda")
    hist = []
    best = 1e9
    for ep in range(args.epochs):
        net.train(); t0 = time.time(); run = 0.0
        opt.zero_grad(set_to_none=True)
        for i, (x, y) in enumerate(tr):
            x, y = x.to(args.device, non_blocking=True), y.to(args.device, non_blocking=True)
            with torch.amp.autocast("cuda", dtype=torch.float16):
                lf, lc = net(x)
                loss = 0.5 * (dice_loss(lf, y, n_cls) + dice_loss(lc, y, n_cls))
            scaler.scale(loss / args.accum).backward()
            run += float(loss)
            if (i + 1) % args.accum == 0:
                scaler.step(opt); scaler.update(); opt.zero_grad(set_to_none=True)
                if sch.last_epoch < sch.total_steps - 1:
                    sch.step()
            if i % 200 == 0:
                log.debug("ep%d %d/%d loss=%.4f", ep, i, len(tr), run / max(i, 1))
        net.eval(); vl = 0.0
        with torch.no_grad():
            for x, y in va:
                x, y = x.to(args.device), y.to(args.device)
                with torch.amp.autocast("cuda", dtype=torch.float16):
                    lf, _ = net(x)
                vl += float(dice_loss(lf, y, n_cls))
        vl /= max(len(va), 1)
        hist.append({"epoch": ep, "train": run / len(tr), "val": vl, "sec": round(time.time() - t0)})
        log.info("epoch %d | train %.4f | val dice_loss %.4f | %ds", ep, run / len(tr), vl, hist[-1]["sec"])
        if vl < best:
            best = vl
            sd = {k: v.half().cpu() for k, v in net.state_dict().items()
                  if "segmentation_head" not in k and "_head" not in k.split(".")[-2:][0]}
            torch.save(sd, out / "weights.pt")
        (out / "history.json").write_text(json.dumps(hist, indent=1))
    log.info("done. best val %.4f -> %s", best, out / "weights.pt")


if __name__ == "__main__":
    main()
