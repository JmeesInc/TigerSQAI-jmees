"""Shared ConvNeXt encoder + dual Unet++ decoder（Task1/2 同時学習）.

初期値は **CholecSeg8k 事前学習済み Unet++**（別プロジェクト stitch_exp の成果物）:
    external/cholecseg8k_ckpt/unetpp/fold{N}.pth
    = tu-convnext_base + Unet++ (13 クラス) を 512x512 で学習したもの

SegFormer 版 (expA14) では CholecSeg8k 事前学習の効果が +0.005 に留まった。
SegFormer の decoder は MLP 4 層 (14 テンソル) しかなく「decoder ごと事前学習できる」
利点が活きないため、と考えられる。Unet++ の decoder は 84 テンソルあるので、
事前学習の恩恵を正しく測れるのはこちら。

実装上の注意 (stitch_exp/exp/02_CholecSeg8k/model.py と同じ回避が必要):
  smp の TimmUniversalEncoder は ConvNeXt 等で 1/2 解像度の特徴を **ダミー (C=0)** で
  埋める。UnetPlusPlusDecoder は skip_channels を out_channels に使う経路があるため
  C=0 が混ざると Conv2d(out_channels=0) を作って落ちる。
  → C=0 の段を encoder_channels と features の両方から除外し、深さを 1 段浅くする
    (convnext_base では 1/4〜1/32 の 4 段 Unet++ = decoder_channels (256,128,64,32))
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Sequence

import torch
import torch.nn as nn
from segmentation_models_pytorch.base import SegmentationHead
from segmentation_models_pytorch.decoders.unetplusplus.decoder import UnetPlusPlusDecoder
from segmentation_models_pytorch.encoders import get_encoder

log = logging.getLogger(__name__)

CHOLEC_CKPT_DIR = Path("external/cholecseg8k_ckpt/unetpp")


def _filter_zero(channels: Sequence[int]) -> list[int]:
    return [channels[0]] + [c for c in channels[1:] if c > 0]


class DualHeadConvNeXtUnetPP(nn.Module):
    def __init__(
        self,
        encoder_name: str = "tu-convnext_base",
        encoder_weights: str | None = "imagenet",
        decoder_channels: Sequence[int] = (256, 128, 64, 32, 16),
        num_classes_fine: int = 31,
        num_classes_coarse: int = 16,
        img_size: tuple[int, int] | None = None,  # ConvNeXt は可変長。API 互換のため受けるだけ
        cholec_fold: int | None = None,
        **_: object,
    ):
        super().__init__()
        self.encoder = get_encoder(encoder_name, in_channels=3, depth=5, weights=encoder_weights)
        full = list(self.encoder.out_channels)
        self._full_channels = full
        kept = _filter_zero(full)
        depth = len(kept) - 1
        assert depth > 0, f"bad encoder channels: {full}"
        dc = tuple(decoder_channels)[:depth]
        assert len(dc) == depth, f"depth={depth} に対し decoder_channels が {len(dc)}"

        common = dict(
            encoder_channels=kept, decoder_channels=dc, n_blocks=depth,
            use_norm="batchnorm", attention_type=None, center=False,
        )
        self.decoder_fine = UnetPlusPlusDecoder(**common)
        self.decoder_coarse = UnetPlusPlusDecoder(**common)
        self.head_fine = SegmentationHead(dc[-1], num_classes_fine, kernel_size=3)
        self.head_coarse = SegmentationHead(dc[-1], num_classes_coarse, kernel_size=3)
        if cholec_fold is not None:
            self.load_cholec_pretrained(cholec_fold)

    def load_cholec_pretrained(self, fold: int) -> None:
        ckpt = CHOLEC_CKPT_DIR / f"fold{fold}.pth"
        assert ckpt.exists(), f"missing CholecSeg8k ckpt: {ckpt}"
        sd = torch.load(ckpt, map_location="cpu", weights_only=False)
        if isinstance(sd, dict) and "state_dict" in sd:
            sd = sd["state_dict"]
        enc = {k.removeprefix("encoder."): v for k, v in sd.items() if k.startswith("encoder.")}
        dec = {k.removeprefix("decoder."): v for k, v in sd.items() if k.startswith("decoder.")}
        assert enc and dec, f"unexpected ckpt layout: {list(sd)[:5]}"
        self.encoder.load_state_dict(enc, strict=True)
        # ckpt の decoder は 1 本 -> fine / coarse の両方に同じ重みを載せる
        for d in (self.decoder_fine, self.decoder_coarse):
            d.load_state_dict(dec, strict=True)
        log.info(
            "CholecSeg8k 事前学習を読み込み: %s (encoder %d / decoder %d tensors) "
            "— head は クラス数不一致 (13 vs 31/16) のため乱数初期化",
            ckpt.name, len(enc), len(dec),
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        feats = self.encoder(x)
        kept = [feats[0]] + [f for f, c in zip(feats[1:], self._full_channels[1:]) if c > 0]
        lf = self.head_fine(self.decoder_fine(kept))
        lc = self.head_coarse(self.decoder_coarse(kept))
        # C=0 段を落とした分だけ出力が 1/2 解像度になるので入力サイズへ戻す
        if lf.shape[-2:] != x.shape[-2:]:
            lf = nn.functional.interpolate(lf, size=x.shape[-2:], mode="bilinear", align_corners=False)
            lc = nn.functional.interpolate(lc, size=x.shape[-2:], mode="bilinear", align_corners=False)
        return lf, lc


DualHeadUnetPP = DualHeadConvNeXtUnetPP
