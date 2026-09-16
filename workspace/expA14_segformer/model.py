"""Shared MiT encoder + dual SegFormer decoder（Task1/2 同時学習）.

expA06 の DualHeadUnetPP と同じ「共有 encoder + タスク別 decoder」構成を SegFormer で組む。
初期値は **CholecSeg8k 事前学習済み SegFormer**（別プロジェクト stitch_exp の成果物）:
    external/cholecseg8k_ckpt/segformer/fold{N}.pth
    = smp.Segformer(encoder_name="mit_b5", classes=13) を 512x512 で学習したもの

ImageNet 事前学習は encoder しか初期化できないのに対し、この重みは **decoder まで
外科ドメインで事前学習済み**。本コンペの学習データが 526 枚しかないため、ここが効く見込み。

- ckpt の decoder は 1 本なので、**fine / coarse の両 decoder に同じ重みをコピー**して開始する
- segmentation_head は クラス数が違う (13 vs 31/16) ので捨てて乱数初期化
- fold i には cholec fold i の重みを当てる（fold 間の多様性を稼ぐ。CholecSeg8k の
  分割は本コンペの fold と無関係なのでリークにはならない）
"""

from __future__ import annotations

import logging
from pathlib import Path

import torch
import torch.nn as nn
from segmentation_models_pytorch.base import SegmentationHead
from segmentation_models_pytorch.decoders.segformer.decoder import SegformerDecoder
from segmentation_models_pytorch.encoders import get_encoder

log = logging.getLogger(__name__)

CHOLEC_CKPT_DIR = Path("external/cholecseg8k_ckpt/segformer")


class DualHeadSegformer(nn.Module):
    def __init__(
        self,
        encoder_name: str = "mit_b5",
        encoder_weights: str | None = "imagenet",
        decoder_segmentation_channels: int = 256,
        num_classes_fine: int = 31,
        num_classes_coarse: int = 16,
        img_size: tuple[int, int] | None = None,  # SegFormer は可変長。API 互換のため受けるだけ
        cholec_fold: int | None = None,           # CholecSeg8k 事前学習重みの fold
        **_: object,
    ):
        super().__init__()
        self.encoder = get_encoder(encoder_name, in_channels=3, depth=5, weights=encoder_weights)
        common = dict(
            encoder_channels=self.encoder.out_channels,
            encoder_depth=5,
            segmentation_channels=decoder_segmentation_channels,
        )
        self.decoder_fine = SegformerDecoder(**common)
        self.decoder_coarse = SegformerDecoder(**common)
        # SegFormer の head は 1x1 conv + x4 upsample（smp 既定）
        self.head_fine = SegmentationHead(
            decoder_segmentation_channels, num_classes_fine, kernel_size=1, upsampling=4
        )
        self.head_coarse = SegmentationHead(
            decoder_segmentation_channels, num_classes_coarse, kernel_size=1, upsampling=4
        )
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
        # smp の MiT encoder は load_state_dict を strict 引数なしで override している
        # (head.* を落とすだけ)。内部は strict=True なので、ズレがあればここで落ちる
        self.encoder.load_state_dict(enc)
        # decoder は 1 本 -> fine / coarse の両方に同じ重みを載せる
        for d in (self.decoder_fine, self.decoder_coarse):
            d.load_state_dict(dec, strict=True)
        log.info(
            "CholecSeg8k 事前学習を読み込み: %s (encoder %d / decoder %d tensors) "
            "— head は クラス数不一致 (13 vs 31/16) のため乱数初期化",
            ckpt.name, len(enc), len(dec),
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        feats = self.encoder(x)
        return (
            self.head_fine(self.decoder_fine(feats)),
            self.head_coarse(self.decoder_coarse(feats)),
        )


# train.py / predict_oof.py が import する名前に合わせる
DualHeadUnetPP = DualHeadSegformer
