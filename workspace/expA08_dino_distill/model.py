"""Shared encoder + dual Unet++ decoder (Task1 fine 31ch / Task2 coarse 16ch)."""

from __future__ import annotations

import torch
import torch.nn as nn
from segmentation_models_pytorch.base import SegmentationHead
from segmentation_models_pytorch.decoders.unetplusplus.decoder import UnetPlusPlusDecoder
from segmentation_models_pytorch.encoders import get_encoder


class DualHeadUnetPP(nn.Module):
    def __init__(
        self,
        encoder_name: str,
        encoder_weights: str | None = "imagenet",
        decoder_channels: tuple[int, ...] = (256, 128, 64, 32, 16),
        num_classes_fine: int = 31,
        num_classes_coarse: int = 16,
        img_size: tuple[int, int] | None = None,
    ):
        super().__init__()
        # maxvit は window/grid attention の分割サイズが入力サイズ依存 (partition_ratio=1/32)。
        # tf_512 既定の window では 576x1024 が割り切れないため img_size を渡して再計算させる
        kwargs = {}
        if "maxvit" in encoder_name and img_size is not None:
            kwargs["img_size"] = tuple(img_size)
        self.encoder = get_encoder(encoder_name, in_channels=3, depth=5, weights=encoder_weights, **kwargs)
        assert len(self.encoder.out_channels) == 6, (
            f"encoder must provide 5-stage features, got out_channels={self.encoder.out_channels}"
        )
        common = dict(
            encoder_channels=self.encoder.out_channels,
            decoder_channels=decoder_channels,
            n_blocks=5,
            use_norm="batchnorm",
            attention_type=None,
            center=False,
        )
        self.decoder_fine = UnetPlusPlusDecoder(**common)
        self.decoder_coarse = UnetPlusPlusDecoder(**common)
        self.head_fine = SegmentationHead(decoder_channels[-1], num_classes_fine, kernel_size=3)
        self.head_coarse = SegmentationHead(decoder_channels[-1], num_classes_coarse, kernel_size=3)

    def forward(self, x: torch.Tensor, return_feats: bool = False):
        feats = self.encoder(x)
        out_f = self.head_fine(self.decoder_fine(feats))
        out_c = self.head_coarse(self.decoder_coarse(feats))
        if return_feats:
            return out_f, out_c, feats[4]  # stride-16 特徴 (蒸留の生徒側)
        return out_f, out_c
