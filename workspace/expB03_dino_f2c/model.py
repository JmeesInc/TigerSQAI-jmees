"""DINOv3-7B (frozen) + MaxViT (trainable) の融合ピラミッド + dual Unet++ decoder.

- MaxViT-Base (tf_512, in21k_ft_in1k, img_size 再計算) が通常の 5 段ピラミッドを供給 (expA05 と同一)
- 凍結 DINOv3-7B の stride-16 特徴 (4096ch) を射影して MaxViT の stride-16/32 ステージへ加算注入
  - 射影後の BatchNorm は **γ=0 初期化** → 学習初期は注入ゼロ (= expA05 と等価) から始め、
    有益なら開く。事前学習済み MaxViT の表現を壊さない
- checkpoint 肥大化対策: 凍結 ViT は checkpoint から除外 (train.py 側の hook とセット)
"""

from __future__ import annotations

import logging

import timm
import torch
import torch.nn as nn
from segmentation_models_pytorch.base import SegmentationHead
from segmentation_models_pytorch.decoders.unetplusplus.decoder import UnetPlusPlusDecoder
from segmentation_models_pytorch.encoders import get_encoder

log = logging.getLogger(__name__)

VIT_NAME = "vit_7b_patch16_dinov3.lvd1689m"
MAXVIT_NAME = "tu-maxvit_base_tf_512.in21k_ft_in1k"


class FusionEncoder(nn.Module):
    def __init__(self, img_size: tuple[int, int], maxvit_weights: str | None = "imagenet",
                 vit_pretrained: bool = True):
        super().__init__()
        self.maxvit = get_encoder(MAXVIT_NAME, in_channels=3, depth=5,
                                  weights=maxvit_weights, img_size=tuple(img_size))
        vit = timm.create_model(VIT_NAME, pretrained=vit_pretrained, num_classes=0, img_size=tuple(img_size))
        vit.eval().half()
        for p in vit.parameters():
            p.requires_grad_(False)
        self.vit = vit
        self.grid = (img_size[0] // 16, img_size[1] // 16)
        d = vit.embed_dim  # 4096
        ch = list(self.maxvit.out_channels)  # [3, 64, 96, 192, 384, 768]
        self.proj16 = nn.Sequential(nn.Conv2d(d, ch[4], 1, bias=False), nn.BatchNorm2d(ch[4]))
        self.proj32 = nn.Sequential(nn.Conv2d(d, ch[5], 3, stride=2, padding=1, bias=False),
                                    nn.BatchNorm2d(ch[5]))
        for m in (self.proj16, self.proj32):  # γ=0 で注入をゼロ初期化
            nn.init.zeros_(m[1].weight)
        self.out_channels = ch
        self.output_stride = 32
        log.info("FusionEncoder: maxvit ch=%s + DINOv3 %.2fB frozen (inject s16/s32, zero-init)",
                 ch, sum(p.numel() for p in vit.parameters()) / 1e9)

    def train(self, mode: bool = True):
        super().train(mode)
        self.vit.eval()
        return self

    def forward(self, x: torch.Tensor) -> list[torch.Tensor]:
        feats = self.maxvit(x)
        with torch.no_grad():
            t = self.vit.forward_features(x.half())
        t = t[:, self.vit.num_prefix_tokens:]
        b, n, d = t.shape
        t = t.transpose(1, 2).reshape(b, d, *self.grid).float()
        feats[4] = feats[4] + self.proj16(t)
        feats[5] = feats[5] + self.proj32(t)
        return feats


class DualHeadUnetPP(nn.Module):
    def __init__(
        self,
        encoder_name: str,
        encoder_weights: str | None = "imagenet",
        decoder_channels: tuple[int, ...] = (256, 128, 64, 32, 16),
        num_classes_fine: int = 31,
        num_classes_coarse: int = 16,
        img_size: tuple[int, int] = (576, 1024),
    ):
        super().__init__()
        assert encoder_name == "maxvit_dinov3_fusion", encoder_name
        self.encoder = FusionEncoder(
            img_size=tuple(img_size),
            maxvit_weights=encoder_weights,
            vit_pretrained=encoder_weights is not None,
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

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        feats = self.encoder(x)
        return (
            self.head_fine(self.decoder_fine(feats)),
            self.head_coarse(self.decoder_coarse(feats)),
        )
