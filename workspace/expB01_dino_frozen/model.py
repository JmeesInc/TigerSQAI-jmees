"""DINOv3-7B (frozen) + ViTDet 風 simple feature pyramid + dual Unet++ decoder.

- encoder: timm `vit_7b_patch16_dinov3.lvd1689m` (6.7B, RoPE なので可変解像度可)。
  **完全凍結 + fp16 + eval 固定**（BN 統計もないので train() で汚れない）
- ViT は stride-16 単一スケール (4096ch) → ViTDet 方式で学習可能ネックから
  stride 4/8/16/32 のピラミッドを生成。stride-2 skip は expA04 と同じ学習可能 conv stem
- checkpoint 肥大化対策: 凍結 ViT (13.4GB) は Lightning の checkpoint から除外する
  (train.py の on_save_checkpoint / strict_loading=False とセット)
"""

from __future__ import annotations

import logging

import timm
import torch
import torch.nn as nn
from segmentation_models_pytorch.base import SegmentationHead
from segmentation_models_pytorch.decoders.unetplusplus.decoder import UnetPlusPlusDecoder

log = logging.getLogger(__name__)

VIT_NAME = "vit_7b_patch16_dinov3.lvd1689m"


def _bnrelu(cout: int) -> list[nn.Module]:
    return [nn.BatchNorm2d(cout), nn.ReLU(inplace=True)]


class DINOv3PyramidEncoder(nn.Module):
    def __init__(
        self,
        img_size: tuple[int, int],
        pretrained: bool = True,
        pyr_ch: int = 256,
        stem_channels: int = 32,
    ):
        super().__init__()
        vit = timm.create_model(VIT_NAME, pretrained=pretrained, num_classes=0, img_size=img_size)
        vit.eval().half()
        for p in vit.parameters():
            p.requires_grad_(False)
        self.vit = vit
        self.grid = (img_size[0] // 16, img_size[1] // 16)
        d = vit.embed_dim  # 4096
        log.info("DINOv3 frozen: %.2fB params fp16, grid=%s",
                 sum(p.numel() for p in vit.parameters()) / 1e9, self.grid)

        self.p4 = nn.Sequential(
            nn.ConvTranspose2d(d, 512, 2, stride=2), *_bnrelu(512),
            nn.ConvTranspose2d(512, pyr_ch, 2, stride=2), *_bnrelu(pyr_ch),
        )
        self.p8 = nn.Sequential(nn.ConvTranspose2d(d, pyr_ch, 2, stride=2), *_bnrelu(pyr_ch))
        self.p16 = nn.Sequential(nn.Conv2d(d, pyr_ch, 1), *_bnrelu(pyr_ch))
        self.p32 = nn.Sequential(nn.Conv2d(d, pyr_ch, 3, stride=2, padding=1), *_bnrelu(pyr_ch))
        self.stem2 = nn.Sequential(
            nn.Conv2d(3, stem_channels, 3, stride=2, padding=1, bias=False), *_bnrelu(stem_channels)
        )
        self.out_channels = [3, stem_channels, pyr_ch, pyr_ch, pyr_ch, pyr_ch]
        self.output_stride = 32

    def train(self, mode: bool = True):  # 凍結 ViT は常に eval
        super().train(mode)
        self.vit.eval()
        return self

    def forward(self, x: torch.Tensor) -> list[torch.Tensor]:
        with torch.no_grad():
            t = self.vit.forward_features(x.half())
        t = t[:, self.vit.num_prefix_tokens:]
        b, n, d = t.shape
        t = t.transpose(1, 2).reshape(b, d, *self.grid).float()
        return [x, self.stem2(x), self.p4(t), self.p8(t), self.p16(t), self.p32(t)]


class DualHeadUnetPP(nn.Module):
    def __init__(
        self,
        encoder_name: str,
        encoder_weights: str | None = "imagenet",  # ここでは pretrained フラグ扱い (None=乱数)
        decoder_channels: tuple[int, ...] = (256, 128, 64, 32, 16),
        num_classes_fine: int = 31,
        num_classes_coarse: int = 16,
        img_size: tuple[int, int] = (576, 1024),
    ):
        super().__init__()
        assert encoder_name == "dinov3_vit7b_pyramid", encoder_name
        self.encoder = DINOv3PyramidEncoder(img_size=tuple(img_size), pretrained=encoder_weights is not None)
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
