"""Shared encoder + dual Unet++ decoder — SurgeNet-Public (CAFormer-S18) encoder 版.

SurgeNet (https://github.com/TimJaspers0801/SurgeNet) の手術動画 DINO 事前学習重みを
timm caformer_s18 にロードする。要点:

- SurgeNet 版 CAFormer は StarReLU ではなく素の ReLU を使う変種
  → timm モデルの StarReLU を全て nn.ReLU に置換してからロード (missing=0 を assert)
- キーは公式 sail-sg 命名 → timm の checkpoint_filter_fn でリマップ後、
  features_only 用に "stages.N." → "stages_N." に変換
- CAFormer は 4 ステージ (stride 4,8,16,32) で smp は stride2 に 0-ch ダミーを入れる
  → Unet++ が壊れるため、学習可能な stride-2 conv stem で実 skip に置き換える
- **ルール注意**: SurgeNet-Public は公開データのみで事前学習された variant
  (SurgeNetXL / SurgeNet / RAMIE は非公開データ RAMIE-UMCU / RARP-AvL を含むため使用不可)
"""

from __future__ import annotations

import logging
import re

import torch
import torch.nn as nn
from segmentation_models_pytorch.base import SegmentationHead
from segmentation_models_pytorch.decoders.unetplusplus.decoder import UnetPlusPlusDecoder
from segmentation_models_pytorch.encoders import get_encoder

log = logging.getLogger(__name__)


def _replace_starrelu_with_relu(module: nn.Module) -> int:
    from timm.models.metaformer import StarReLU

    n = 0
    for name, child in module.named_children():
        if isinstance(child, StarReLU):
            setattr(module, name, nn.ReLU())
            n += 1
        else:
            n += _replace_starrelu_with_relu(child)
    return n


class SurgeNetCAFormerEncoder(nn.Module):
    """tu-caformer_s18 (ReLU 版) + SurgeNet 重み + stride-2 学習可能 stem."""

    def __init__(self, surgenet_ckpt: str | None, stem_channels: int = 32):
        super().__init__()
        self.inner = get_encoder("tu-caformer_s18", in_channels=3, depth=5, weights=None)
        n_rep = _replace_starrelu_with_relu(self.inner)
        log.info("replaced %d StarReLU -> ReLU (SurgeNet variant)", n_rep)

        if surgenet_ckpt:
            import timm
            from timm.models.metaformer import checkpoint_filter_fn

            state = torch.load(surgenet_ckpt, map_location="cpu", weights_only=False)
            std = timm.create_model("caformer_s18", pretrained=False)
            mapped = checkpoint_filter_fn(state, std)
            remap = {re.sub(r"^stages\.(\d+)\.", r"stages_\1.", k): v for k, v in mapped.items()}
            missing, unexpected = self.inner.model.load_state_dict(remap, strict=False)
            # act 置換後は missing 0 のはず。unexpected は最終 norm / head のみ許容
            assert len(missing) == 0, f"missing keys after remap: {missing[:8]}"
            bad = [k for k in unexpected if not (k.startswith("head.") or k.startswith("norm."))]
            assert not bad, f"unexpected non-head keys: {bad[:8]}"
            log.info("SurgeNet weights loaded: %d tensors (unexpected head/norm: %d)",
                     len(remap) - len(unexpected), len(unexpected))

        self.stem2 = nn.Sequential(
            nn.Conv2d(3, stem_channels, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(stem_channels),
            nn.ReLU(inplace=True),
        )
        inner_ch = list(self.inner.out_channels)  # [3, 0, 64, 128, 320, 512]
        assert inner_ch[1] == 0, f"expected dummy stage-1, got {inner_ch}"
        self.out_channels = [inner_ch[0], stem_channels, *inner_ch[2:]]
        self.output_stride = 32

    def forward(self, x: torch.Tensor) -> list[torch.Tensor]:
        feats = self.inner(x)
        feats[1] = self.stem2(x)
        return feats


class DualHeadUnetPP(nn.Module):
    def __init__(
        self,
        encoder_name: str,
        encoder_weights: str | None = None,  # ここでは SurgeNet ckpt パス
        decoder_channels: tuple[int, ...] = (256, 128, 64, 32, 16),
        num_classes_fine: int = 31,
        num_classes_coarse: int = 16,
    ):
        super().__init__()
        assert encoder_name == "surgenet_caformer_s18", encoder_name
        self.encoder = SurgeNetCAFormerEncoder(surgenet_ckpt=encoder_weights)
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
