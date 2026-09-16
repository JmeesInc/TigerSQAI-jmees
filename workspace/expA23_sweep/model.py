"""expA23: 任意の smp アーキ × 任意の timm encoder を dual-head 化する汎用ビルダ.

Task1 (coarse 16) と Task2 (fine 31) は **encoder を共有し decoder を 2 本持つ**
（expA00 以来の全実験と同じ構成。ここを変えると過去の比較が切れる）。

3 つの構築経路:

1. `source: timm` + `arch: unetplusplus`
   smp の TimmUniversalEncoder は ConvNeXt 等で 1/2 解像度の特徴を **ダミー (C=0)** で
   埋めるため、UnetPlusPlusDecoder が Conv2d(out_channels=0) を作って落ちる
   (expA19 model.py と同じ問題)。C=0 段を encoder_channels と features の双方から
   除外して 1 段浅い Unet++ を作る。`attention: scse` はここで効く。

2. `source: timm` + それ以外の arch (upernet/fpn/manet/deeplabv3plus/segformer/dpt/pan/pspnet)
   `smp.create_model` を fine/coarse の 2 本作り、**coarse 側の encoder を fine 側と
   同一オブジェクトに差し替える**ことで重みを共有する。
   (unetplusplus と linknet 以外は C=0 段があっても構築・forward が通ることを確認済み)

3. `source: smp_hub`
   `smp.from_pretrained("smp-hub/...")` で **decoder ごと事前学習された**モデルを読み、
   segmentation_head だけ 31/16 クラスに差し替える。ADE20k 等で学習済みの decoder は
   ImageNet encoder + 乱数 decoder とは初期化が根本的に違うのでアンサンブル多様性が高い。
   swin のように入力サイズ固定の encoder には img_size を渡す必要がある。
"""

from __future__ import annotations

import logging
from typing import Sequence

import segmentation_models_pytorch as smp
import torch
import torch.nn as nn
from segmentation_models_pytorch.base import SegmentationHead
from segmentation_models_pytorch.decoders.unetplusplus.decoder import UnetPlusPlusDecoder
from segmentation_models_pytorch.encoders import get_encoder

log = logging.getLogger(__name__)

# 入力サイズを encoder に渡す必要があるもの (window/grid 分割やパッチ数が入力依存)
_NEEDS_IMG_SIZE = ("maxvit", "swin", "beit", "vit_", "eva")


def _filter_zero(channels: Sequence[int]) -> list[int]:
    return [channels[0]] + [c for c in channels[1:] if c > 0]


class _DualUnetPP(nn.Module):
    """経路 1: 自前で encoder 1 本 + UnetPlusPlusDecoder 2 本を組む。"""

    def __init__(self, encoder_name, encoder_weights, decoder_channels,
                 n_fine, n_coarse, attention=None, img_size=None, **_):
        super().__init__()
        kw = {}
        # swin/maxvit は window 分割が入力サイズ依存なので img_size を渡して再計算させる
        if img_size is not None and any(k in encoder_name for k in ("swin", "maxvit")):
            kw["img_size"] = tuple(img_size)
        self.encoder = get_encoder(encoder_name, in_channels=3, depth=5,
                                   weights=encoder_weights, **kw)
        full = list(self.encoder.out_channels)
        self._full_channels = full
        kept = _filter_zero(full)
        depth = len(kept) - 1
        assert depth > 0, f"bad encoder channels: {full}"
        dc = tuple(decoder_channels)[:depth]
        assert len(dc) == depth, f"depth={depth} に対し decoder_channels が {len(dc)} 個"
        common = dict(encoder_channels=kept, decoder_channels=dc, n_blocks=depth,
                      use_norm="batchnorm", attention_type=attention, center=False)
        self.decoder_fine = UnetPlusPlusDecoder(**common)
        self.decoder_coarse = UnetPlusPlusDecoder(**common)
        self.head_fine = SegmentationHead(dc[-1], n_fine, kernel_size=3)
        self.head_coarse = SegmentationHead(dc[-1], n_coarse, kernel_size=3)

    def forward(self, x):
        feats = self.encoder(x)
        kept = [feats[0]] + [f for f, c in zip(feats[1:], self._full_channels[1:]) if c > 0]
        return self.head_fine(self.decoder_fine(kept)), self.head_coarse(self.decoder_coarse(kept))


class _DualSMP(nn.Module):
    """経路 2/3: smp のモデルを 2 本作り encoder を共有する。"""

    def __init__(self, model_fine: nn.Module, model_coarse: nn.Module):
        super().__init__()
        # coarse 側の encoder を fine 側と同一モジュールにする (= 重み共有・パラメータ 1 組)
        model_coarse.encoder = model_fine.encoder
        self.m_fine = model_fine
        self.m_coarse = model_coarse

    def forward(self, x):
        out = self.m_fine.encoder(x)
        # DPT の encoder は (features, prefix_tokens) を返し decoder が両方を要求する
        args = out if (isinstance(out, tuple) and len(out) == 2
                       and isinstance(out[0], (list, tuple))) else (out,)
        lf = self.m_fine.segmentation_head(self.m_fine.decoder(*args))
        lc = self.m_coarse.segmentation_head(self.m_coarse.decoder(*args))
        return lf, lc


def _replace_head(model: nn.Module, n_classes: int) -> None:
    """head の **最後の Conv2d だけ**を差し替える.

    smp の head は arch ごとに形が違う (Sequential / DPTSegmentationHead / ...)。
    出力クラス数を決めているのは常に最後の Conv2d なので、そこだけ作り直せば
    残りの構造 (upsampling や中間 conv) を壊さずにクラス数を変えられる。
    """
    head = model.segmentation_head
    last_name, last_conv = None, None
    for name, mod in head.named_modules():
        if isinstance(mod, nn.Conv2d):
            last_name, last_conv = name, mod
    assert last_conv is not None, f"head に Conv2d が無い: {type(head).__name__}"
    new_conv = nn.Conv2d(last_conv.in_channels, n_classes,
                         kernel_size=last_conv.kernel_size, stride=last_conv.stride,
                         padding=last_conv.padding, bias=last_conv.bias is not None)
    parent = head
    *parents, leaf = last_name.split(".")
    for p_ in parents:
        parent = getattr(parent, p_) if not p_.isdigit() else parent[int(p_)]
    if leaf.isdigit():
        parent[int(leaf)] = new_conv
    else:
        setattr(parent, leaf, new_conv)


class DualHeadSeg(nn.Module):
    """(logit_fine, logit_coarse) を **必ず入力解像度で**返すラッパ。"""

    def __init__(self, core: nn.Module):
        super().__init__()
        self.core = core

    def forward(self, x):
        lf, lc = self.core(x)
        if lf.shape[-2:] != x.shape[-2:]:
            lf = nn.functional.interpolate(lf, size=x.shape[-2:], mode="bilinear", align_corners=False)
        if lc.shape[-2:] != x.shape[-2:]:
            lc = nn.functional.interpolate(lc, size=x.shape[-2:], mode="bilinear", align_corners=False)
        return lf, lc


def build_model(m: dict, img_size: tuple[int, int]) -> DualHeadSeg:
    n_fine = int(m.get("num_classes_fine", 31))
    n_coarse = int(m.get("num_classes_coarse", 16))
    source = m.get("source", "timm")

    if source == "smp_hub":
        hub_id = m["hub_id"]
        kw = {}
        # swin は window 分割が入力サイズ依存なので img_size が要る。
        # 一方 DPT/ViT 系は位置埋め込みが img_size で作られるため、渡すと
        # 事前学習重みの strict load に失敗する (可変長補間は encoder 側が持っている)
        if "swin" in hub_id:
            kw["img_size"] = tuple(img_size)
        mf = smp.from_pretrained(hub_id, **kw)
        mc = smp.from_pretrained(hub_id, **kw)
        _replace_head(mf, n_fine)
        _replace_head(mc, n_coarse)
        log.info("smp_hub %s: %s / decoder ごと事前学習、head のみ %d/%d クラスへ差し替え",
                 hub_id, type(mf).__name__, n_fine, n_coarse)
        core = _DualSMP(mf, mc)
    elif m["arch"] == "unetplusplus":
        core = _DualUnetPP(
            encoder_name=m["encoder_name"], encoder_weights=m.get("encoder_weights", "imagenet"),
            decoder_channels=tuple(m.get("decoder_channels", (256, 128, 64, 32, 16))),
            n_fine=n_fine, n_coarse=n_coarse, attention=m.get("attention"),
            img_size=img_size,
        )
        log.info("unetplusplus %s attention=%s", m["encoder_name"], m.get("attention"))
    else:
        kw = dict(arch=m["arch"], encoder_name=m["encoder_name"],
                  encoder_weights=m.get("encoder_weights", "imagenet"))
        extra = dict(m.get("arch_kwargs", {}) or {})
        if any(k in m["encoder_name"] for k in _NEEDS_IMG_SIZE):
            extra.setdefault("img_size", tuple(img_size))
        mf = smp.create_model(**kw, classes=n_fine, **extra)
        mc = smp.create_model(**kw, classes=n_coarse, **extra)
        log.info("%s %s (%s)", m["arch"], m["encoder_name"], extra or "no extra kwargs")
        core = _DualSMP(mf, mc)

    model = DualHeadSeg(core)
    n_par = sum(p.numel() for p in model.parameters()) / 1e6
    log.info("model params: %.1fM", n_par)
    return model
