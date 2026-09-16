"""アンサンブル追加メンバー (A09/A10/A11/A05) の重みを fp16 model-only で書き出す.

v003 までは expA06 の 5 fold だけだった。5 レシピ × 5 fold = 25 モデルの
softmax 平均にするため、追加 4 レシピ分を model_extra/ に変換する。
全レシピが同一アーキ (DualHeadUnetPP) なので model_def.py はそのまま使える。
A09 の lymph 補助枝など、学習時だけの枝は "model." 配下に無いので自動的に落ちる。
"""
from __future__ import annotations

import sys
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent))
from model_def import DualHeadUnetPP  # noqa: E402

EXTRA = {
    "A09": "expA09_lymph_aux",
    "A10": "expA10_toolmask",
    "A11": "expA11_v2data",
    "A05": "expA05_maxvit",
}
OUT = Path(__file__).parent / "model_extra"


def main() -> None:
    OUT.mkdir(exist_ok=True)
    for tag, exp in EXTRA.items():
        for fold in range(5):
            ck = REPO / "workspace" / exp / "results" / exp / f"fold{fold}" / "best.ckpt"
            assert ck.exists(), f"missing {ck}"
            sd = torch.load(ck, map_location="cpu", weights_only=False)["state_dict"]
            sd = {k.removeprefix("model."): v for k, v in sd.items() if k.startswith("model.")}
            # strict ロードでキー一致を検証してから fp16 で保存する
            m = DualHeadUnetPP(
                encoder_name="tu-maxvit_base_tf_512.in21k_ft_in1k", encoder_weights=None,
                num_classes_fine=31, num_classes_coarse=16, img_size=(576, 1024),
            )
            m.load_state_dict(sd, strict=True)
            half = {k: v.half() for k, v in m.state_dict().items()}
            p = OUT / f"{tag}_fold{fold}.pt"
            torch.save(half, p)
            print(f"{p.name}: {p.stat().st_size / 1e6:.0f} MB")
    print(f"total {sum(f.stat().st_size for f in OUT.glob('*.pt')) / 1e9:.2f} GB")


if __name__ == "__main__":
    main()
