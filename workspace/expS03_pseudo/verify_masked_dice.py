"""MaskedDiceLoss が valid=全1 のとき MONAI DiceLoss と一致することを確認する.

一致しないと expA06 (ベースライン) との比較が「loss を変えたせい」と混同されるため,
学習を回す前に必ず通す.
"""
import sys
from pathlib import Path
import torch
from monai.losses import DiceLoss

sys.path.insert(0, str(Path(__file__).resolve().parent))
from train import MaskedDiceLoss  # noqa: E402

torch.manual_seed(0)
B, C, H, W = 3, 31, 32, 48
w = [1, 1, 1, 2, 2, 3] + [1] * (C - 6)

logits = torch.randn(B, C, H, W)
target = torch.randint(0, C, (B, 1, H, W))

monai = DiceLoss(softmax=True, to_onehot_y=True, include_background=True,
                 weight=torch.tensor(w, dtype=torch.float32))
mine = MaskedDiceLoss(w)

a = monai(logits, target).item()
b = mine(logits, target, torch.ones(B, H, W)).item()
c = mine(logits, target, None).item()
print(f"MONAI      : {a:.8f}")
print(f"masked(1)  : {b:.8f}   diff {abs(a-b):.2e}")
print(f"masked(None): {c:.8f}   diff {abs(a-c):.2e}")
assert abs(a - b) < 1e-6 and abs(a - c) < 1e-6, "MONAI と一致しない"

# ignore 画素が本当に効いているか: 無効領域の予測を壊しても loss が変わらないこと
valid = torch.ones(B, H, W)
valid[:, :, 24:] = 0
base = mine(logits, target, valid).item()
broken = logits.clone()
broken[:, :, :, 24:] = torch.randn_like(broken[:, :, :, 24:]) * 10
after = mine(broken, target, valid).item()
print(f"ignore 領域を壊す前後: {base:.8f} -> {after:.8f}  diff {abs(base-after):.2e}")
assert abs(base - after) < 1e-6, "ignore 画素が loss に漏れている"

# sample_w: 重み 0 のサンプルが寄与しないこと
sw = torch.tensor([1.0, 1.0, 0.0])
l_all = mine(logits[:2], target[:2], torch.ones(2, H, W))
l_w = mine(logits, target, torch.ones(B, H, W), sw)
print(f"sample_w=0 の除外: {l_all.item():.8f} vs {l_w.item():.8f}  diff {abs(l_all-l_w).item():.2e}")
assert abs(l_all - l_w).item() < 1e-6, "sample_w が効いていない"
print("\nOK: 3 件すべて通過")
