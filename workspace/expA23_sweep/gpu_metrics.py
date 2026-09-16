"""公式 weighted Dice + 正規化 HD の GPU 版（選択用）.

  * Dice: 原寸・厳密（公式と同じ空集合規約: 両方空=1.0 / 片方空=0.0）
  * HD  : 近似。長辺が ~960 になるよう max-pool で縮小し、JFA(EDT 近似) で
          両方向の directed HD を取る。距離は縮小率を掛け戻して対角線で正規化。
          空集合規約は公式と同じ（両方空=0.0 / 片方空=1.0）。
          誤差は「max-pool による最大 s 画素の膨張 + JFA の近似」で、対角線正規化後は
          1e-3 オーダー（validate_gpu_metrics.py で実測）。
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "reference" / "tigersqai_challenge"))
from fast_edt import edt2d_jfa  # noqa: E402


def _class_table(classes, device):
    ids = torch.tensor([c.label_id for c in classes], device=device, dtype=torch.int64)
    w = torch.tensor([float(c.weight) for c in classes], device=device)
    return ids, w, float(sum(c.weight for c in classes))


@torch.no_grad()
def weighted_scores_gpu(pred: torch.Tensor, gt: torch.Tensor, classes, hd_long: int = 960):
    """pred/gt: (H,W) uint8 on GPU。戻り値 (dice, hd) の float。"""
    dev = pred.device
    ids, w, wtot = _class_table(classes, dev)
    H, W = pred.shape
    P = pred[None] == ids[:, None, None]          # (C,H,W) bool
    G = gt[None] == ids[:, None, None]
    psum = P.flatten(1).sum(1).float()
    gsum = G.flatten(1).sum(1).float()
    inter = (P & G).flatten(1).sum(1).float()
    both0 = (psum == 0) & (gsum == 0)
    one0 = ((psum == 0) | (gsum == 0)) & ~both0
    dice = torch.where(both0, torch.ones_like(psum),
                       torch.where(one0, torch.zeros_like(psum), 2 * inter / (psum + gsum).clamp(min=1)))

    # --- HD (近似)
    s = max(1, int(math.ceil(max(H, W) / hd_long)))
    if s > 1:
        Ps = F.max_pool2d(P[None].float(), s, s)[0] > 0.5
        Gs = F.max_pool2d(G[None].float(), s, s)[0] > 0.5
    else:
        Ps, Gs = P, G
    need = ~(both0 | one0)
    hd = torch.where(both0, torch.zeros_like(psum), torch.ones_like(psum))
    if bool(need.any()):
        idx = need.nonzero().flatten()
        Pn, Gn = Ps[idx][None], Gs[idx][None]                 # (1,c,h,w)
        dG = edt2d_jfa(Gn)[0]                                  # G までの距離（P の画素で評価）
        dP = edt2d_jfa(Pn)[0]
        big = torch.tensor(-1.0, device=dev)
        a = torch.where(Pn[0], dG, big).flatten(1).max(1).values
        b = torch.where(Gn[0], dP, big).flatten(1).max(1).values
        diag = math.sqrt(H * H + W * W)
        hd[idx] = (torch.maximum(a, b) * s / diag).clamp(max=1.0)
    return float((dice * w).sum() / wtot), float((hd * w).sum() / wtot)
