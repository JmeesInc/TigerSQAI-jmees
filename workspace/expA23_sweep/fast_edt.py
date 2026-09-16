"""2D 専用の高速 EDT（GPU, 完全微分不可だが forward のみで十分）と Hausdorff 系 loss の部品.

MONAI の `HausdorffDTLoss` は 3D も想定した汎用実装で、距離変換に scipy（CPU）を使うため
1 step 8 秒かかり 20 epoch に 12 時間必要だった（expA23 で一度却下）。
2D に限れば Felzenszwalb & Huttenlocher の分離可能アルゴリズムを
**GPU 上でバッチ × クラスを一括**に回せる。1 次元の下側包絡線走査は位置方向に逐次だが、
他の全次元をベクトル化すれば実用速度になる。

  edt2d_sq(mask) -> 各画素から mask=1 の最近傍までの **二乗**ユークリッド距離

GT 側の距離マップは dataset 側で前計算済み（diceboundary が使用）。
ここが必要なのは **予測側の距離マップ**（対称 HD の残り半分 = 遠い false negative）。
"""

from __future__ import annotations

import torch

INF = 1e10


def _edt1d_sq(f: torch.Tensor) -> torch.Tensor:
    """最終次元に沿った 1 次元二乗距離変換（下側包絡線法）.

    f: (..., N) 各位置のコスト（0 = 前景, INF = 背景）
    返り値: (..., N) 二乗距離
    """
    *rest, n = f.shape
    f = f.reshape(-1, n)
    b = f.shape[0]
    dev, dt = f.device, f.dtype
    v = torch.zeros(b, n, dtype=torch.long, device=dev)          # 放物線の中心
    z = torch.full((b, n + 1), INF, dtype=dt, device=dev)        # 交点
    z[:, 0] = -INF
    k = torch.zeros(b, dtype=torch.long, device=dev)             # 現在の包絡線の末尾
    ar = torch.arange(b, device=dev)
    for q in range(1, n):
        fq = f[:, q]
        while True:
            vk = v[ar, k]
            s = ((fq + q * q) - (f[ar, vk] + vk * vk)) / (2.0 * q - 2.0 * vk)
            bad = (s <= z[ar, k]) & (k > 0)
            if not bool(bad.any()):
                break
            k = torch.where(bad, k - 1, k)
        vk = v[ar, k]
        s = ((fq + q * q) - (f[ar, vk] + vk * vk)) / (2.0 * q - 2.0 * vk)
        k = k + 1
        v[ar, k] = q
        z[ar, k] = s
        z[ar, k + 1] = INF
    out = torch.empty_like(f)
    k = torch.zeros(b, dtype=torch.long, device=dev)
    for q in range(n):
        while True:
            adv = z[ar, k + 1] < q
            if not bool(adv.any()):
                break
            k = torch.where(adv, k + 1, k)
        vk = v[ar, k]
        out[:, q] = (q - vk).to(dt) ** 2 + f[ar, vk]
    return out.reshape(*rest, n)


def edt2d_sq(mask: torch.Tensor) -> torch.Tensor:
    """mask=1 の集合までの二乗ユークリッド距離。mask: (..., H, W) bool/float。"""
    f = torch.where(mask > 0.5, torch.zeros_like(mask, dtype=torch.float32),
                    torch.full_like(mask, INF, dtype=torch.float32))
    f = _edt1d_sq(f)                       # 行方向
    f = _edt1d_sq(f.transpose(-1, -2)).transpose(-1, -2)   # 列方向
    return f


@torch.no_grad()
def pred_distance_maps(logits: torch.Tensor, thr: float = 0.5, down: int = 4) -> torch.Tensor:
    """予測（argmax ではなく確率 > thr）までの距離マップ。(B,C,H/down,W/down) を返す。

    対称 Hausdorff の「GT にあるのに予測から遠い画素」を罰する側で使う。
    勾配は距離マップ側には流さない（距離変換は微分不可）。重みとしてのみ使う。
    """
    p = torch.nn.functional.avg_pool2d(logits.float().softmax(1), down)
    return edt2d_sq(p > thr).sqrt()


# ---------------------------------------------------------------- Jump Flooding (JFA)
# 上の包絡線法は厳密だが位置方向の逐次ループが GPU で遅い（1/4 解像度 62 枚で 10.6 秒）。
# JFA は log2(N) 回の近傍参照だけで最近傍シードを伝播させる完全並列アルゴリズムで、
# まれに誤差が出るが距離を **loss の重み**に使う用途では十分。

def edt2d_jfa(mask: torch.Tensor) -> torch.Tensor:
    """mask=1 までのユークリッド距離（JFA 近似）。mask: (B,C,H,W)。"""
    b, c, h, w = mask.shape
    dev = mask.device
    yy, xx = torch.meshgrid(torch.arange(h, device=dev), torch.arange(w, device=dev), indexing="ij")
    big = torch.tensor(1e4, device=dev, dtype=torch.float32)
    seed_y = torch.where(mask > 0.5, yy.expand(b, c, h, w).float(), big)
    seed_x = torch.where(mask > 0.5, xx.expand(b, c, h, w).float(), big)
    step = 1 << (max(h, w) - 1).bit_length() - 1
    while step >= 1:
        best_y, best_x = seed_y, seed_x
        d_best = (best_y - yy) ** 2 + (best_x - xx) ** 2
        for dy in (-step, 0, step):
            for dx in (-step, 0, step):
                if dy == 0 and dx == 0:
                    continue
                sy = torch.roll(seed_y, shifts=(dy, dx), dims=(2, 3))
                sx = torch.roll(seed_x, shifts=(dy, dx), dims=(2, 3))
                d = (sy - yy) ** 2 + (sx - xx) ** 2
                upd = d < d_best
                d_best = torch.where(upd, d, d_best)
                best_y = torch.where(upd, sy, best_y)
                best_x = torch.where(upd, sx, best_x)
        seed_y, seed_x = best_y, best_x
        step //= 2
    return ((seed_y - yy) ** 2 + (seed_x - xx) ** 2).clamp(max=1e8).sqrt()
