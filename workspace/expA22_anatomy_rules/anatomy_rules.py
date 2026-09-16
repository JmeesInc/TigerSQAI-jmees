"""解剖グラフ由来のルール loss (Task1/2 用).

ルール行列は workspace/anatomy_graph/rules.py が GT 統計から作る (fold 別, val 除外):
  W_adj  (C,C) 隣接ペナルティ [0,1] — GT で「両方写っているのに接したことが無い」クラス対ほど 1 に近い
  E_excl (C,C) 排他ペナルティ {0,1} — 同一画像に共起したことが無いクラス対

loss は softmax 確率 P (B,C,H,W) に対して微分可能に定義する:
  adjacency: 4 近傍の画素対 (x, x+d) について  sum_{a,b} W[a,b] P_a(x) P_b(x+d)
             を「境界質量」 sum_x (1 - sum_c P_c(x) P_c(x+d)) で正規化 (= 境界のうち禁止ペアの割合)。
             正規化項は detach して「境界を増やして比率を下げる」逆インセンティブを消す。
  exclusive: 画像単位の soft presence s_c = max_x avgpool(P_c) を使い  sum_{a<b} E[a,b] s_a s_b。
             片方を出さないことで下がる (どちらを消すかは Dice が決める)。
「個数」ルールは微分可能な定式化が重いので loss には入れず, 後処理 (K_max) で扱う。
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class AnatomyRuleLoss(nn.Module):
    def __init__(self, rules_npz: str | Path, presence_pool: int = 16, eps: float = 1e-6):
        super().__init__()
        r = np.load(rules_npz, allow_pickle=True)
        self.register_buffer("W", torch.from_numpy(r["W_adj"]).float())
        self.register_buffer("E", torch.triu(torch.from_numpy(r["E_excl"]).float(), diagonal=1))
        self.names = [str(n) for n in r["names"]]
        self.pool = presence_pool
        self.eps = eps
        self.n_adj_pairs = int((self.W > 0).sum().item() // 2)
        self.n_excl_pairs = int(self.E.sum().item())

    def _pairs(self, P: torch.Tensor):
        # (P(x), P(x+d)) を右・下シフトで作る
        yield P[:, :, :, :-1], P[:, :, :, 1:]
        yield P[:, :, :-1, :], P[:, :, 1:, :]

    def forward(self, logits: torch.Tensor) -> dict[str, torch.Tensor]:
        P = logits.float().softmax(1)
        num = P.new_zeros(())
        bmass = P.new_zeros(())
        for A, B in self._pairs(P):
            Q = torch.einsum("bchw,cd->bdhw", B, self.W)      # Q_d(x) = sum_c W[c,d] P_c(x+d)  (W 対称)
            num = num + (A * Q).sum()
            bmass = bmass + (1.0 - (A * B).sum(1)).sum()      # soft 境界質量
        loss_adj = num / (bmass.detach() + self.eps)
        s = F.avg_pool2d(P, self.pool).amax(dim=(2, 3))        # (B,C) soft presence
        loss_excl = torch.einsum("bi,ij,bj->b", s, self.E, s).mean()
        return {"adj": loss_adj, "excl": loss_excl}

    @torch.no_grad()
    def hard_violations(self, pred: torch.Tensor) -> dict[str, float]:
        """argmax 予測 (B,H,W) long のルール違反率 (監視用)。
        adj_rate: 異クラス境界画素対のうち W>0 (重み付き) の割合
        excl_pairs: 1 画像あたりの排他ペア共起数
        """
        C = self.W.shape[0]
        oh = F.one_hot(pred, C).permute(0, 3, 1, 2).float()
        num = 0.0
        den = 0.0
        for A, B in self._pairs(oh):
            Q = torch.einsum("bchw,cd->bdhw", B, self.W)
            num += (A * Q).sum().item()
            den += (1.0 - (A * B).sum(1)).sum().item()
        s = (oh.sum(dim=(2, 3)) > 0).float()
        excl = torch.einsum("bi,ij,bj->b", s, self.E, s).mean().item()
        return {"adj_rate": num / max(den, 1.0), "excl_pairs": excl}
