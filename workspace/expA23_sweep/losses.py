"""expA23: multi-class セグメンテーション loss 動物園.

全ての loss は **同一シグネチャ**を持つ:

    forward(logits (B,C,H,W), target (B,1,H,W) long, valid (B,H,W)|None,
            sample_w (B,)|None) -> scalar

- `valid`: 擬似ラベルの ignore(255) 画素を落とすマスク (1=使う)。expS03 と同じ規約
- `sample_w`: 擬似サンプルの loss 重みを下げるためのサンプル別係数

基準は `MaskedDiceLoss` (= MONAI DiceLoss(softmax, to_onehot_y, include_background,
weight) と数値一致することを expS03/verify_masked_dice.py で検証済み)。
新しい loss はすべて「dice を土台に項を足す」形にして、ベースラインとの差分が
1 項だけになるようにしている。

**公式指標との対応（設計の根拠）**
公式 Dice は per-class で「GT も pred も空 = 1.0 / 片方だけ在 = 0.0」。
つまり **存在しないクラスを 1 画素でも出すとそのクラスが 0 点**になる。
`DetectionHinge` 項はこの規約を直接最適化する (absent クラスは max 確率を閾値以下へ、
present クラスは Dice が閾値を超えたらそれ以上追わない)。
`SizeWeighted` は小面積クラス (= weight 3 の細構造) の per-image 寄与を持ち上げる。
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

EPS_NR, EPS_DR = 1e-5, 1e-5
_RMI_POS_ALPHA = 5e-4


def _prep(logits: torch.Tensor, target: torch.Tensor, valid: torch.Tensor | None):
    """softmax 確率と one-hot GT を作り、valid で両方をマスクする。"""
    p = logits.float().softmax(1)
    t = torch.zeros_like(p).scatter_(1, target.clamp(min=0), 1.0)
    if valid is not None:
        m = valid.unsqueeze(1).float()
        p, t = p * m, t * m
    return p, t


def _reduce(per_bc: torch.Tensor, w: torch.Tensor, sample_w: torch.Tensor | None) -> torch.Tensor:
    """(B,C) の項を クラス重み -> サンプル重み の順に畳む。"""
    per_bc = per_bc * w.view(1, -1)
    if sample_w is not None:
        per_bc = per_bc * sample_w.view(-1, 1)
        return per_bc.sum() / (sample_w.sum() * per_bc.size(1)).clamp(min=1e-8)
    return per_bc.mean()


class MaskedDiceLoss(nn.Module):
    """expS03 と同一実装 (MONAI DiceLoss と数値一致)。全 loss の土台。"""

    def __init__(self, class_weights, softmax: bool = True):
        super().__init__()
        self.register_buffer("w", torch.tensor(class_weights, dtype=torch.float32))
        self.softmax = softmax

    def forward(self, logits, target, valid=None, sample_w=None, dist=None):
        if self.softmax:
            p, t = _prep(logits, target, valid)
        else:  # 既に確率 (f2c の合算確率など)
            p = logits.float()
            t = torch.zeros_like(p).scatter_(1, target.clamp(min=0), 1.0)
            if valid is not None:
                m = valid.unsqueeze(1).float()
                p, t = p * m, t * m
        inter = (p * t).sum((2, 3))
        denom = p.sum((2, 3)) + t.sum((2, 3))
        f = 1.0 - (2.0 * inter + EPS_NR) / (denom + EPS_DR)
        return _reduce(f, self.w, sample_w)


class _DiceBase(nn.Module):
    """dice を土台に追加項を足す loss の共通部分。"""

    def __init__(self, class_weights, extra_weight: float = 1.0):
        super().__init__()
        self.register_buffer("w", torch.tensor(class_weights, dtype=torch.float32))
        self.dice = MaskedDiceLoss(class_weights)
        self.extra_weight = float(extra_weight)

    def extra(self, logits, target, valid, sample_w, dist) -> torch.Tensor:
        raise NotImplementedError

    def forward(self, logits, target, valid=None, sample_w=None, dist=None):
        return self.dice(logits, target, valid, sample_w) + self.extra_weight * self.extra(
            logits, target, valid, sample_w, dist
        )


def _pixel_reduce(px: torch.Tensor, valid, sample_w) -> torch.Tensor:
    """(B,H,W) の画素別 loss を valid / sample_w で畳む。"""
    m = torch.ones_like(px) if valid is None else valid.float()
    if sample_w is not None:
        m = m * sample_w.view(-1, 1, 1)
    return (px * m).sum() / m.sum().clamp(min=1e-8)


class DiceCE(_DiceBase):
    def __init__(self, class_weights, extra_weight: float = 1.0, **_):
        super().__init__(class_weights, extra_weight)

    def extra(self, logits, target, valid, sample_w, dist=None):
        px = F.cross_entropy(logits.float(), target.squeeze(1).clamp(min=0),
                             weight=self.w, reduction="none")
        return _pixel_reduce(px, valid, sample_w)


class DiceFocal(_DiceBase):
    def __init__(self, class_weights, extra_weight: float = 1.0, gamma: float = 2.0, **_):
        super().__init__(class_weights, extra_weight)
        self.gamma = float(gamma)

    def extra(self, logits, target, valid, sample_w, dist=None):
        ce = F.cross_entropy(logits.float(), target.squeeze(1).clamp(min=0),
                             weight=self.w, reduction="none")
        with torch.no_grad():
            pt = torch.exp(-ce / self.w[target.squeeze(1).clamp(min=0)].clamp(min=1e-8))
        return _pixel_reduce((1.0 - pt).pow(self.gamma) * ce, valid, sample_w)


class DiceDetection(_DiceBase):
    """公式規約 (absent クラスの誤検出 = そのクラス 0 点) を直接叩く hinge 項.

    present: relu(thr - dice_c)  … 閾値を超えたらそれ以上 Dice を追わない
    absent : relu(maxprob_c - thr) … GT に無いクラスの最大確率を閾値以下へ押し下げる
    """

    def __init__(self, class_weights, extra_weight: float = 1.0, threshold: float = 0.5, **_):
        super().__init__(class_weights, extra_weight)
        self.thr = float(threshold)

    def extra(self, logits, target, valid, sample_w, dist=None):
        p, t = _prep(logits, target, valid)
        inter = (p * t).sum((2, 3))
        denom = p.sum((2, 3)) + t.sum((2, 3))
        dice = (2.0 * inter + EPS_NR) / (denom + EPS_DR)          # (B,C)
        maxp = p.amax((2, 3))                                      # (B,C)
        present = t.sum((2, 3)) > 0
        hinge = torch.where(present,
                            F.relu(self.thr - dice),
                            F.relu(maxp - self.thr))
        return _reduce(hinge, self.w, sample_w)


class SizeWeightedDice(nn.Module):
    """小面積クラスほど per-image の重みを上げた dice.

    w_size = (1/area_ratio)^alpha を **present なクラスだけ**に掛け、
    バッチ内平均が 1 になるよう正規化して loss スケールを保つ。
    absent クラスは neg_weight 固定 (= 誤検出ペナルティの強さを別に決められる)。
    """

    def __init__(self, class_weights, alpha: float = 0.5, neg_weight: float = 1.0, **_):
        super().__init__()
        self.register_buffer("w", torch.tensor(class_weights, dtype=torch.float32))
        self.alpha, self.neg_weight = float(alpha), float(neg_weight)

    def forward(self, logits, target, valid=None, sample_w=None, dist=None):
        p, t = _prep(logits, target, valid)
        area = t.sum((2, 3))                                       # (B,C)
        n_pix = t.shape[2] * t.shape[3]
        inter = (p * t).sum((2, 3))
        denom = p.sum((2, 3)) + t.sum((2, 3))
        f = 1.0 - (2.0 * inter + EPS_NR) / (denom + EPS_DR)
        present = area > 0
        s = torch.full_like(f, self.neg_weight)
        if present.any():
            ratio = (area[present] / n_pix).clamp(min=1e-6)
            sw = ratio.pow(-self.alpha)
            s[present] = sw / sw.mean()
        return _reduce(f * s, self.w, sample_w)


class FocalTversky(nn.Module):
    """FN を FP より重く罰する (beta>alpha) → 小さい構造の取りこぼしを減らす。"""

    def __init__(self, class_weights, alpha: float = 0.3, beta: float = 0.7,
                 gamma: float = 0.75, **_):
        super().__init__()
        self.register_buffer("w", torch.tensor(class_weights, dtype=torch.float32))
        self.a, self.b, self.g = float(alpha), float(beta), float(gamma)

    def forward(self, logits, target, valid=None, sample_w=None, dist=None):
        p, t = _prep(logits, target, valid)
        tp = (p * t).sum((2, 3))
        fp = (p * (1 - t)).sum((2, 3))
        fn = ((1 - p) * t).sum((2, 3))
        ti = (tp + EPS_NR) / (tp + self.a * fp + self.b * fn + EPS_DR)
        return _reduce((1.0 - ti).pow(self.g), self.w, sample_w)


class DiceHausdorffDT(_DiceBase):
    """dice + MONAI HausdorffDTLoss。公式スコアの半分は正規化 Hausdorff なので、
    そこを直接下げに行く唯一の項 (距離変換が入るぶん重い)。"""

    def __init__(self, class_weights, extra_weight: float = 0.1, alpha: float = 2.0,
                 downsample: int = 4, **_):
        super().__init__(class_weights, extra_weight)
        from monai.losses import HausdorffDTLoss
        self.hd = HausdorffDTLoss(include_background=True, to_onehot_y=True,
                                  softmax=True, alpha=float(alpha), batch=True)
        # 距離変換は CPU 実装で重い (576x1024/31クラスで 15s/step)。
        # HD は形の指標なので 1/4 解像度で十分効く。ここを下げないと 20ep が回らない
        self.ds = int(downsample)

    def extra(self, logits, target, valid, sample_w, dist=None):
        # 距離変換は valid を扱えない。擬似ラベル併用時はこの項を使わない前提。
        if self.ds > 1:
            logits = F.avg_pool2d(logits.float(), self.ds)
            target = target[:, :, ::self.ds, ::self.ds]
        return self.hd(logits.float(), target)


class DiceBoundary(_DiceBase):
    """dice + 距離重み付き偽陽性ペナルティ (Kervadec の boundary loss と同型).

        L_bd = Σ_c w_c * mean_x( p_c(x) · φ_c(x) )

    φ_c = 「クラス c の GT 画素までの距離 / 画像対角長」(GT 内は 0、GT に無いクラスは
    全面 1.0)。**公式スコアの半分を占める正規化 Hausdorff は「GT から最も遠い誤検出画素」
    で決まる**ので、境界の精度ではなく「遠くに出た確率」を叩くこの形が HD に効く。
    φ は dataset 側で aug 後のマスクから 1/4 解像度で作って渡す (GPU 上は積和のみ)。
    """

    def __init__(self, class_weights, extra_weight: float = 1.0, **_):
        super().__init__(class_weights, extra_weight)

    def extra(self, logits, target, valid, sample_w, dist=None):
        assert dist is not None, "diceboundary には data.dist_maps.enabled が必要"
        p = logits.float().softmax(1)
        if p.shape[-2:] != dist.shape[-2:]:
            p = F.adaptive_avg_pool2d(p, dist.shape[-2:])
        per_bc = (p * dist.float()).mean((2, 3))
        return _reduce(per_bc, self.w, sample_w)


class DiceRMI(_DiceBase):
    """dice + Region Mutual Information (局所構造の一致を測る).

    参考実装 (自前の binary RMI loss) を
    multi-class へ拡張: softmax 確率と one-hot GT の各クラスを独立チャネルとして扱う。
    """

    def __init__(self, class_weights, extra_weight: float = 0.5, radius: int = 3,
                 pool_stride: int = 4, **_):
        super().__init__(class_weights, extra_weight)
        self.radius, self.stride = int(radius), int(pool_stride)
        self.half_d = self.radius * self.radius

    @staticmethod
    def _pairs(x: torch.Tensor, radius: int) -> torch.Tensor:
        h, w = x.shape[2], x.shape[3]
        nh, nw = h - (radius - 1), w - (radius - 1)
        out = [x[:, :, y:y + nh, x0:x0 + nw] for y in range(radius) for x0 in range(radius)]
        return torch.stack(out, dim=2)

    def extra(self, logits, target, valid, sample_w, dist=None):
        p, t = _prep(logits, target, valid)
        s = self.stride
        if s > 1:
            p = F.max_pool2d(p, kernel_size=s, stride=s)
            t = F.max_pool2d(t, kernel_size=s, stride=s)
        p = p.clamp(min=1e-6)
        t = t.clamp(min=1e-6)
        la = self._pairs(t, self.radius)
        pr = self._pairs(p, self.radius)
        n, c = t.shape[:2]
        la = la.reshape(n, c, self.half_d, -1).double().detach()
        pr = pr.reshape(n, c, self.half_d, -1).double()
        diag = torch.eye(self.half_d, dtype=torch.float64, device=p.device)[None, None]
        la = la - la.mean(3, keepdim=True)
        pr = pr - pr.mean(3, keepdim=True)
        la_cov = la @ la.transpose(2, 3)
        pr_cov = pr @ pr.transpose(2, 3)
        pr_inv = torch.inverse(pr_cov + diag * _RMI_POS_ALPHA)
        la_pr = la @ pr.transpose(2, 3)
        appro = la_cov - (la_pr @ pr_inv) @ la_pr.transpose(-2, -1)
        chol = torch.linalg.cholesky(appro + diag * _RMI_POS_ALPHA)
        rmi = 2.0 * torch.log(torch.diagonal(chol, dim1=-2, dim2=-1) + 1e-8).sum(-1) * 0.5
        rmi = (rmi / float(self.half_d)).float()                  # (B,C)
        # 情報量の下界なので最大化 = 負号を付けて最小化。クラス重みで畳む。
        return _reduce(-rmi, self.w, sample_w)


class DiceDetBoundary(_DiceBase):
    """dicedet の hinge 項 + diceboundary の距離重み付き偽陽性項 (両方 extra_weight 倍).

    存在判定 (dicedet) と遠方誤検出 (boundary) は機序が独立なので足す。
    k3_dicedet_rules_boundary は config 不備で k_rules_boundary と同一だった → こちらが本物。
    """

    def __init__(self, class_weights, extra_weight: float = 1.0, threshold: float = 0.5, **_):
        super().__init__(class_weights, extra_weight)
        self.det = DiceDetection(class_weights, 1.0, threshold)
        self.bd = DiceBoundary(class_weights, 1.0)

    def extra(self, logits, target, valid, sample_w, dist=None):
        return (self.det.extra(logits, target, valid, sample_w)
                + self.bd.extra(logits, target, valid, sample_w, dist))


_REGISTRY = {
    "dice": lambda cw, **kw: MaskedDiceLoss(cw),
    "dicece": DiceCE,
    "dicefocal": DiceFocal,
    "dicedet": DiceDetection,
    "sizeweighted": SizeWeightedDice,
    "focaltversky": FocalTversky,
    "dicehd": DiceHausdorffDT,
    "diceboundary": DiceBoundary,
    "dicedetboundary": DiceDetBoundary,
    "dicermi": DiceRMI,
}


def build_loss(name: str, class_weights, **params) -> nn.Module:
    if name not in _REGISTRY:
        raise ValueError(f"unknown loss: {name} (choices: {sorted(_REGISTRY)})")
    return _REGISTRY[name](class_weights, **params)
