"""公式 HD の高速等価実装 (EDT ベース) と、公式実装へのモンキーパッチ.

公式 `metrics.metrics._directed_hausdorff` は総当たり O(|A|*|B|) で、
4K 画像 (背景クラス = 数百万点) では計算不能なレベルに遅い。
directed Hausdorff h(A→B) = max_{a∈A} EDT_B(a) なので、
scipy の exact Euclidean distance transform で厳密に同じ値を計算できる。

Dice・空集合の規約 (両方空=0.0 / 片方空=1.0)・対角線正規化・集約は
すべて公式コードのまま。差し替えるのは _binary_normalized_hausdorff のみ。

`patch_official_metrics()` はパッチ前にランダムケースで
公式実装との一致 (<1e-6) を assert する。
"""

from __future__ import annotations

import numpy as np
from scipy.ndimage import distance_transform_edt


def fast_binary_normalized_hausdorff(pred: np.ndarray, gt: np.ndarray, diagonal: float) -> float:
    """公式 _binary_normalized_hausdorff と同一値を EDT で計算する."""
    pred_b = pred.astype(bool)
    gt_b = gt.astype(bool)
    pred_any = bool(pred_b.any())
    gt_any = bool(gt_b.any())
    if not pred_any and not gt_any:
        return 0.0
    if not pred_any or not gt_any:
        return 1.0
    # h(P→G) = max_{p∈P} dist(p, G): G の補集合の EDT を P の点で max
    h_pg = float(distance_transform_edt(~gt_b)[pred_b].max())
    h_gp = float(distance_transform_edt(~pred_b)[gt_b].max())
    return float(min(max(h_pg, h_gp) / diagonal, 1.0))


def _verify_equivalence(n_cases: int = 20, seed: int = 0) -> None:
    """ランダムな小マスクで公式実装と突き合わせる (パッチ前の安全確認)."""
    import metrics.metrics as M

    rng = np.random.default_rng(seed)
    for _ in range(n_cases):
        h, w = int(rng.integers(20, 80)), int(rng.integers(20, 80))
        diagonal = float(np.hypot(h, w))
        # 空 / 疎 / 密 を混ぜる
        p_thr, g_thr = rng.uniform(0.85, 1.01, size=2)
        pred = rng.random((h, w)) > p_thr
        gt = rng.random((h, w)) > g_thr
        ref = M._binary_normalized_hausdorff(pred, gt, diagonal)
        fast = fast_binary_normalized_hausdorff(pred, gt, diagonal)
        assert abs(ref - fast) < 1e-6, f"HD mismatch: official={ref} fast={fast}"


def patch_official_metrics() -> None:
    """等価性を確認してから metrics.metrics にパッチを当てる.

    呼び出し前に reference/tigersqai_challenge が sys.path に入っていること。
    """
    import metrics.metrics as M

    _verify_equivalence()
    M._binary_normalized_hausdorff = fast_binary_normalized_hausdorff
