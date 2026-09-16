"""Tiger 自身の器具インスタンスを貼り付ける augmentation.

expA02 は SAR-RARP50 / SurgToolLoc の器具を貼っていたが, 異ドメインの器具分布が
Tiger の器具分布を汚し, 細長い weight=3 構造 (L_RLN -0.29 / L_IPL -0.20) を
壊したと診断されている。ここでは 2 点を変える:

  1. 器具は Tiger の学習画像から切り出す (同ドメイン, 外部データ開示も不要)
  2. weight=3 クラスを一定割合以上隠す配置は棄却する

貼り付け後にラベルは Instrument (fine=1 / coarse=12) で上書きする。
Tiger では器具自体が採点対象クラスであるため, ラベルを消すのではなく書き込む。
"""
from __future__ import annotations

import logging
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

FINE_INSTRUMENT_ID = 1
COARSE_INSTRUMENT_ID = 12
MIN_AREA = 1500


def weight3_fine_ids(labelmap_csv: str | Path) -> set[int]:
    lm = pd.read_csv(labelmap_csv)
    return {int(r.fine_id) for r in lm.itertuples()
            if not pd.isna(r.fine_id) and int(r.weight) == 3}


class InDomainToolPaster:
    def __init__(self, images_dir, labels_fine_dir, labelmap_csv, filenames,
                 p: float = 0.5, max_tools: int = 2, protect_w3: float = 0.15,
                 min_area: int = MIN_AREA, cache_limit: int = 400):
        """protect_w3: 1 つの weight=3 クラスをこの割合より多く隠す配置は棄却する。"""
        self.images_dir = Path(images_dir)
        self.labels_dir = Path(labels_fine_dir)
        self.p = p
        self.max_tools = max_tools
        self.protect_w3 = protect_w3
        self.min_area = min_area
        self.w3 = weight3_fine_ids(labelmap_csv)
        self.pool: list[tuple[np.ndarray, np.ndarray]] = []
        self._build(filenames, cache_limit)

    def _build(self, filenames, limit):
        from scipy import ndimage
        rng = np.random.default_rng(0)
        for fn in rng.permutation(np.asarray(filenames)):
            if len(self.pool) >= limit:
                break
            lp, ip = self.labels_dir / fn, self.images_dir / fn
            if not (lp.exists() and ip.exists()):
                continue
            m = cv2.imread(str(lp), cv2.IMREAD_GRAYSCALE)
            b = m == FINE_INSTRUMENT_ID
            if not b.any():
                continue
            img = cv2.imread(str(ip))
            lab, k = ndimage.label(b)
            for i in range(1, k + 1):
                comp = lab == i
                if comp.sum() < self.min_area:
                    continue
                ys, xs = np.where(comp)
                y0, y1, x0, x1 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
                self.pool.append((img[y0:y1, x0:x1].copy(), comp[y0:y1, x0:x1].copy()))
        log.info("InDomainToolPaster: %d 個の器具インスタンスを %d 枚から収集 (w=3 保護 %.0f%%)",
                 len(self.pool), min(len(filenames), limit), self.protect_w3 * 100)

    def __call__(self, img, fine, coarse):
        if not self.pool or np.random.rand() > self.p:
            return img, fine, coarse
        H, W = fine.shape
        img, fine, coarse = img.copy(), fine.copy(), coarse.copy()
        for _ in range(np.random.randint(1, self.max_tools + 1)):
            patch, mask = self.pool[np.random.randint(len(self.pool))]
            s = np.random.uniform(0.6, 1.3)
            ph, pw = max(8, int(patch.shape[0] * s)), max(8, int(patch.shape[1] * s))
            if ph >= H or pw >= W:
                continue
            p_img = cv2.resize(patch, (pw, ph), interpolation=cv2.INTER_LINEAR)
            p_m = cv2.resize(mask.astype(np.uint8), (pw, ph),
                             interpolation=cv2.INTER_NEAREST).astype(bool)
            if np.random.rand() < 0.5:
                p_img, p_m = p_img[:, ::-1], p_m[:, ::-1]
            for _try in range(8):     # w=3 を壊さない配置が見つかるまで試す
                y = np.random.randint(0, H - ph)
                x = np.random.randint(0, W - pw)
                sub = fine[y:y + ph, x:x + pw]
                ok = True
                for cid in self.w3:
                    tot = int((fine == cid).sum())
                    if tot and int((sub[p_m] == cid).sum()) / tot > self.protect_w3:
                        ok = False
                        break
                if ok:
                    img[y:y + ph, x:x + pw][p_m] = p_img[p_m]
                    fine[y:y + ph, x:x + pw][p_m] = FINE_INSTRUMENT_ID
                    coarse[y:y + ph, x:x + pw][p_m] = COARSE_INSTRUMENT_ID
                    break
        return img, fine, coarse
