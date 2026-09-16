"""器具貼り付け augmentation (Tiger 適応版).

原案: 自前の ToolPaste 実装 (augment.py)
cutouts: 同ディレクトリの tool_instances/instances.csv
         (SAR-RARP50 + SurgToolLoc 由来。いずれも公開データセット — write-up で要開示)

v10.41 との違い (Tiger 適応):
- ラベルをゼロにするのではなく **Instrument クラスを書き込む**
  (fine=1 / coarse=12。Tiger では器具自体が採点対象クラスのため)
- albumentations transform ではなく Dataset.__getitem__ 内で
  albumentations の前に呼ぶ素朴な関数にした (masks が fine/coarse の2枚あるため。
  後段の幾何 aug は貼り付け後の画像+マスクに一貫して掛かる)
- 配置は「エッジのアンカー点 → 解剖組織上の狙い点」を base→tip に写す相似変換。
  tip が必ず解剖組織上に乗る (v10.41 の overlap 保証の簡略化)

キャンバスは学習解像度 (1024x576) 前提。
"""

from __future__ import annotations

import logging
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch.utils.data

log = logging.getLogger(__name__)

DEFAULT_INSTANCES = Path("external/tool_instances/instances.csv")  # SAR-RARP50 / SurgToolLoc から切り出した器具 RGBA
FINE_INSTRUMENT_ID = 1
COARSE_INSTRUMENT_ID = 12
MAX_CUTOUT_DIM = 768  # ロード時にこの長辺まで縮小 (メモリ節約。貼り付け時に再スケールされる)


class ToolPaster:
    def __init__(
        self,
        instances_csv: str | Path = DEFAULT_INSTANCES,
        p: float = 0.5,
        max_tools: int = 2,
        seed: int | None = None,
    ):
        self.p = p
        self.max_tools = max_tools
        self.rng = np.random.default_rng(seed)
        self._rng_worker: int | None = None
        self.cutouts: list[dict] = []
        df = pd.read_csv(instances_csv)
        for row in df.itertuples():
            rgba = cv2.imread(str(row.path), cv2.IMREAD_UNCHANGED)
            if rgba is None or rgba.ndim != 3 or rgba.shape[2] != 4 or rgba[..., 3].max() == 0:
                continue
            rgba = cv2.cvtColor(rgba, cv2.COLOR_BGRA2RGBA)
            h, w = rgba.shape[:2]
            s = MAX_CUTOUT_DIM / max(h, w)
            if s < 1.0:
                rgba = cv2.resize(rgba, (int(w * s), int(h * s)), interpolation=cv2.INTER_AREA)
                h, w = rgba.shape[:2]
            self.cutouts.append(
                {
                    "rgba": rgba,
                    "base": np.array([row.base_rx * w, row.base_ry * h], dtype=np.float64),
                    "tip": np.array([row.tip_rx * w, row.tip_ry * h], dtype=np.float64),
                }
            )
        if not self.cutouts:
            log.warning("ToolPaster: no cutouts loaded from %s -> no-op", instances_csv)
        else:
            log.info("ToolPaster: loaded %d cutouts", len(self.cutouts))

    @staticmethod
    def _similarity(src0: np.ndarray, src1: np.ndarray, dst0: np.ndarray, dst1: np.ndarray) -> np.ndarray:
        """2点対応 (src0->dst0, src1->dst1) の相似変換 2x3 行列."""
        sv = src1 - src0
        dv = dst1 - dst0
        ls, ld = np.hypot(*sv), np.hypot(*dv)
        if ls < 1e-6:
            ls = 1e-6
        s = ld / ls
        ang = np.arctan2(dv[1], dv[0]) - np.arctan2(sv[1], sv[0])
        ca, sa = s * np.cos(ang), s * np.sin(ang)
        R = np.array([[ca, -sa], [sa, ca]])
        t = dst0 - R @ src0
        return np.hstack([R, t[:, None]])

    def __call__(
        self, img: np.ndarray, fine: np.ndarray, coarse: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """img: (H,W,3) uint8 RGB / fine, coarse: (H,W) uint8。破壊的変更はしない。"""
        # DataLoader worker は fork で RNG 状態ごと複製される → worker 毎に一度だけ再シード
        # (numpy global は pl.seed_everything(workers=True) が worker 毎に別シードを設定済み)
        info = torch.utils.data.get_worker_info()
        wid = info.id if info is not None else -1
        if wid != self._rng_worker:
            self._rng_worker = wid
            self.rng = np.random.default_rng(np.random.randint(0, 2**31 - 1))
        if not self.cutouts or self.rng.random() >= self.p:
            return img, fine, coarse
        H, W = fine.shape
        anat_ys, anat_xs = np.where(fine >= 3)  # id 0,1,2 = Non-anatomical
        img = img.copy()
        fine = fine.copy()
        coarse = coarse.copy()
        # レターボックス黒帯の検出: 器具が黒帯に乗るのは物理的にあり得ないので
        # コンテンツ bbox 内にのみ貼り付ける
        gray = img.mean(axis=2)
        col_ok = np.where((gray > 12).mean(axis=0) > 0.02)[0]
        row_ok = np.where((gray > 12).mean(axis=1) > 0.02)[0]
        if len(col_ok) > 0 and len(row_ok) > 0:
            x0, x1 = int(col_ok[0]), int(col_ok[-1]) + 1
            y0, y1 = int(row_ok[0]), int(row_ok[-1]) + 1
        else:
            x0, y0, x1, y1 = 0, 0, W, H
        cw, ch = x1 - x0, y1 - y0
        n = int(self.rng.integers(1, self.max_tools + 1))
        for _ in range(n):
            cut = self.cutouts[int(self.rng.integers(len(self.cutouts)))]
            for _try in range(3):
                # 狙い点: 解剖組織上のランダム画素 (無ければ中央付近)
                if len(anat_xs) > 0:
                    j = int(self.rng.integers(len(anat_xs)))
                    aim = np.array([anat_xs[j], anat_ys[j]], dtype=np.float64)
                else:
                    aim = np.array([W / 2, H / 2]) + self.rng.uniform(-0.2, 0.2, 2) * (W, H)
                # アンカー: コンテンツ bbox のランダムな辺のわずかに外側
                edge = int(self.rng.integers(4))
                m = 0.03  # bbox 外オフセット率
                if edge == 0:  # left
                    anchor = np.array([x0 - m * cw, self.rng.uniform(y0, y1)])
                elif edge == 1:  # right
                    anchor = np.array([x1 + m * cw, self.rng.uniform(y0, y1)])
                elif edge == 2:  # top
                    anchor = np.array([self.rng.uniform(x0, x1), y0 - m * ch])
                else:  # bottom
                    anchor = np.array([self.rng.uniform(x0, x1), y1 + m * ch])
                # tip の食い込み: aim を少し越えた点まで tip を伸ばす
                overshoot = self.rng.uniform(1.0, 1.25)
                dst_tip = anchor + (aim - anchor) * overshoot
                M = self._similarity(cut["base"], cut["tip"], anchor, dst_tip)
                # スケール暴走の抑制 (画像対角に対して器具が長すぎ/短すぎ)
                s = float(np.hypot(M[0, 0], M[1, 0]))
                tool_len = s * np.hypot(*(cut["tip"] - cut["base"]))
                diag = np.hypot(cw, ch)
                if not (0.15 * diag <= tool_len <= 0.85 * diag):
                    continue
                warped = cv2.warpAffine(
                    cut["rgba"], M, (W, H),
                    flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0,
                )
                alpha = warped[..., 3]
                # コンテンツ bbox 外 (レターボックス・UI 帯) には描かない
                alpha = alpha.copy()
                alpha[:y0, :] = 0
                alpha[y1:, :] = 0
                alpha[:, :x0] = 0
                alpha[:, x1:] = 0
                paste = alpha > 127
                if paste.sum() < 200:  # ほぼ画面外
                    continue
                a = (alpha[..., None].astype(np.float32)) / 255.0
                img[:] = (warped[..., :3].astype(np.float32) * a + img.astype(np.float32) * (1 - a)).astype(np.uint8)
                fine[paste] = FINE_INSTRUMENT_ID
                coarse[paste] = COARSE_INSTRUMENT_ID
                break
        return img, fine, coarse
