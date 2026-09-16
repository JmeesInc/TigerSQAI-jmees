"""Dataset: 1024x576 キャッシュ (workspace/data_proc) を読む。マスクは fine/coarse の 2 枚。

expS03: 生成クリップ由来の擬似ラベルも同じ形式で読む。擬似ラベルは採用できなかった画素が
255 (ignore) になっているため, valid マスクを一緒に返して loss 側で除外する。

expA23: `data.hflip: false` で水平反転 aug を切れるようにした。クラスには L/R の別
(L_RLN vs R_RLN, R_Vagal_Nerve, L_Inf_Pul_Lig ...) があり, 反転画像に元のラベルを
付けたまま学習すると「左右は見た目から決まらない」と教えることになる。
expA00 以来ずっと有効だったので, 切った方が良いかを 1 本で測る。
"""

from __future__ import annotations

from pathlib import Path

import albumentations as A
import cv2
import numpy as np
import pandas as pd
import torch
from albumentations.pytorch import ToTensorV2
from scipy.ndimage import distance_transform_edt
from torch.utils.data import Dataset

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def build_transforms(img_h: int, img_w: int, train: bool, strength: str = "normal",
                     hflip: bool = True) -> A.Compose:
    """strength: normal = expA01 相当 / strong = 擬似データ併用時の過学習対策版.

    strong では幾何・測光をひと回り強め, さらに Downscale と CoarseDropout を足す。
    Downscale は実画像をわざとボカすことで, 848x480 から拡大した生成フレームとの
    鮮鋭度の差が手がかりにならないようにする狙いもある。
    """
    if train and strength == "strong":
        tfs = [
            A.Resize(img_h, img_w),
            A.HorizontalFlip(p=0.5 if hflip else 0.0),
            A.Affine(scale=(0.7, 1.4), translate_percent=(-0.15, 0.15),
                     rotate=(-45, 45), shear=(-8, 8), p=0.9),
            A.OneOf(
                [
                    A.ElasticTransform(alpha=100, sigma=10),
                    A.GridDistortion(num_steps=6, distort_limit=0.3),
                ],
                p=0.5,
            ),
            A.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.2, p=0.8),
            A.RandomGamma(gamma_limit=(60, 160), p=0.5),
            A.OneOf(
                [
                    A.GaussianBlur(blur_limit=(3, 9)),
                    A.MotionBlur(blur_limit=(3, 13)),
                ],
                p=0.4,
            ),
            A.GaussNoise(std_range=(0.02, 0.12), p=0.4),
            # 実画像を生成フレーム相当までボカし, 解像度差を手がかりにさせない
            A.Downscale(scale_range=(0.6, 0.9), p=0.3),
            A.CoarseDropout(num_holes_range=(1, 6),
                            hole_height_range=(0.05, 0.15),
                            hole_width_range=(0.05, 0.15), p=0.3),
        ]
    elif train:
        # expA01: expA00 の弱 aug から増強 (幾何を強く + 歪み + 測光系を追加)
        tfs = [
            A.Resize(img_h, img_w),  # キャッシュが既に同サイズなら no-op
            A.HorizontalFlip(p=0.5 if hflip else 0.0),
            A.Affine(
                scale=(0.8, 1.25),
                translate_percent=(-0.1, 0.1),
                rotate=(-25, 25),
                p=0.8,
            ),
            A.OneOf(
                [
                    A.ElasticTransform(alpha=60, sigma=8),
                    A.GridDistortion(num_steps=5, distort_limit=0.2),
                ],
                p=0.3,
            ),
            A.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.15, p=0.7),
            A.RandomGamma(gamma_limit=(70, 140), p=0.3),
            A.OneOf(
                [
                    A.GaussianBlur(blur_limit=(3, 7)),
                    A.MotionBlur(blur_limit=(3, 9)),
                ],
                p=0.3,
            ),
            A.GaussNoise(std_range=(0.02, 0.08), p=0.3),
        ]
    else:
        tfs = [A.Resize(img_h, img_w)]
    tfs += [A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD), ToTensorV2()]
    return A.Compose(tfs)


IGNORE = 255


def class_distance_maps(mask: np.ndarray, n_classes: int, ds: int) -> np.ndarray:
    """各クラスの GT までの距離マップ φ (0=GT 内部, 1=対角長以上) を作る.

    boundary loss 用。**GT に存在しないクラスは全面 1.0** にする
    (= そのクラスの確率はどこに出しても最大ペナルティ。公式の「GT に無いクラスを
    出したらそのクラス 0 点」規約と同じ向き)。
    """
    if ds > 1:
        mask = mask[::ds, ::ds]
    h, w = mask.shape
    diag = float(np.hypot(h, w))
    out = np.ones((n_classes, h, w), dtype=np.float32)
    present = np.unique(mask)
    for c in present:
        if c >= n_classes:
            continue
        m = mask == c
        out[c] = np.minimum(distance_transform_edt(~m) / diag, 1.0)
    return out


class TigerDataset(Dataset):
    """df に is_pseudo 列があれば擬似ラベル側のディレクトリから読む。"""

    def __init__(self, df: pd.DataFrame, repo: Path, cfg: dict, train: bool):
        self.df = df.reset_index(drop=True)
        p = cfg["paths"]
        self.dirs = {
            False: (repo / p["images_dir"], repo / p["labels_fine_dir"],
                    repo / p["labels_coarse_dir"]),
            True: (repo / p.get("images_pseudo_dir", p["images_dir"]),
                   repo / p.get("labels_fine_pseudo_dir", p["labels_fine_dir"]),
                   repo / p.get("labels_coarse_pseudo_dir", p["labels_coarse_dir"])),
        }
        self.tf = build_transforms(cfg["data"]["img_h"], cfg["data"]["img_w"], train,
                                   cfg["data"].get("aug", "normal"),
                                   bool(cfg["data"].get("hflip", True)))
        dm = cfg["data"].get("dist_maps", {}) or {}
        self.dist_on = bool(dm.get("enabled", False))
        self.dist_ds = int(dm.get("downsample", 4))
        self.n_fine = int(cfg["model"].get("num_classes_fine", 31))
        self.n_coarse = int(cfg["model"].get("num_classes_coarse", 16))
        self.toolpaster = None
        tp = cfg.get("toolpaste", {})
        if train and tp.get("enabled"):
            from toolpaste_indomain import InDomainToolPaster
            self.toolpaster = InDomainToolPaster(
                images_dir=repo / p["images_dir"],
                labels_fine_dir=repo / p["labels_fine_dir"],
                labelmap_csv=repo / "data/labelmap.csv",
                filenames=df[~df.get("is_pseudo", False).astype(bool)].filename.tolist()
                if "is_pseudo" in df else df.filename.tolist(),
                p=tp.get("p", 0.5), max_tools=tp.get("max_tools", 2),
                protect_w3=tp.get("protect_w3", 0.15))

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, i: int):
        row = self.df.iloc[i]
        pseudo = bool(getattr(row, "is_pseudo", False))
        img_dir, fine_dir, coarse_dir = self.dirs[pseudo]
        img = cv2.cvtColor(cv2.imread(str(img_dir / row.filename)), cv2.COLOR_BGR2RGB)
        fine = cv2.imread(str(fine_dir / row.filename), cv2.IMREAD_GRAYSCALE)
        coarse = cv2.imread(str(coarse_dir / row.filename), cv2.IMREAD_GRAYSCALE)
        # ignore 画素は aug の補間で壊れないよう, valid を別マスクとして一緒に変換する
        valid = ((fine != IGNORE) & (coarse != IGNORE)).astype(np.uint8)
        fine = np.where(fine == IGNORE, 0, fine).astype(np.uint8)
        coarse = np.where(coarse == IGNORE, 0, coarse).astype(np.uint8)
        if self.toolpaster is not None:
            before = fine.copy()
            img, fine, coarse = self.toolpaster(img, fine, coarse)
            # 貼り付けた画素はラベルが確実に Instrument なので, 擬似の ignore を上書きして有効化する
            valid = np.where(fine != before, 1, valid).astype(np.uint8)
        out = self.tf(image=img, masks=[fine, coarse, valid])
        item = {
            "image": out["image"],
            "fine": torch.from_numpy(np.ascontiguousarray(out["masks"][0])).long(),
            "coarse": torch.from_numpy(np.ascontiguousarray(out["masks"][1])).long(),
            "valid": torch.from_numpy(np.ascontiguousarray(out["masks"][2])).float(),
            "is_pseudo": float(pseudo),
            "case_id": row.case_id,
            "filename": row.filename,
        }
        if self.dist_on:
            fm = np.ascontiguousarray(out["masks"][0])
            cm = np.ascontiguousarray(out["masks"][1])
            item["dist_fine"] = torch.from_numpy(
                class_distance_maps(fm, self.n_fine, self.dist_ds))
            item["dist_coarse"] = torch.from_numpy(
                class_distance_maps(cm, self.n_coarse, self.dist_ds))
        return item
