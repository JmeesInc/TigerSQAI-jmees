"""Dataset: 1024x576 キャッシュ (workspace/data_proc) を読む。マスクは fine/coarse の 2 枚。

expS03: 生成クリップ由来の擬似ラベルも同じ形式で読む。擬似ラベルは採用できなかった画素が
255 (ignore) になっているため, valid マスクを一緒に返して loss 側で除外する。
"""

from __future__ import annotations

from pathlib import Path

import albumentations as A
import cv2
import numpy as np
import pandas as pd
import torch
from albumentations.pytorch import ToTensorV2
from torch.utils.data import Dataset

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def build_transforms(img_h: int, img_w: int, train: bool, strength: str = "normal") -> A.Compose:
    """strength: normal = expA01 相当 / strong = 擬似データ併用時の過学習対策版.

    strong では幾何・測光をひと回り強め, さらに Downscale と CoarseDropout を足す。
    Downscale は実画像をわざとボカすことで, 848x480 から拡大した生成フレームとの
    鮮鋭度の差が手がかりにならないようにする狙いもある。
    """
    if train and strength == "strong":
        tfs = [
            A.Resize(img_h, img_w),
            A.HorizontalFlip(p=0.5),
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
            A.HorizontalFlip(p=0.5),
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
                                   cfg["data"].get("aug", "normal"))
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
        return {
            "image": out["image"],
            "fine": torch.from_numpy(np.ascontiguousarray(out["masks"][0])).long(),
            "coarse": torch.from_numpy(np.ascontiguousarray(out["masks"][1])).long(),
            "valid": torch.from_numpy(np.ascontiguousarray(out["masks"][2])).float(),
            "is_pseudo": float(pseudo),
            "case_id": row.case_id,
            "filename": row.filename,
        }
