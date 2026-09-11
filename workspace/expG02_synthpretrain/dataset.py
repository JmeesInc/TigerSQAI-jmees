"""Dataset: 1024x576 キャッシュ (workspace/data_proc) を読む。マスクは fine/coarse の 2 枚。"""

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


def build_transforms(img_h: int, img_w: int, train: bool) -> A.Compose:
    if train:
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


class TigerDataset(Dataset):
    def __init__(self, df: pd.DataFrame, repo: Path, cfg: dict, train: bool):
        self.df = df.reset_index(drop=True)
        self.images_dir = repo / cfg["paths"]["images_dir"]
        self.fine_dir = repo / cfg["paths"]["labels_fine_dir"]
        self.coarse_dir = repo / cfg["paths"]["labels_coarse_dir"]
        self.tf = build_transforms(cfg["data"]["img_h"], cfg["data"]["img_w"], train)

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, i: int):
        row = self.df.iloc[i]
        img = cv2.cvtColor(cv2.imread(str(self.images_dir / row.filename)), cv2.COLOR_BGR2RGB)
        fine = cv2.imread(str(self.fine_dir / row.filename), cv2.IMREAD_GRAYSCALE)
        coarse = cv2.imread(str(self.coarse_dir / row.filename), cv2.IMREAD_GRAYSCALE)
        out = self.tf(image=img, masks=[fine, coarse])
        return {
            "image": out["image"],
            "fine": torch.from_numpy(np.ascontiguousarray(out["masks"][0])).long(),
            "coarse": torch.from_numpy(np.ascontiguousarray(out["masks"][1])).long(),
            "case_id": row.case_id,
            "filename": row.filename,
        }
