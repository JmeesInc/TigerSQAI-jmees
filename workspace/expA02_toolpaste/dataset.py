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
        tfs = [
            A.Resize(img_h, img_w),  # キャッシュが既に同サイズなら no-op
            A.HorizontalFlip(p=0.5),
            A.Affine(
                scale=(0.9, 1.1),
                translate_percent=(-0.05, 0.05),
                rotate=(-10, 10),
                p=0.5,
            ),
            A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1, p=0.5),
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
        # expA02: 器具貼り付け aug (train のみ。albumentations の前に適用)
        self.toolpaster = None
        tp = cfg.get("toolpaste", {})
        if train and tp.get("enabled", False):
            from toolpaste import ToolPaster
            self.toolpaster = ToolPaster(p=tp.get("p", 0.5), max_tools=tp.get("max_tools", 2))

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, i: int):
        row = self.df.iloc[i]
        img = cv2.cvtColor(cv2.imread(str(self.images_dir / row.filename)), cv2.COLOR_BGR2RGB)
        fine = cv2.imread(str(self.fine_dir / row.filename), cv2.IMREAD_GRAYSCALE)
        coarse = cv2.imread(str(self.coarse_dir / row.filename), cv2.IMREAD_GRAYSCALE)
        if self.toolpaster is not None:
            img, fine, coarse = self.toolpaster(img, fine, coarse)
        out = self.tf(image=img, masks=[fine, coarse])
        return {
            "image": out["image"],
            "fine": torch.from_numpy(np.ascontiguousarray(out["masks"][0])).long(),
            "coarse": torch.from_numpy(np.ascontiguousarray(out["masks"][1])).long(),
            "case_id": row.case_id,
            "filename": row.filename,
        }
