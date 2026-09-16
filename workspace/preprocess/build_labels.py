"""RGB マスク → uint8 クラス ID ラベルへの一括変換 + 学習解像度キャッシュ生成.

入力 (read-only):
    data/labelmap.csv
    data/images/*.png            (524 枚, 720p/1080p/4K 混在, 全て 16:9)
    data/masks_fine/*.png        (RGB PNG, 31 色)
    data/masks_coarse/*.png      (RGB PNG, 16 色)

出力 (workspace/data_proc/):
    labels_fine/*.png            uint8 fine_id   (元解像度)
    labels_coarse/*.png          uint8 merged_id (元解像度)
    images_1024/*.png            1024x576 BGR→RGB 縮小画像
    labels_fine_1024/*.png       1024x576 nearest 縮小ラベル
    labels_coarse_1024/*.png     1024x576 nearest 縮小ラベル
    class_weights.json           {"fine": [31 weights], "coarse": [16 weights]}
                                 coarse weight = 構成 fine クラスの max
                                 (公式 metrics/classes_merged.py と同一の定義)

未知色ピクセルが 1 つでもあれば AssertionError で落とす。
"""

from __future__ import annotations

import json
import logging
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
DATA = REPO / "data"
OUT = REPO / "workspace" / "data_proc"
TRAIN_W, TRAIN_H = 1024, 576
N_WORKERS = 8

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
log = logging.getLogger(__name__)


def _pack(rgb: np.ndarray) -> np.ndarray:
    """(H,W,3) uint8 → (H,W) uint32 packed RGB."""
    rgb = rgb.astype(np.uint32)
    return (rgb[..., 0] << 16) | (rgb[..., 1] << 8) | rgb[..., 2]


def build_luts(labelmap: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """packed RGB (2^24) → class id の LUT。未知色は 255。"""
    lut_fine = np.full(1 << 24, 255, dtype=np.uint8)
    lut_coarse = np.full(1 << 24, 255, dtype=np.uint8)
    for _, r in labelmap.iterrows():
        lut_fine[(int(r.fine_r) << 16) | (int(r.fine_g) << 8) | int(r.fine_b)] = int(r.fine_id)
        lut_coarse[(int(r.merged_r) << 16) | (int(r.merged_g) << 8) | int(r.merged_b)] = int(r.merged_id)
    return lut_fine, lut_coarse


def convert_one(args: tuple[str, str, str]) -> str:
    """1 サンプル分 (fine/coarse マスク変換 + 3 種の縮小キャッシュ)。"""
    name, lut_fine_path, lut_coarse_path = args
    lut_fine = np.load(lut_fine_path)
    lut_coarse = np.load(lut_coarse_path)

    for mask_dir, lut, out_full, out_small in [
        ("masks_fine", lut_fine, OUT / "labels_fine", OUT / "labels_fine_1024"),
        ("masks_coarse", lut_coarse, OUT / "labels_coarse", OUT / "labels_coarse_1024"),
    ]:
        bgr = cv2.imread(str(DATA / mask_dir / name), cv2.IMREAD_COLOR)
        assert bgr is not None, f"failed to read {mask_dir}/{name}"
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        ids = lut[_pack(rgb)]
        n_unknown = int((ids == 255).sum())
        assert n_unknown == 0, f"{mask_dir}/{name}: {n_unknown} unknown-color pixels"
        cv2.imwrite(str(out_full / name), ids)
        small = cv2.resize(ids, (TRAIN_W, TRAIN_H), interpolation=cv2.INTER_NEAREST)
        cv2.imwrite(str(out_small / name), small)

    img = cv2.imread(str(DATA / "images" / name), cv2.IMREAD_COLOR)
    assert img is not None, f"failed to read images/{name}"
    img_small = cv2.resize(img, (TRAIN_W, TRAIN_H), interpolation=cv2.INTER_AREA)
    cv2.imwrite(str(OUT / "images_1024" / name), img_small)
    return name


def main() -> None:
    labelmap = pd.read_csv(DATA / "labelmap.csv")

    for d in ["labels_fine", "labels_coarse", "images_1024", "labels_fine_1024", "labels_coarse_1024"]:
        (OUT / d).mkdir(parents=True, exist_ok=True)

    # クラス重み: fine は labelmap の weight 列、coarse は構成 fine の max
    fine_w = labelmap.sort_values("fine_id")["weight"].astype(int).tolist()
    coarse_w = labelmap.groupby("merged_id")["weight"].max().sort_index().astype(int).tolist()
    weights = {"fine": fine_w, "coarse": coarse_w}
    (OUT / "class_weights.json").write_text(json.dumps(weights, indent=2))
    log.info("class_weights.json: fine=%s coarse=%s", fine_w, coarse_w)
    assert len(fine_w) == 31 and len(coarse_w) == 16

    lut_fine, lut_coarse = build_luts(labelmap)
    np.save(OUT / "lut_fine.npy", lut_fine)
    np.save(OUT / "lut_coarse.npy", lut_coarse)

    names = sorted(p.name for p in (DATA / "images").glob("*.png"))
    log.info("converting %d samples with %d workers", len(names), N_WORKERS)
    tasks = [(n, str(OUT / "lut_fine.npy"), str(OUT / "lut_coarse.npy")) for n in names]
    done = 0
    with ProcessPoolExecutor(max_workers=N_WORKERS) as ex:
        for _ in ex.map(convert_one, tasks, chunksize=8):
            done += 1
            if done % 100 == 0:
                log.info("  %d / %d", done, len(names))
    log.info("done: %d samples", done)

    # 検証: 1 枚を ID→RGB 逆変換して元マスクと一致することを確認
    name = names[0]
    ids = cv2.imread(str(OUT / "labels_fine" / name), cv2.IMREAD_GRAYSCALE)
    id2rgb = np.zeros((256, 3), dtype=np.uint8)
    for _, r in labelmap.iterrows():
        id2rgb[int(r.fine_id)] = (int(r.fine_r), int(r.fine_g), int(r.fine_b))
    rgb_back = id2rgb[ids]
    orig = cv2.cvtColor(cv2.imread(str(DATA / "masks_fine" / name)), cv2.COLOR_BGR2RGB)
    assert np.array_equal(rgb_back, orig), f"round-trip mismatch on {name}"
    log.info("round-trip check OK on %s", name)


if __name__ == "__main__":
    sys.exit(main())
