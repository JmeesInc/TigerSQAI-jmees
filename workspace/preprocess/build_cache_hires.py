"""既存の元解像度ラベルから **任意解像度の学習キャッシュ**を追加生成する.

build_labels.py が作った `labels_fine/` `labels_coarse/`（元解像度 uint8 ID）を再利用し、
画像とラベルを指定サイズへ縮小するだけ。RGB→ID 変換はやり直さない（決定済みの成果物）。

動機: 現行の学習入力は 1024x576 で、4K の 109 枚は 0.27 倍まで潰れている。
weight=3 の細長い構造（神経・靭帯）は最も解像度の影響を受けるため、
ConvNeXt（完全畳み込み。32 の倍数なら任意サイズ可）で高解像度入力を試す。

Usage: python3 build_cache_hires.py --width 1536 --height 864 [--workers 16]
出力:  workspace/data_proc/{images_1536,labels_fine_1536,labels_coarse_1536}/
"""

from __future__ import annotations

import argparse
import logging
from concurrent.futures import ProcessPoolExecutor
from functools import partial
from pathlib import Path

import cv2

REPO = Path(__file__).resolve().parents[2]
OUT = REPO / "workspace" / "data_proc"

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
log = logging.getLogger("hires")


def convert_one(name: str, w: int, h: int, tag: str) -> tuple[str, tuple[int, int]]:
    img = cv2.imread(str(REPO / "data" / "images" / name))
    assert img is not None, f"missing image {name}"
    orig = (img.shape[1], img.shape[0])
    # 拡大になる 720p はライン補間、縮小は面積平均が最適
    interp = cv2.INTER_AREA if orig[0] >= w else cv2.INTER_LINEAR
    cv2.imwrite(str(OUT / f"images_{tag}" / name), cv2.resize(img, (w, h), interpolation=interp))
    for src, dst in [("labels_fine", f"labels_fine_{tag}"), ("labels_coarse", f"labels_coarse_{tag}")]:
        ids = cv2.imread(str(OUT / src / name), cv2.IMREAD_GRAYSCALE)
        assert ids is not None, f"missing {src}/{name} — 先に build_labels.py を実行すること"
        cv2.imwrite(str(OUT / dst / name), cv2.resize(ids, (w, h), interpolation=cv2.INTER_NEAREST))
    return name, orig


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--width", type=int, required=True)
    ap.add_argument("--height", type=int, required=True)
    ap.add_argument("--workers", type=int, default=16)
    args = ap.parse_args()
    w, h = args.width, args.height
    assert w % 32 == 0 and h % 32 == 0, f"encoder の stride 32 で割り切れない: {w}x{h}"
    assert abs(w / h - 16 / 9) < 1e-6, f"16:9 でない: {w}x{h}"
    tag = str(w)

    for d in (f"images_{tag}", f"labels_fine_{tag}", f"labels_coarse_{tag}"):
        (OUT / d).mkdir(parents=True, exist_ok=True)
    names = sorted(p.name for p in (REPO / "data" / "images").glob("*.png"))
    log.info("%d 枚を %dx%d へ変換 (tag=%s)", len(names), w, h, tag)

    upscaled = 0
    with ProcessPoolExecutor(args.workers) as ex:
        for i, (name, orig) in enumerate(ex.map(partial(convert_one, w=w, h=h, tag=tag), names, chunksize=4)):
            if orig[0] < w:
                upscaled += 1
            if (i + 1) % 100 == 0:
                log.info("  %d/%d", i + 1, len(names))
    log.info("完了: %d 枚 (うち拡大になったもの %d 枚)", len(names), upscaled)
    for d in (f"images_{tag}", f"labels_fine_{tag}", f"labels_coarse_{tag}"):
        n = len(list((OUT / d).glob("*.png")))
        assert n == len(names), f"{d}: {n} != {len(names)}"
    log.info("枚数検証 OK -> %s", OUT)


if __name__ == "__main__":
    main()
