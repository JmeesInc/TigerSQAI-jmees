"""Build matched depth conditions for real and synthetic frames.

Real frames have no depth, so they get monocular estimates from Depth-Anything-V2.
Synthetic frames have exact metric depth from Blender. Both are written in the same
convention as the depth ControlNet expects: per-image normalised INVERSE depth,
uint8, near = bright.
"""
import argparse, logging, sys
from pathlib import Path
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]


def to_condition(inv):
    """Per-image min-max on finite values; flat images stay mid-grey rather than NaN."""
    m = np.isfinite(inv)
    if not m.any():
        return np.full(inv.shape, 128, np.uint8)
    lo, hi = np.percentile(inv[m], 1), np.percentile(inv[m], 99)
    if hi - lo < 1e-9:
        return np.full(inv.shape, 128, np.uint8)
    out = np.clip((inv - lo) / (hi - lo), 0, 1)
    out[~m] = 0.0
    return (out * 255).astype(np.uint8)


def synthetic(src, dst, log):
    import OpenEXR, Imath
    dst.mkdir(parents=True, exist_ok=True)
    files = sorted(src.glob('*.exr'))
    for i, f in enumerate(files):
        x = OpenEXR.InputFile(str(f)); h = x.header()
        dw = h['dataWindow']; W = dw.max.x - dw.min.x + 1; H = dw.max.y - dw.min.y + 1
        ch = 'Z' if 'Z' in h['channels'] else list(h['channels'])[0]
        z = np.frombuffer(x.channel(ch, Imath.PixelType(Imath.PixelType.FLOAT)),
                          np.float32).reshape(H, W).astype(np.float64)
        # Metric depth -> inverse depth so that near is bright, matching the estimator.
        inv = np.where(z > 1e-6, 1.0 / z, np.nan)
        Image.fromarray(to_condition(inv), mode='L').save(dst / f'{f.stem}.png')
        if (i + 1) % 250 == 0:
            log.info(f'synthetic {i+1}/{len(files)}')
    log.info(f'synthetic done: {len(files)} -> {dst}')


def real(src, dst, log, batch=8):
    import torch
    from transformers import pipeline
    dst.mkdir(parents=True, exist_ok=True)
    pipe = pipeline('depth-estimation', model='depth-anything/Depth-Anything-V2-Small-hf',
                    device=0 if torch.cuda.is_available() else -1)
    files = sorted(src.glob('*.png'))
    for i in range(0, len(files), batch):
        chunk = files[i:i + batch]
        imgs = [Image.open(f).convert('RGB') for f in chunk]
        outs = pipe(imgs)
        for f, o in zip(chunk, outs):
            # The estimator already returns inverse (relative) depth: near = large.
            d = np.array(o['predicted_depth'].squeeze().cpu().numpy(), np.float64)
            d = np.array(Image.fromarray(d).resize(imgs[0].size, Image.BILINEAR)) \
                if d.shape[::-1] != Image.open(f).size else d
            Image.fromarray(to_condition(d), mode='L').save(dst / f.name)
        if (i + batch) % 80 == 0:
            log.info(f'real {min(i+batch, len(files))}/{len(files)}')
    log.info(f'real done: {len(files)} -> {dst}')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--mode', choices=['real', 'synthetic'], required=True)
    p.add_argument('--src', required=True, type=Path)
    p.add_argument('--dst', required=True, type=Path)
    a = p.parse_args()
    logging.basicConfig(level=logging.INFO, format='%(asctime)s | %(levelname)s | %(message)s',
                        stream=sys.stdout)
    log = logging.getLogger()
    (synthetic if a.mode == 'synthetic' else real)(a.src, a.dst, log)


if __name__ == '__main__':
    main()
