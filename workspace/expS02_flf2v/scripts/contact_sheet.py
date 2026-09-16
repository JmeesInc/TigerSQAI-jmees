"""生成クリップを 1 枚のコンタクトシートにまとめる (目視評価用)."""
import argparse, glob, logging, os, re
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image

LOG = logging.getLogger(__name__)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--clips", default="workspace/expS02_flf2v/outputs/clips")
    ap.add_argument("--out", default="workspace/expS02_flf2v/outputs/contact_sheet.png")
    ap.add_argument("--ncols", type=int, default=9)
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")

    dirs = sorted(d for d in glob.glob(os.path.join(args.clips, "*"))
                  if os.path.isdir(d) and os.path.exists(os.path.join(d, "f000.png")))
    if not dirs:
        LOG.error("no finished clips"); return
    LOG.info("clips=%d", len(dirs))

    meta = {}
    for d in dirs:
        p = os.path.join(d, "meta.txt")
        m = re.search(r"ncc=([\d.]+)", open(p).read()) if os.path.exists(p) else None
        meta[d] = m.group(1) if m else "?"

    nrows, nc = len(dirs), args.ncols
    fig, axes = plt.subplots(nrows, nc, figsize=(nc * 2.3, nrows * 1.55))
    axes = np.atleast_2d(axes)
    for r, d in enumerate(dirs):
        frames = sorted(glob.glob(os.path.join(d, "f*.png")))
        # 両端は GT (入力), 中間は生成された補間フレーム
        idx = np.linspace(0, len(frames) - 1, nc - 2).round().astype(int)
        panels = [("GT first", os.path.join(d, "first.png"))]
        panels += [(f"gen f{i:03d}", frames[i]) for i in idx]
        panels += [("GT last", os.path.join(d, "last.png"))]
        for c, (title, p) in enumerate(panels[:nc]):
            ax = axes[r, c]; ax.imshow(Image.open(p)); ax.axis("off")
            if r == 0:
                ax.set_title(title, fontsize=8)
            for s in ("top", "bottom", "left", "right"):
                ax.spines[s].set_visible(False)
        axes[r, 0].text(-0.06, 0.5, f"{os.path.basename(d)}\nNCC={meta[d]}",
                        transform=axes[r, 0].transAxes, ha="right", va="center", fontsize=8)
    fig.suptitle("Wan2.1-FLF2V  first/last = real annotated frames of two corresponding stations, "
                 "middle = generated  (848x480, 49 frames, 30 steps)", fontsize=11, y=0.995)
    fig.subplots_adjust(left=0.14, right=0.995, top=0.93, bottom=0.005, wspace=0.02, hspace=0.06)
    fig.savefig(args.out, dpi=110, bbox_inches="tight")
    LOG.info("wrote %s", args.out)


if __name__ == "__main__":
    main()
