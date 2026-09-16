"""Write-up figures. Run from repo root: python3 submit/writeup/make_figures.py
fig1: pipeline diagram (matplotlib). fig2: out-of-fold qualitative examples (candB_new7 OOF, held-out cases).
"""
from pathlib import Path
import cv2, numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

REPO = Path(__file__).resolve().parents[2]
OUT = REPO / "submit/writeup/figures"; OUT.mkdir(exist_ok=True, parents=True)

# ------------------------------------------------------------------ fig1 pipeline
fig, ax = plt.subplots(figsize=(14, 6.2)); ax.set_xlim(0, 14); ax.set_ylim(0, 6.7); ax.axis("off")
def box(x, y, w, h, text, fc="#eef3fb", ec="#3b5b8f", fs=9.5, bold=False):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.12", fc=fc, ec=ec, lw=1.4))
    ax.text(x + w/2, y + h/2, text, ha="center", va="center", fontsize=fs, weight="bold" if bold else "normal", linespacing=1.35)
def arrow(x0, y0, x1, y1, text=None, color="#333"):
    ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle="-|>", mutation_scale=14, lw=1.3, color=color))
    if text: ax.text((x0+x1)/2, (y0+y1)/2 + 0.14, text, ha="center", va="bottom", fontsize=8, color=color)

box(0.2, 4.2, 2.2, 1.3, "Input frame\n(720p / 1080p / 4K)\n→ resize 1024×576\nImageNet norm.", fc="#f7f7f7", ec="#666")
box(3.0, 3.2, 3.6, 2.8, "Segmentation ensemble\n7 recipes · 31 all-data checkpoints\n"
    "ConvNeXt-L/XL + DeepLabV3+ / Unet++\nflip TTA on 5 recipes\n"
    "Dice + DiceDet + AnatomyLoss\n+ fine→merged consistency\n"
    "CholecSeg8k / EndoVis18 pre-training\n2 heads: fine (31) + coarse (16)", fc="#eef3fb", bold=False)
box(7.3, 4.6, 2.6, 1.4, "Softmax averaging\n(seeds within a recipe,\nthen 7 recipes equally)", fc="#fff6e5", ec="#b07b1e")
box(7.3, 2.8, 2.6, 1.4, "Output\n• α on fine head\n• bilinear → original size, argmax\n• islands < 0.4 % of frame removed", fc="#fff6e5", ec="#b07b1e")
box(10.6, 4.6, 3.1, 1.4, "Task 1: /output/task1/*.png\n(merged, 16 RGB colours)", fc="#e8f6ea", ec="#2f7d3a", bold=True)
box(10.6, 2.8, 3.1, 1.4, "Task 2: /output/task2/*.png\n(fine, 31 RGB colours)", fc="#e8f6ea", ec="#2f7d3a", bold=True)
box(3.0, 0.3, 3.6, 2.2, "Task 3 features\n• 1741 mask features (inventory 315,\n  context 202, 6×12 grid 1224)\n• GAP of 5 frozen fold encoders\n  (one DeepLabV3+ recipe)", fc="#f3ecfb", ec="#6b3fa0", fs=9)
box(7.3, 0.3, 2.6, 2.2, "Station heads\n• 15 XGBoost multi-output (1741)\n• 15 MLPs (GAP ⊕ inventory 315)\nblend 0.7 / 0.3\n→ piecewise-linear calibration", fc="#f3ecfb", ec="#6b3fa0", fs=8.6)
box(10.6, 0.7, 3.1, 1.4, "Task 3: /output/task3.csv\n14 station probabilities", fc="#e8f6ea", ec="#2f7d3a", bold=True)
arrow(2.4, 4.85, 3.0, 4.85); arrow(6.6, 5.3, 7.3, 5.3); arrow(8.6, 4.6, 8.6, 4.2)
arrow(9.9, 5.3, 10.6, 5.3); arrow(9.9, 3.5, 10.6, 3.5)
arrow(8.6, 2.8, 8.6, 2.5); arrow(6.6, 1.4, 7.3, 1.4); arrow(9.9, 1.4, 10.6, 1.4)
arrow(4.8, 3.2, 4.8, 2.5, "masks")
ax.text(0.2, 6.6, "TigerSQAI submission pipeline (single Docker container, offline, GPU)", fontsize=12, weight="bold", va="top")
fig.savefig(OUT / "fig1_pipeline.png", dpi=170, bbox_inches="tight"); plt.close(fig)

# ------------------------------------------------------------------ fig2 qualitative (OOF)
lm = pd.read_csv(REPO / "data/labelmap.csv")
fine_cols = set(zip(lm.fine_r, lm.fine_g, lm.fine_b)); coarse_cols = set(zip(lm.merged_r, lm.merged_g, lm.merged_b))
pred_root = REPO / "workspace/expE01_ensemble/results/candB_new7"
def colours(p):
    im = cv2.cvtColor(cv2.imread(str(p)), cv2.COLOR_BGR2RGB); return set(map(tuple, im.reshape(-1, 3)[::501]))
samples = sorted((pred_root / "task1").glob("*.png"))
if not samples: raise SystemExit("fig2 source masks missing; fig1 written")
sample = samples[0]
fine_dir, coarse_dir = ("task1", "task2") if colours(sample) <= fine_cols else ("task2", "task1")
frames = ["center_2_case_5_7R.png", "center_6_case_3_12L.png", "center_3_case_4_10R.png"]
frames = [f for f in frames if (pred_root / fine_dir / f).exists()] or [p.name for p in sorted((pred_root / fine_dir).glob("*.png"))[::200][:3]]
fig, axes = plt.subplots(len(frames), 5, figsize=(20, 2.4 * len(frames)))
titles = ["Input", "Fine GT (Task 2)", "Fine prediction (OOF)", "Coarse GT (Task 1)", "Coarse prediction (OOF)"]
for r, f in enumerate(frames):
    ims = [cv2.imread(str(REPO / "data/images" / f)), cv2.imread(str(REPO / "data/masks_fine" / f)),
           cv2.imread(str(pred_root / fine_dir / f)), cv2.imread(str(REPO / "data/masks_coarse" / f)),
           cv2.imread(str(pred_root / coarse_dir / f))]
    for c, im in enumerate(ims):
        im = cv2.cvtColor(cv2.resize(im, (960, 540), interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2RGB)
        if c in (1, 2, 3, 4):  # blend on image for readability
            base = cv2.cvtColor(cv2.resize(ims[0], (960, 540), interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2RGB)
            im = (0.45 * base + 0.55 * im).astype(np.uint8)
        axes[r, c].imshow(im); axes[r, c].axis("off")
        if r == 0: axes[r, c].set_title(titles[c], fontsize=12)
    axes[r, 0].text(0, -12, f.replace(".png", ""), fontsize=9, color="#333")
fig.suptitle("Out-of-fold predictions of the 7-recipe fold ensemble (cases never seen by the predicting models)", fontsize=13)
fig.tight_layout(); fig.savefig(OUT / "fig2_qualitative_oof.png", dpi=110, bbox_inches="tight"); plt.close(fig)
print("wrote", sorted(p.name for p in OUT.iterdir()), "fine_dir=", fine_dir)
