"""Figures for the two 3-minute videos. Run from repo root: python3 submit/writeup/video/make_slide_figures.py
Outputs to submit/writeup/video/figures/. One hue per single-series chart, thin marks, direct labels, recessive axes."""
from pathlib import Path
import matplotlib; matplotlib.use("Agg"); import matplotlib.ticker
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

OUT = Path(__file__).resolve().parent / "figures"; OUT.mkdir(exist_ok=True)
INK, MUTED, GRID, BLUE, PURPLE, RED, GREEN = "#1f2937", "#6b7280", "#e5e7eb", "#3b6fb6", "#6b3fa0", "#b04a4a", "#2f7d3a"
plt.rcParams.update({"font.size": 13, "axes.edgecolor": GRID, "axes.labelcolor": INK, "xtick.color": INK, "ytick.color": INK})

def style(ax):
    for s in ("top", "right", "left"): ax.spines[s].set_visible(False)
    ax.tick_params(length=0); ax.grid(axis="x", color=GRID, lw=0.8); ax.set_axisbelow(True)

# --- V1 fig A: what moved the score (gains that survived to the final configuration)
fig, ax = plt.subplots(figsize=(9.5, 3.9))
items = [("Fine→merged consistency loss (Task 1 Dice, early)", 0.078),
         ("DiceDet + AnatomyLoss (val score, vs Dice only)", 0.025),
         ("Encoder+decoder pre-training on CholecSeg8k / EndoVis18", 0.015),
         ("Recipe-diverse ensemble, 7 recipes vs 2 (mean Dice)", 0.020),
         ("Class-scaled island removal (Task 2 Dice, final ensemble)", 0.008),
         ("Horizontal-flip TTA on 5 recipes (Task 2 Dice)", 0.002)]
items.sort(key=lambda t: t[1])
ax.barh([i[0] for i in items], [i[1] for i in items], color=BLUE, height=0.5)
for i, (_, v) in enumerate(items): ax.text(v + 0.001, i, f"+{v:.3f}", va="center", color=INK, fontsize=12)
ax.set_xlim(0, 0.09); ax.set_xlabel("gain (5-fold CV, official code; metric noted per row)", color=MUTED); style(ax)
ax.set_title("What moved the segmentation score", loc="left", color=INK, fontsize=14, weight="bold")
fig.tight_layout(); fig.savefig(OUT / "v1_score_movers.png", dpi=180, bbox_inches="tight"); plt.close(fig)

# --- V1 fig B: ensemble size vs Task 2 / Task 1 Dice (5-fold CV of the selection)
sizes = [1, 4, 6, 8, 10, 14]; t2 = [0.7185, 0.7310, 0.7264, 0.7293, 0.7321, 0.7247]; t1 = [0.6943, 0.7062, 0.7083, 0.7109, 0.7098, 0.7097]
fig, axes = plt.subplots(1, 2, figsize=(10, 3.6), sharex=True)
for ax, y, name, best in zip(axes, [t1, t2], ["Task 1 (merged) Dice", "Task 2 (fine) Dice"], [3, 1]):
    ax.plot(sizes, y, color=BLUE, lw=2, marker="o", ms=7)
    for x_, y_ in zip(sizes, y): ax.text(x_, y_ + 0.0015, f"{y_:.4f}", ha="center", fontsize=10, color=INK)
    ax.set_title(name, loc="left", color=INK, weight="bold"); ax.set_xticks(sizes); ax.set_xlabel("recipes in the ensemble (added in single-model rank order)", color=MUTED, fontsize=10)
    for s in ("top", "right"): ax.spines[s].set_visible(False)
    ax.grid(axis="y", color=GRID, lw=0.8); ax.set_axisbelow(True); ax.tick_params(length=0)
    ax.set_ylim(min(y) - 0.006, max(y) + 0.008)
axes[1].annotate("adding weaker recipes\nhurts Task 2", xy=(6, 0.7264), xytext=(7.5, 0.7205), fontsize=10, color=MUTED, arrowprops=dict(arrowstyle="-|>", color=MUTED))
fig.suptitle("Ensemble size vs official Dice (5-fold CV, fold models, before α and island removal)", x=0.01, ha="left", fontsize=12, color=INK)
fig.tight_layout(); fig.savefig(OUT / "v1_ensemble_size.png", dpi=180); plt.close(fig)

# --- V1 fig C: dual-head architecture + losses
fig, ax = plt.subplots(figsize=(11, 4.2)); ax.set_xlim(0, 11); ax.set_ylim(0, 4.2); ax.axis("off")
def box(x, y, w, h, t, fc="#eef3fb", ec=BLUE, fs=11, bold=False):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.1", fc=fc, ec=ec, lw=1.4))
    ax.text(x + w / 2, y + h / 2, t, ha="center", va="center", fontsize=fs, color=INK, weight="bold" if bold else "normal", linespacing=1.3)
def arr(x0, y0, x1, y1): ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle="-|>", mutation_scale=14, lw=1.3, color=INK))
box(0.2, 1.4, 1.9, 1.4, "Frame\n1024×576", fc="#f7f7f7", ec="#888")
box(2.7, 1.0, 2.6, 2.2, "Shared encoder\nConvNeXt-Large\n(ImageNet-22k →\nCholecSeg8k / EndoVis18\nencoder+decoder pre-training)", bold=False)
box(6.0, 2.5, 2.4, 1.3, "Decoder A\nDeepLabV3+ / Unet++", fc="#fff6e5", ec="#b07b1e")
box(6.0, 0.5, 2.4, 1.3, "Decoder B\nDeepLabV3+ / Unet++", fc="#fff6e5", ec="#b07b1e")
box(9.0, 2.5, 1.9, 1.3, "Task 1\n15 merged classes", fc="#e8f6ea", ec="#2f7d3a", bold=True)
box(9.0, 0.5, 1.9, 1.3, "Task 2\n30 fine classes", fc="#e8f6ea", ec="#2f7d3a", bold=True)
arr(2.1, 2.1, 2.7, 2.1); arr(5.3, 2.4, 6.0, 3.1); arr(5.3, 1.8, 6.0, 1.15); arr(8.4, 3.15, 9.0, 3.15); arr(8.4, 1.15, 9.0, 1.15)
ax.text(5.5, 0.15, "Loss = 0.5·L_fine + 0.5·L_coarse + 0.25·L_fine→coarse consistency  (+ DiceDet hinge, + AnatomyLoss)", ha="center", fontsize=11, color=MUTED)
fig.tight_layout(); fig.savefig(OUT / "v1_dual_head.png", dpi=180); plt.close(fig)

# --- V2 fig A: local patch vs whole-frame inventory (station identification accuracy)
fig, ax = plt.subplots(figsize=(8, 3.2))
labels = ["Chance level", "Local neighbourhood of the\nlymph-node component", "Inventory of the whole frame"]
vals = [0.244, 0.269, 0.344]
ax.barh(labels, vals, color=[GRID, PURPLE, PURPLE], height=0.5)
for i, v in enumerate(vals): ax.text(v + 0.004, i, f"{v:.3f}", va="center", color=INK, fontsize=12)
ax.set_xlim(0, 0.42); ax.set_xlabel("station identification accuracy (case-grouped CV, multi-station frames)", color=MUTED, fontsize=10); style(ax)
ax.set_title("Where the station signal is", loc="left", color=INK, fontsize=14, weight="bold")
fig.tight_layout(); fig.savefig(OUT / "v2_where_signal.png", dpi=180); plt.close(fig)

# --- V2 fig B: feature blocks + two branches
fig, ax = plt.subplots(figsize=(11, 4.6)); ax.set_xlim(0, 11); ax.set_ylim(0, 4.6); ax.axis("off")
box(0.2, 1.5, 2.2, 1.6, "Predicted masks\n(fine + coarse)\nfrom the Task 1/2\nensemble", fc="#f7f7f7", ec="#888")
box(3.0, 3.1, 2.7, 1.2, "Anatomy inventory · 315\narea, presence, centroid,\ncomponents, extent per class", fc="#f3ecfb", ec=PURPLE, fs=10)
box(3.0, 1.7, 2.7, 1.2, "Anatomy context · 202\ncontact matrix, what surrounds\nlymph nodes / fat, proximity", fc="#f3ecfb", ec=PURPLE, fs=10)
box(3.0, 0.3, 2.7, 1.2, "Coarse occupancy grid · 1224\n6 × 12 cells × classes", fc="#f3ecfb", ec=PURPLE, fs=10)
box(6.4, 2.6, 2.2, 1.5, "15 MLPs\nGAP of 5 frozen fold encoders\n⊕ inventory 315", fc="#eef3fb", ec=BLUE, fs=10)
box(6.4, 0.6, 2.2, 1.5, "15 XGBoost multi-output\ntrees, 1741 features\n(one tree, 14-d leaf)", fc="#eef3fb", ec=BLUE, fs=10)
box(9.2, 1.5, 1.7, 1.6, "0.3 MLP / 0.7 XGB\n→ per-station\ncalibration\n→ task3.csv", fc="#e8f6ea", ec="#2f7d3a", bold=True, fs=10)
for y in (3.7, 2.3, 0.9): arr(2.4, 2.3, 3.0, y)
arr(5.7, 3.7, 6.4, 3.3)
for y in (3.7, 2.3, 0.9): arr(5.7, y, 6.4, 1.35)
arr(8.6, 3.3, 9.2, 2.6); arr(8.6, 1.35, 9.2, 2.0)
ax.text(5.5, 4.45, "Task 3: station visibility from what the segmentation sees — no station name, no file name", ha="center", fontsize=12, color=INK, weight="bold")
fig.tight_layout(); fig.savefig(OUT / "v2_features_branches.png", dpi=180); plt.close(fig)

# --- V2 fig C: results ladder
fig, ax = plt.subplots(figsize=(9, 3.4))
rows = [("Encoder GAP → MLP (image only, earlier baseline)", 0.7070), ("Inventory → LightGBM (masks only)", 0.7560),
        ("MLP on GAP ⊕ inventory", 0.7700), ("MLPs + per-station LightGBM on\n1741 features, blended (earlier image)", 0.7890),
        ("XGBoost multi-output on 1741 features alone", 0.7917), ("Final: 0.7 XGBoost + 0.3 MLP[GAP ⊕ 315],\ncalibrated", 0.8056)]
ys = range(len(rows))
ax.hlines(ys, 0.65, [r[1] for r in rows], color=GRID, lw=1.5)
ax.plot([r[1] for r in rows], list(ys), "o", color=PURPLE, ms=11)
ax.set_yticks(list(ys)); ax.set_yticklabels([r[0] for r in rows])
for i, (_, v) in enumerate(rows): ax.text(v + 0.004, i, f"{v:.3f}", va="center", color=INK, fontsize=12)
ax.set_xlim(0.65, 0.83); ax.set_xlabel("weighted F1 @ 0.5 (5-fold OOF, 518 frames, official code)", color=MUTED, fontsize=10); style(ax)
ax.set_title("Task 3 out-of-fold results", loc="left", color=INK, fontsize=14, weight="bold")
fig.tight_layout(); fig.savefig(OUT / "v2_results.png", dpi=180); plt.close(fig)
print(sorted(p.name for p in OUT.iterdir()))

# ---------------------------------------------------------------- "what did not work" charts
def neg_chart(items, title, xlabel, fname, color):
    items = sorted(items, key=lambda t: t[1])
    fig, ax = plt.subplots(figsize=(9.5, 0.42 * len(items) + 1.4))
    ys = range(len(items))
    ax.barh([i[0] for i in items], [i[1] for i in items], color=[GRID if v == 0 else color for _, v in items], height=0.55)
    for i, (_, v) in enumerate(items):
        ax.text(v - 0.002 if v < 0 else 0.002, i, f"{v:+.3f}" if v else "±0.000", va="center", ha="right" if v < 0 else "left", color=INK, fontsize=11)
    ax.axvline(0, color=INK, lw=1); ax.set_xlim(min(v for _, v in items) - 0.02, 0.03)
    ax.set_xlabel(xlabel, color=MUTED, fontsize=10); style(ax); ax.grid(axis="x", color=GRID, lw=0.8)
    ax.set_title(title, loc="left", color=INK, fontsize=14, weight="bold")
    ax.set_xticks([-0.10, -0.05, 0.0]); ax.xaxis.set_major_formatter(matplotlib.ticker.FormatStrFormatter("%.2f"))
    fig.tight_layout(); fig.savefig(OUT / fname, dpi=180, bbox_inches="tight"); plt.close(fig)

neg_chart([("Generated pseudo-labels (Wan2.1 video + SAM3 propagation)", -0.016),
           ("Frozen DINOv3-7B feature branch (with f2c loss)", -0.005),
           ("SurgeNetXL surgical encoder (CAFormer-S18)", -0.056),
           ("CAFormer-M36 encoder", -0.027),
           ("DPT-Large (ViT) decoder", -0.092),
           ("Random-init UPerNet decoder", -0.067),
           ("Dice + cross-entropy", -0.030),
           ("Size-weighted Dice", -0.031),
           ("Anatomy rules as post-processing", 0.0),
           ("SWA / EMA checkpoint averaging", -0.006),
           ("Multi-task with Task 3 head (segmentation side)", -0.032),
           ("Boundary loss on Swin/UPerNet (loss gains do not transfer)", -0.054),
           ("Adding recipes beyond the top group (Task 2, 6 → 14 members)", -0.006),
           ("Island removal at 0.4 % of frame: Task 1 HD +0.022 (Dice +0.005)", -0.017),
           ("Class-wise probability scaling α on the final ensemble (Task 2 Dice−HD)", -0.007)],
          "What did not work (Tasks 1 & 2)", "Δ validation Dice vs. baseline (≥ 2 folds each)",
          "v1_did_not_work.png", "#b04a4a")

neg_chart([("Classify the local lymph-node neighbourhood (0.269 vs chance 0.244)", -0.075),
           ("Multi-task Task 3 head inside the segmentation net (AUROC)", -0.016),
           ("Add 1536-d encoder features to LightGBM (F1)", -0.021),
           ("Logistic regression instead of LightGBM (F1)", -0.027),
           ("Stacking the CNN probability as a feature (F1 vs blend)", -0.014),
           ("Per-station LightGBM instead of multi-output XGBoost (F1+AUROC)", -0.020),
           ("1741 features into the MLP instead of 315 (AUROC)", -0.017),
           ("29 XGBoost hyper-parameter variants (all within seed noise)", 0.0)],
          "What did not work (Task 3)", "Δ in the stated metric vs. the alternative we kept",
          "v2_did_not_work.png", "#b04a4a")
print("neg charts done")

# --- V1 fig: island-removal threshold sweep on the final ensemble
rows = [("none", 0.7134, 0.2021, 0.7371, 0.1876), ("0.05 % of frame", 0.7177, 0.2043, 0.7429, 0.1869),
        ("0.1 % of frame", 0.7194, 0.2067, 0.7457, 0.1875), ("0.4 % of frame (old)", 0.7188, 0.2242, 0.7465, 0.1974),
        ("1 % of class size", 0.7179, 0.2029, 0.7427, 0.1851), ("2 % of class size (final)", 0.7193, 0.2053, 0.7453, 0.1848)]
fig, axes = plt.subplots(1, 2, figsize=(11, 3.6))
for ax, (di, hi, title) in zip(axes, ((1, 2, "Task 1 (merged)"), (3, 4, "Task 2 (fine)"))):
    for r in rows:
        c = GREEN if "final" in r[0] else (RED if "old" in r[0] else BLUE)
        ax.plot(r[hi], r[di], "o", color=c, ms=9); ax.annotate(r[0], (r[hi], r[di]), textcoords="offset points", xytext=(6, -3), fontsize=9, color=INK)
    ax.set_xlabel("normalised Hausdorff (lower is better)", color=MUTED); ax.set_ylabel("weighted Dice", color=MUTED)
    ax.set_title(title, loc="left", color=INK, weight="bold"); ax.invert_xaxis()
    for s_ in ("top", "right"): ax.spines[s_].set_visible(False)
    ax.grid(color=GRID, lw=0.8); ax.set_axisbelow(True); ax.tick_params(length=0)
fig.suptitle("Island removal on the final ensemble: Dice vs Hausdorff (5-fold OOF, official code)", x=0.01, ha="left", fontsize=12, color=INK)
fig.tight_layout(); fig.savefig(OUT / "v1_island_sweep.png", dpi=180, bbox_inches="tight"); plt.close(fig)

# ---------------------------------------------------------------- progression charts (official OOF, Task 1 / Task 2 Dice)
def progression(steps, title, fname, color, ylabel):
    fig, ax = plt.subplots(figsize=(12.5, 4.4))
    xs = range(len(steps))
    for key, lab, ls in (("t1", "Task 1 (merged)", "--"), ("t2", "Task 2 (fine)", "-")):
        ys = [s[key] for s in steps]
        if all(y is None for y in ys): continue
        ax.plot(list(xs), ys, ls, color=color, lw=2, marker="o", ms=7, label=lab)
        for x, y in zip(xs, ys):
            if y is not None: ax.text(x, y + 0.004, f"{y:.3f}", ha="center", fontsize=9.5, color=INK)
    ax.set_xticks(list(xs)); ax.set_xticklabels([s["name"] for s in steps], fontsize=9)
    for s_ in ("top", "right"): ax.spines[s_].set_visible(False)
    ax.grid(axis="y", color=GRID, lw=0.8); ax.set_axisbelow(True); ax.tick_params(length=0)
    ax.set_ylabel(ylabel, color=MUTED); ax.set_title(title, loc="left", color=INK, fontsize=14, weight="bold")
    if any(s["t1"] is not None for s in steps): ax.legend(frameon=False, loc="lower right")
    fig.tight_layout(); fig.savefig(OUT / fname, dpi=180, bbox_inches="tight"); plt.close(fig)

progression([
    dict(name="EfficientNet-B7\nUnet++ (start)", t1=0.5759, t2=0.6041),
    dict(name="+ stronger\naugmentation", t1=0.5792, t2=0.6245),
    dict(name="MaxViT-B Unet++\n1024×576", t1=0.5887, t2=0.6342),
    dict(name="+ fine→coarse\nconsistency loss", t1=0.6663, t2=0.6530),
    dict(name="5-recipe\nensemble", t1=0.6731, t2=0.6820),
    dict(name="ConvNeXt-L,\nDiceDet+AnatomyLoss\n7-recipe ensemble", t1=0.6900, t2=0.7007),
    dict(name="CV-selected\n7 recipes\n+ flip TTA", t1=0.7134, t2=0.7371),
    dict(name="+ class-scaled\nisland removal\n(final)", t1=0.7193, t2=0.7453),
], "Segmentation: official weighted Dice over the project (5-fold OOF)", "v1_progression.png", BLUE, "weighted Dice")

t3_steps = [("Frozen encoder\nfeatures → MLP", 0.7070, 0.8867), ("Anatomy inventory\n→ LightGBM", 0.7560, 0.8982),
            ("+ inventory into\nthe MLP head", 0.7700, 0.9015), ("3 encoders × 3 seeds\n+ 6 LightGBM, blended", 0.7765, 0.9145),
            ("+ context + grid\nfeatures (LightGBM image)", 0.7890, 0.9165), ("XGBoost multi-output\n+ MLP[GAP ⊕ 315] (final)", 0.8056, 0.9253)]
fig, axes = plt.subplots(2, 1, figsize=(11, 6.4))
for ax, idx, lab in ((axes[0], 1, "weighted F1 @ 0.5"), (axes[1], 2, "AUROC")):
    ys = [t[idx] for t in t3_steps]; xs = list(range(len(t3_steps)))
    ax.plot(xs, ys, "-", color=PURPLE, lw=2, marker="o", ms=7)
    for x, y in zip(xs, ys): ax.text(x, y + (0.002 if idx == 1 else 0.001), f"{y:.3f}", ha="center", fontsize=9.5, color=INK)
    ax.set_xticks(xs); ax.set_xticklabels([t[0] for t in t3_steps], fontsize=10)
    for s_ in ("top", "right"): ax.spines[s_].set_visible(False)
    ax.grid(axis="y", color=GRID, lw=0.8); ax.set_axisbelow(True); ax.tick_params(length=0)
    ax.set_title(lab, loc="left", color=INK, fontsize=13, weight="bold"); ax.set_ylim(min(ys) - 0.01, max(ys) + 0.012)
fig.suptitle("Task 3 over the project (5-fold OOF, 518 labelled frames, official evaluate_cls)", x=0.01, ha="left", fontsize=12, color=INK)
fig.tight_layout(); fig.savefig(OUT / "v2_progression.png", dpi=180, bbox_inches="tight"); plt.close(fig)
print("progression done")
