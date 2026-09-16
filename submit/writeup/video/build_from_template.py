"""Fill the user's template deck (TigerSQAI_task12_jmees.pptx) with the final content, and build the Task 3 deck
from the same template.  Run from repo root:  .venv/bin/python3 submit/writeup/build_from_template.py
Outputs: TigerSQAI_task12_jmees_v2.pptx, TigerSQAI_task3_jmees.pptx  (narration in each slide's notes pane)."""
import copy
from pathlib import Path
from pptx import Presentation
from pptx.util import Inches, Pt
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN
from lxml import etree

HERE = Path(__file__).resolve().parent
TPL = HERE / "TigerSQAI_task12_jmees.pptx"
FIG = HERE / "figures"
ACCENT = RGBColor(0x3B, 0x6F, 0xB6); RED = RGBColor(0xB0, 0x4A, 0x4A); GREEN = RGBColor(0x2F, 0x7D, 0x3A); MUTED = RGBColor(0x6B, 0x72, 0x80)
FONT = "Segoe UI"


def _run(p, text, size, bold=False, color=None, italic=False):
    r = p.add_run(); r.text = text; r.font.size = Pt(size); r.font.bold = bold; r.font.italic = italic; r.font.name = FONT
    if color is not None: r.font.color.rgb = color
    return r


def fill(tf, blocks, size=13):
    """blocks: ('h', text) header | ('b', text[, level]) bullet | ('t', text) plain. **bold** inside text."""
    # wipe existing paragraphs
    for p in list(tf.paragraphs)[1:]:
        p._p.getparent().remove(p._p)
    p0 = tf.paragraphs[0]
    for r in list(p0.runs): r._r.getparent().remove(r._r)
    first = True
    for blk in blocks:
        kind, text = blk[0], blk[1]; level = blk[2] if len(blk) > 2 else 0
        p = p0 if first else tf.add_paragraph(); first = False
        p.space_after = Pt(3)
        if kind == 'h':
            p.space_before = Pt(5); _run(p, text, size + 1, bold=True, color=ACCENT); continue
        if kind == 'b':
            _run(p, ("      " * level) + ("•  " if level == 0 else "–  "), size - level, color=ACCENT)
        parts = text.split("**")
        for j, part in enumerate(parts):
            if part: _run(p, part, size - level, bold=(j % 2 == 1))


def body_box(slide, x=0.45, y=0.87, w=9.2, h=4.4):
    tb = slide.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h)); tb.text_frame.word_wrap = True
    return tb.text_frame


def set_title(slide, text):
    slide.shapes.title.text_frame.text = text


def notes(slide, text):
    slide.notes_slide.notes_text_frame.text = text


def delete_slide(prs, idx):
    sldIdLst = prs.slides._sldIdLst; sld = sldIdLst[idx]
    prs.part.drop_rel(sld.rId); sldIdLst.remove(sld)


def new_slide(prs, title):
    s = prs.slides.add_slide(prs.slide_layouts[1]); set_title(s, title)
    for ph in s.placeholders:
        if ph.placeholder_format.type == 13:  # slide number
            ph.text_frame.text = str(len(prs.slides) - 1)
    return s


def member_table(slide, rows, x=0.45, y=0.9, w=9.2, size=9):
    hdr = ["slot", "encoder", "decoder", "pre-training", "loss / AnatomyLoss graph / note", "checkpoints", "TTA", "CV score"]
    tbl = slide.shapes.add_table(len(rows) + 1, len(hdr), Inches(x), Inches(y), Inches(w), Inches(0.22 * (len(rows) + 1))).table
    widths = [0.45, 1.0, 1.0, 1.45, 2.65, 1.25, 0.5, 0.9]
    for j, wd in enumerate(widths): tbl.columns[j].width = Inches(wd)
    for r_ in tbl.rows: r_.height = Inches(0.2)
    for j, h in enumerate(hdr):
        c = tbl.cell(0, j); c.text = h
        for p in c.text_frame.paragraphs:
            for r in p.runs: r.font.size = Pt(size); r.font.bold = True; r.font.name = FONT
    for i, row in enumerate(rows, 1):
        for j, v in enumerate(row):
            c = tbl.cell(i, j); c.text = str(v)
            for p in c.text_frame.paragraphs:
                for r in p.runs: r.font.size = Pt(size); r.font.name = FONT
    return tbl


MEMBERS = [  # one row per recipe slot (= Appendix A of the write-up). CV = internal val score (mean fine/merged Dice, 5-fold).
    ("A1", "ConvNeXt-L", "DeepLabV3+", "EndoVis18", "DiceDet / 3-D anatomy", "17 seeds", "on", "0.710"),
    ("A2", "ConvNeXt-XL", "DeepLabV3+", "CholecSeg8k", "DiceDet / 3-D anatomy", "seeds 42, 43", "off", "0.710"),
    ("A3", "ConvNeXt-L", "DeepLabV3+", "CholecSeg8k (via base)", "DiceDet / 3-D anatomy; 8-ep fine-specialised fine-tune", "seeds 42, 43", "on", "0.706"),
    ("A4", "ConvNeXt-L", "DeepLabV3+", "CholecSeg8k + EndoVis18", "DiceDet / 3-D anatomy", "seeds 42, 43", "on", "0.709"),
    ("A5", "ConvNeXt-L", "Unet++", "– (ImageNet)", "DiceDet / data-derived graph", "seeds 42, 43, 43', 44", "on", "0.689"),
    ("A6", "ConvNeXt-L", "Unet++", "– (ImageNet)", "DiceDet / –", "seeds 42, 43", "on", "0.676"),
    ("A7", "ConvNeXt-L", "DeepLabV3+", "CholecSeg8k", "DiceDet / 3-D anatomy; no horizontal flip", "seeds 42, 43", "off", "0.710"),
]

# ====================================================================== Tasks 1 & 2 (fill the template in place)
prs = Presentation(TPL)
S = list(prs.slides)
notes(S[0], "Tiger SQ-AI 2026, Tasks 1 and 2, the solution of team Jmees. All numbers are five-fold cross-validation, grouped by case, scored with the official code.")

# slide 2: CV score digest (picture already placed; refresh it with the current figure)
for sh in list(S[1].shapes):
    if sh.shape_type == 13: sh._element.getparent().remove(sh._element)
S[1].shapes.add_picture(str(FIG / "v1_progression.png"), Inches(0.5), Inches(1.0), width=Inches(9.0))
notes(S[1], "The digest. We started from a two-head Unet-plus-plus at fine Dice zero point six zero. Stronger augmentation gave two points, MaxViT one more. The first big step was a consistency loss between the two heads, plus eight points on Task 1. Ensembling, ConvNeXt, metric-aligned losses, surgical pre-training, TTA and class-scaled island removal took us to zero point seven two and zero point seven five.")

# slide 3: basic recipe
tb = [sh for sh in S[2].shapes if sh.has_text_frame and not sh.is_placeholder][0]
fill(tb.text_frame, [
    ('b', "**Encoder:** ConvNeXt-Large (ImageNet-22k→1k, 384) — replaced MaxViT-B (+0.025) and the EfficientNet-B7 baseline"),
    ('b', "**Decoder:** DeepLabV3+ or Unet++; one encoder, **two heads** (fine 31 classes / merged 16 classes)"),
    ('b', "**Loss:** 0.5·Dice_fine + 0.5·Dice_merged + 0.25·fine→merged consistency, official 3/2/1 class weights inside the Dice"),
    ('b', "**Training:** 20 epochs, AdamW 2e-4, wd 0.01, 3 warm-up + cosine, batch 2 × accum 4, AMP; final-epoch weights for all-data models"),
    ('b', "**Preprocessing:** resize to 1024×576, ImageNet normalization; predictions bilinearly upsampled to the original size before argmax"),
    ('b', "**Augmentation:** hflip, affine (scale 0.8–1.25, ±25°), elastic / grid distortion, colour jitter, gamma, blur, noise"),
    ('b', "**Model choice:** 5-fold CV grouped by case, official Dice + Hausdorff at original resolution; single-fold noise σ ≈ 0.007 → decide on ≥ 2 folds; submit all-data-trained models (final epoch), seed replicas averaged inside a recipe"),
    ('b', "**Inference:** 7 recipes × 31 checkpoints, horizontal-flip TTA on 5 recipes, softmax average → upsample → argmax → class-scaled island removal"),
], size=12)
notes(S[2], "The basic recipe: ConvNeXt-Large, which replaced MaxViT for two and a half points, a DeepLabV3-plus or Unet-plus-plus decoder, and two heads on one encoder. Dice on both heads plus a consistency term between them. Twenty epochs, AdamW, cosine, at ten-twenty-four by five-seven-six. Everything is selected on case-grouped five-fold cross-validation with the official code; the submitted models are retrained on all data, with seed replicas averaged inside each recipe.")

# slide 4: what worked 1
tb = [sh for sh in S[3].shapes if sh.has_text_frame and not sh.is_placeholder][0]
fill(tb.text_frame, [
    ('h', "Custom losses"),
    ('b', "**Fine→merged consistency:** fine softmax summed per merged class, Dice against the merged mask → **+0.078 Task 1 / +0.019 Task 2**"),
    ('b', "**DiceDet:** hinge on the *maximum* probability of every class absent from the reference (the metric scores such a class 0) → **+0.011**"),
    ('b', "**AnatomyLoss:** differentiable penalty on forbidden contacts and mutually exclusive structures → **+0.009**; together with DiceDet **+0.025** (super-additive)"),
    ('b', "The rule graph written from 3-D mediastinal anatomy (23 classes, no data statistics) scores ≥ the one estimated from the masks: 0.693 vs 0.689"),
    ('h', "Pre-training weights (public data only)"),
    ('b', "**Encoder + decoder** pre-trained on CholecSeg8k / EndoVis 2018 at 1024×576, transfer all but the heads → **+0.015 on every fold** (0.693 → 0.708–0.710); corpus choice barely matters; encoder-only pre-training had given 0"),
    ('h', "Architecture"),
    ('b', "ConvNeXt-L over MaxViT-B **+0.025**; DeepLabV3+ ≈ Unet++ but with different errors → both kept for the ensemble"),
], size=12)
notes(S[3], "Part one. The consistency loss between the two heads was the largest single gain, plus eight points on Task 1. Two losses target the metric, which gives zero to a class present in only one of prediction and reference: DiceDet, a hinge on the maximum probability of every absent class, and AnatomyLoss, a penalty on impossible contacts from a graph written from anatomy, not from data: plus one each, plus two and a half together. Pre-training encoder and decoder on CholecSeg8k and EndoVis 2018 gave one and a half points on every fold; encoder-only pre-training had given nothing.")

# slide 5: what worked 2
tb = [sh for sh in S[4].shapes if sh.has_text_frame and not sh.is_placeholder][0]
fill(tb.text_frame, [
    ('h', "Ensemble"),
    ('b', "**Diversity > strength:** 2 → 4 → 7 recipes 0.675 → 0.687 → 0.695; the two strongest alone lose by 0.010 to seven ordinary ones; adding recipes outside the pre-trained group hurts Task 2 → 7 recipes (Appendix)"),
    ('b', "**Seed replicas** averaged inside a recipe slot, slots averaged equally (31 checkpoints, 7 slots)"),
    ('b', "**Horizontal-flip TTA** on 5 recipes → Task 2 Dice 0.7350 → 0.7371, HD 0.1888 → 0.1876 (off for the no-flip and the ConvNeXt-XL recipe)"),
    ('h', "Post-processing — validated on Dice **and** Hausdorff"),
    ('b', "**Island removal with a class-scaled threshold:** components < 2 % of the class's mean area → **Task 1 0.7134 → 0.7193, Task 2 0.7371 → 0.7453 Dice**, Task 2 HD 0.1876 → 0.1848, Task 1 HD +0.003"),
    ('b', "A frame-relative threshold (0.4 %) had looked like +0.03 Dice on an earlier ensemble but costs **+0.022 HD** on the final one: official HD = 1.0 for a class present in only one of prediction and reference → deleting a true small weight-3 structure is the worst case"),
    ('h', "Augmentation"),
    ('b', "Stronger geometric + photometric augmentation (early) → +0.020 Task 2"),
], size=12)
notes(S[4], "Part two. The ensemble: diversity beats strength, the two strongest recipes alone lose to seven ordinary ones, but Task 2 drops once recipes outside the pre-trained group are added, so we stop at seven; seeds are averaged inside each recipe, and flip TTA adds a little. Post-processing had to be re-validated on both halves of the metric: islands below a fixed fraction of the frame looked like plus three Dice but cost twice that on Hausdorff, because deleting a true small structure is the metric's worst case. A threshold scaled to each class's size keeps the Dice gain, plus six and plus eight, with Hausdorff unchanged.")

# slide 6: what did not work
tb = [sh for sh in S[5].shapes if sh.has_text_frame and not sh.is_placeholder][0]
fill(tb.text_frame, [
    ('h', "Custom losses"),
    ('b', "Size-weighted Dice −0.031 · Dice + CE −0.030 · focal-Tversky −0.004 · RMI ±0 · Hausdorff-DT (no gain, 8 s/step) · boundary loss +0.004 on ConvNeXt but **−0.054 on Swin** · the same anatomy rules as **post-processing ±0.000**"),
    ('h', "Architecture"),
    ('b', "ConvNeXt-XL ≈ L (−0.018 with DiceDet) · ConvNeXt-V2 · CAFormer −0.027 · SE-ResNeXt −0.09 · SegFormer-B5 −0.01 · DPT-L −0.09 · random-init UPerNet / FPN / MAnet −0.07…−0.16 · ADE20k Swin-UPerNet wins fold 0 only · 1536×864 input −0.011 / −0.028 · 30 / 40 epochs ±0"),
    ('h', "Augmentation / data"),
    ('b', "**Pseudo-labels** (Wan2.1 first-last-frame video between two station views + SAM3 propagation): labels are good (Dice 0.76–0.79 vs 0.57 copy control) but training on them **−0.016** · strong augmentation on the final recipe ±0 · no-hflip +0.010 alone but not additive with DiceDet"),
    ('h', "Pre-training weights"),
    ('b', "Encoder-only Cholec pre-training 0 · SurgeNetXL −0.056 · frozen DINOv3-7B feature branch −0.005"),
    ('h', "Post-processing"),
    ('b', "Islands < 0.4 % of frame: Dice +0.005 but HD +0.022 (Task 1) · class-wise probability scaling α (+0.014 Dice on an earlier ensemble): −0.002 Dice / +0.006 HD on the final one → dropped · anatomy rules / component-count limits as post-processing 0 / −0.007"),
    ('h', "Other"),
    ('b', "SWA / EMA −0.006 · multi-task with the Task 3 head −0.032 · merged-only specialist 0 · fine-only specialist added with weight 4 to the fine head −0.013"),
], size=10)
notes(S[5], "What did not work, each on at least two folds. Losses: size-weighted, cross-entropy, focal-Tversky, RMI and Hausdorff terms, a boundary loss that helps ConvNeXt but hurts Swin. Post-processing: anatomy rules change nothing, and two steps that helped older ensembles, frame-relative island removal and per-class probability scaling, hurt the final one on Hausdorff. Architectures: larger or newer encoders, SegFormer, a ViT decoder, random-initialised decoders, higher resolution. Pseudo-labels from a video generator with SAM3 propagation: good labels, but minus one and a half points when trained on. Encoder-only pre-training, a frozen DINOv3 branch, checkpoint averaging, multi-task training with Task 3.")

# slide 7: final members
S[6].shapes.title.text_frame.text = "Final ensemble: 7 recipes, 31 all-data checkpoints"
member_table(S[6], MEMBERS, y=0.9, size=9)
tf = body_box(S[6], y=2.9, h=1.2)
fill(tf, [('t', "All checkpoints: all 526 frames, final epoch, 1024×576, loss incl. fine→merged consistency 0.25. CV score = internal val (mean fine/merged Dice, 5-fold); ensemble CV (official, fold models): Task 1 Dice 0.7193 / HD 0.2053, Task 2 Dice 0.7453 / HD 0.1848 with TTA + class-scaled island removal. Inference: seeds averaged within a recipe → 7 recipes averaged → flip TTA on 5 → upsample → argmax → island removal. Container: frame-chunked, budget-controlled; 139 frames ≈ 50 min on one GPU.")], size=9)
notes(S[6], "The final ensemble: seven recipes, thirty-one all-data checkpoints. Five surgically pre-trained DeepLabV3-plus variants, EndoVis, CholecSeg8k with ConvNeXt-XL, a fine-specialised fine-tune, joint pre-training and a no-flip variant, plus two ImageNet-only Unet-plus-plus models for diversity. Seeds are averaged inside each recipe, the recipes averaged equally, flip TTA on five, then upsampling, argmax and class-scaled island removal. Thank you.")
out1 = HERE / "TigerSQAI_task12_jmees_v2.pptx"; prs.save(out1); print("wrote", out1, len(prs.slides), "slides")

# ====================================================================== Task 3 (same template)
prs = Presentation(TPL)
for i in range(len(prs.slides) - 1, 0, -1): delete_slide(prs, i)
t = prs.slides[0].shapes.title.text_frame
t.paragraphs[0].runs[0].text = "Tiger SQ-AI Challenge 2026"
# keep the remaining runs/paragraphs of the title placeholder as they are (authors / affiliation); fix the task line
for p in t.paragraphs:
    for r in p.runs:
        if "Tasks 1 & 2" in r.text: r.text = r.text.replace("Tasks 1 & 2", "Task 3")
notes(prs.slides[0], "Tiger SQ-AI 2026, Task 3, lymph-node station visibility, team Jmees. Five-fold out-of-fold numbers on the five hundred eighteen labelled frames, official code, and nothing in the pipeline reads a file name.")

s = new_slide(prs, "CV score digest")
s.shapes.add_picture(str(FIG / "v2_progression.png"), Inches(1.4), Inches(0.85), height=Inches(4.4))
notes(s, "The digest. We started with a classification head on the frozen features of our segmentation encoder: F1 zero point seven one. The turning point was to stop looking at the image and read our own predicted masks instead: an anatomy inventory into gradient boosting, zero point seven six. Feeding the same features into the network head and adding context and layout features took us to zero point seven nine, and replacing per-station LightGBM by a multi-output XGBoost blended with one MLP to zero point eight one F1, AUROC zero point nine three.")

s = new_slide(prs, "Basic recipe")
fill(body_box(s), [
    ('b', "**Input:** our own predicted fine + merged masks from the Task 1/2 ensemble (out-of-fold masks at training time, so features match test time) + GAP features of the 5 frozen fold models of one DeepLabV3+ segmentation recipe"),
    ('b', "**Features (1741):** anatomy inventory 315 (per class: area, presence, centroid, components, extent) · anatomy context 202 (contact matrix, surroundings of lymph nodes / fat, proximity) · 6×12 occupancy grid 1224"),
    ('b', "**Models:** 15 XGBoost multi-output trees (one tree with a 14-d leaf shared by all stations; depth 6, 1000 trees, lr 0.02; 1741 features; 3 seeds × 5 folds) + 15 MLPs (GAP ⊕ inventory 315 → 256 → 14; 3 seeds × 5 folds)"),
    ('b', "**Blend:** 0.7 XGBoost / 0.3 MLP → per-station piecewise-linear calibration so the OOF-optimal threshold lands on 0.5"),
    ('b', "**Validation:** same case-grouped 5 folds as segmentation; official evaluate_cls; **no station name or file name used**"),
], size=13)
notes(s, "The recipe. The input is not the image but our own predicted masks, out-of-fold at training time so the features look as they will at test time, plus pooled features of five frozen fold encoders. From the masks we compute seventeen hundred features: an inventory of what is present and how large, a context block of what touches what, and a coarse six-by-twelve occupancy grid for layout. Two branches: a multi-output XGBoost, one tree shared by all fourteen stations, and a small MLP on encoder features plus the inventory, blended seventy percent boosting and calibrated per station.")

s = new_slide(prs, "What worked")
fill(body_box(s), [
    ('h', "Finding"),
    ('b', "Classifying the **local neighbourhood** of a lymph-node component: 0.269 vs chance 0.244; an **inventory of the whole frame**: 0.344 → the station is read from the surrounding anatomy, and 179/518 frames have no lymph-node pixels at all"),
    ('h', "Features"),
    ('b', "**Anatomy inventory → LightGBM** (masks only) F1 **0.756** vs image-only head 0.741 (same honest protocol)"),
    ('b', "**+ anatomy context + occupancy grid** → **+0.010**; **mask features fed into the CNN head** → **+0.015**"),
    ('h', "Models"),
    ('b', "**Multi-output XGBoost** (one tree, 14-d leaf) instead of per-station LightGBM → alone 0.7917 / 0.9208, above the whole earlier LightGBM blend (0.7890 / 0.9165)"),
    ('b', "**Blend** 0.7 XGBoost + 0.3 MLP[GAP ⊕ 315] → **+0.014 F1** over XGBoost alone; each branch gets the feature set that suits it (1741 helps trees, hurts the MLP)"),
    ('b', "**Calibration for the fixed 0.5 threshold** (monotone map, AUROC unchanged) → **+0.028**"),
    ('h', "Robustness"),
    ('b', "Trained on predicted masks, evaluated on ground-truth masks: no degradation → better segmentation cannot hurt Task 3"),
    ('t', "Final: weighted F1 **0.806** · AUROC **0.925** (image-only start: 0.707 / 0.887); better than the earlier LightGBM blend on all 5 folds"),
], size=12)
notes(s, "What worked. The finding everything rests on: the neighbourhood of a lymph node identifies the station at chance level, and a third of the frames contain no lymph-node pixels at all; the signal is which structures are visible, and where. An inventory of our predicted masks into gradient boosting already beats the image-only head. Context and layout features add one point, the mask features inside the network head one and a half. A single multi-output XGBoost, one tree shared by all stations, beats the whole earlier LightGBM blend on its own; blending it with the MLP adds another point and a half, and calibrating for the fixed threshold almost three. Zero point eight one F1, zero point nine three AUROC.")

s = new_slide(prs, "What did not work")
fill(body_box(s), [
    ('h', "Input"),
    ('b', "Classifying from the lymph-node region itself → chance level"),
    ('b', "Adding the 1536-d encoder features to LightGBM → **−0.021** (518 frames cannot support 1 851 features in a tree model)"),
    ('h', "Models"),
    ('b', "Multi-task Task 3 head inside the segmentation network → worse on both tasks (AUROC −0.016, segmentation −0.032)"),
    ('b', "Logistic regression instead of LightGBM → −0.027 · stacking the CNN probability as a feature instead of blending → −0.014"),
    ('b', "Selecting heads by best validation epoch → optimistic OOF; final-epoch heads are used"),
    ('h', "Features / search"),
    ('b', "1741 features into the MLP (AUROC 0.883 vs 0.901 with 315) · dropping the occupancy grid from XGBoost −0.011 · 29 XGBoost hyper-parameter variants all within seed noise"),
    ('b', "Free greedy selection over 78 individual members (nested) → worse than the fixed blend: 518 frames do not support it · features from a different ensemble than the container's → not reproducible at test time (discarded)"),
], size=12)
notes(s, "What did not work. The lymph-node region on its own. Adding the encoder features to the gradient boosting model, minus two points, because five hundred frames cannot support that many features in a tree model. A multi-task head inside the segmentation network, worse on both tasks. Logistic regression instead of boosting, stacking instead of blending, and the big feature set inside the MLP. Tuning XGBoost changed nothing beyond seed noise, and letting a greedy search pick members freely over-fitted five hundred frames and lost to the fixed blend. Thank you.")
out2 = HERE / "TigerSQAI_task3_jmees.pptx"; prs.save(out2); print("wrote", out2, len(prs.slides), "slides")
