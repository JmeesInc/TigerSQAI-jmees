"""Build the two 3-minute presentation decks (16:9) with speaker notes.
Run from repo root:  .venv/bin/python3 submit/writeup/video/build_pptx.py
Outputs: submit/writeup/video/video1_task12_jmees.pptx, video2_task3_jmees.pptx
Each slide's narration is stored in the notes pane (View > Notes in PowerPoint)."""
from pathlib import Path
from pptx import Presentation
from pptx.util import Inches, Pt, Emu
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN, MSO_ANCHOR
from pptx.enum.shapes import MSO_SHAPE

HERE = Path(__file__).resolve().parent
FIG = HERE / "figures"; FIG2 = HERE.parent / "figures"
INK, MUTED = RGBColor(0x1f, 0x29, 0x37), RGBColor(0x6b, 0x72, 0x80)
BLUE, PURPLE, RED, GREEN = RGBColor(0x3b, 0x6f, 0xb6), RGBColor(0x6b, 0x3f, 0xa0), RGBColor(0xb0, 0x4a, 0x4a), RGBColor(0x2f, 0x7d, 0x3a)
W, H = Inches(13.333), Inches(7.5)


class Deck:
    def __init__(self, accent):
        self.prs = Presentation(); self.prs.slide_width, self.prs.slide_height = W, H
        self.blank = self.prs.slide_layouts[6]; self.accent = accent; self.n = 0

    def _text(self, slide, x, y, w, h, text, size=18, bold=False, color=INK, align=PP_ALIGN.LEFT, anchor=MSO_ANCHOR.TOP):
        tb = slide.shapes.add_textbox(x, y, w, h); tf = tb.text_frame; tf.word_wrap = True; tf.vertical_anchor = anchor
        for i, line in enumerate(text if isinstance(text, list) else [text]):
            p = tf.paragraphs[0] if i == 0 else tf.add_paragraph(); p.alignment = align
            r = p.add_run(); r.text = line; r.font.size = Pt(size); r.font.bold = bold; r.font.color.rgb = color; r.font.name = "Calibri"
        return tb

    def _bullets(self, slide, x, y, w, h, items, size=18):
        tb = slide.shapes.add_textbox(x, y, w, h); tf = tb.text_frame; tf.word_wrap = True
        first = True
        for it in items:
            level, txt = (it if isinstance(it, tuple) else (0, it))
            p = tf.paragraphs[0] if first else tf.add_paragraph(); first = False
            p.level = level; p.space_after = Pt(6)
            # manual bullet glyph (python-pptx has no simple bullet API on textboxes)
            parts = txt.split("**")
            glyph = "•  " if level == 0 else "–  "
            r0 = p.add_run(); r0.text = ("    " * level) + glyph; r0.font.size = Pt(size - 2 * level); r0.font.color.rgb = self.accent
            for j, part in enumerate(parts):
                if not part: continue
                r = p.add_run(); r.text = part; r.font.size = Pt(size - 2 * level); r.font.name = "Calibri"
                r.font.bold = (j % 2 == 1); r.font.color.rgb = INK
        return tb

    def _chrome(self, slide, title, kicker=None):
        self.n += 1
        bar = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, 0, 0, W, Inches(0.12)); bar.fill.solid(); bar.fill.fore_color.rgb = self.accent; bar.line.fill.background()
        self._text(slide, Inches(0.5), Inches(0.3), Inches(12.3), Inches(0.8), title, size=28, bold=True)
        if kicker: self._text(slide, Inches(0.5), Inches(0.95), Inches(12.3), Inches(0.5), kicker, size=14, color=MUTED)
        self._text(slide, Inches(11.6), Inches(7.05), Inches(1.5), Inches(0.3), f"Team Jmees · {self.n}", size=10, color=MUTED, align=PP_ALIGN.RIGHT)

    def _notes(self, slide, notes):
        slide.notes_slide.notes_text_frame.text = notes

    def title(self, title, sub, lines, notes):
        s = self.prs.slides.add_slide(self.blank); self.n += 1
        bar = s.shapes.add_shape(MSO_SHAPE.RECTANGLE, 0, 0, Inches(0.35), H); bar.fill.solid(); bar.fill.fore_color.rgb = self.accent; bar.line.fill.background()
        self._text(s, Inches(0.9), Inches(1.4), Inches(11.5), Inches(1.6), title, size=34, bold=True)
        self._text(s, Inches(0.9), Inches(3.0), Inches(11.5), Inches(0.7), sub, size=20, color=self.accent, bold=True)
        self._text(s, Inches(0.9), Inches(3.9), Inches(11.5), Inches(2.5), lines, size=16, color=MUTED)
        self._notes(s, notes); return s

    def bullets_fig(self, title, kicker, items, fig=None, notes="", fig_w=6.3, size=17):
        s = self.prs.slides.add_slide(self.blank); self._chrome(s, title, kicker)
        tw = Inches(12.3) if fig is None else Inches(12.3 - fig_w - 0.3)
        self._bullets(s, Inches(0.5), Inches(1.55), tw, Inches(5.3), items, size=size)
        if fig is not None:
            s.shapes.add_picture(str(fig), W - Inches(fig_w + 0.5), Inches(1.6), width=Inches(fig_w))
        self._notes(s, notes); return s

    def fig_only(self, title, kicker, fig, notes="", caption=None, fig_w=11.5):
        s = self.prs.slides.add_slide(self.blank); self._chrome(s, title, kicker)
        pic = s.shapes.add_picture(str(fig), Inches(0.5), Inches(1.55), width=Inches(fig_w))
        if pic.top + pic.height > Inches(6.9):   # keep inside the slide
            scale = (Inches(6.9) - Inches(1.55)) / pic.height; pic.height = int(pic.height * scale); pic.width = int(pic.width * scale)
        pic.left = int((W - pic.width) / 2)
        if caption: self._text(s, Inches(0.5), Inches(6.85), Inches(12.3), Inches(0.4), caption, size=13, color=MUTED)
        self._notes(s, notes); return s

    def save(self, path): self.prs.save(path); print("wrote", path, self.n, "slides")


# =============================================================== VIDEO 1 — Tasks 1 & 2
d = Deck(BLUE)
d.title("Tasks 1 & 2: what worked and what did not",
        "Tiger SQ-AI Challenge 2026 · segmentation of thoracoscopic esophagectomy frames",
        ["Team Jmees — Shunsuke Kikuchi, Atsushi Kouno, Hiroki Matsuzaki (Jmees Inc.)",
         "All numbers: 5-fold cross-validation, case-grouped folds, official evaluation code, original resolution",
         "Final: 16 dual-head ConvNeXt-L models · Task 1 Dice 0.717 / Task 2 Dice 0.732 (fold ensemble, OOF)"],
        "Team Jmees, Tasks 1 and 2: where we started, what moved the score, and what did not. All numbers are five-fold cross-validation with the official code.")

d.fig_only("The road: from a plain Unet++ to the final ensemble", "Official weighted Dice, 5-fold out-of-fold, same folds throughout",
           FIG / "v1_progression.png",
           "We started from a two-head Unet-plus-plus at fine Dice zero point six zero. Stronger augmentation, plus two points. MaxViT-Base at ten-twenty-four by five-seven-six, plus one. Then the first big step: a consistency loss that makes the fine probabilities, summed per merged class, agree with the coarse head, plus eight points on Task 1. Ensembling, a better backbone, metric-aligned losses and post-processing took us to zero point seven three.",
           caption="Each point is a 5-fold OOF evaluation of the full 526-frame training set.")

d.bullets_fig("Backbone and training recipe", "One encoder, two decoders; everything measured on ≥ 2 folds (single-fold noise σ ≈ 0.007)",
              ["**Worked:** ConvNeXt-Large (IN-22k, 384) replaces MaxViT-B: **+0.025** on the same recipe",
               "**Worked:** DeepLabV3+ as a second decoder family (≈ Unet++, different errors → ensemble value)",
               "**Worked:** 20 epochs, AdamW 2e-4, cosine; 40 epochs no better",
               "**Did not:** ConvNeXt-XL (≈ L, and −0.018 with DiceDet), ConvNeXt-V2, CAFormer, SE-ResNeXt, SegFormer-B5, DPT-L (−0.01 to −0.09); 1536×864 input (−0.011 / −0.028)",
               "**Did not:** random-init UPerNet / FPN / MAnet decoders (−0.07 to −0.16); ADE20k-pre-trained UPerNet-Swin-L wins fold 0 but loses on 3 of 5 folds",
               "**Did not:** frozen DINOv3-7B feature branch (+0 once the consistency loss is present)"],
              FIG / "v1_dual_head.png",
              "Backbone: ConvNeXt-Large replaced MaxViT for two and a half points, and DeepLabV3-plus matched Unet-plus-plus while making different errors. Twenty epochs were enough. Larger or newer encoders, SegFormer, a ViT decoder, random-initialised UPerNet and FPN, higher input resolution, and a frozen DINOv3 branch all failed to help.",
              fig_w=5.6, size=15)

d.bullets_fig("Losses that match the metric", "The score gives 0 to a class present in only one of prediction and reference — so stray classes are the enemy",
              ["**DiceDet:** hinge on the *maximum* probability of every class absent from the reference → **+0.011**",
               "**AnatomyLoss:** differentiable penalty on forbidden contacts and mutually exclusive structures → **+0.009**",
               "Together **+0.025** (super-additive): one fixes *which* classes appear, the other *where*",
               "The graph written from 3-D mediastinal anatomy (23 classes, no data statistics) scores ≥ the one estimated from the masks: 0.693 vs 0.689",
               "**Did not:** the same rules as post-processing (±0.000), Dice+CE (−0.030), size-weighted Dice (−0.031), focal-Tversky, Hausdorff-DT; boundary loss helps ConvNeXt (+0.004) but costs −0.054 on Swin",
               "**Did not:** dropping horizontal flip helps alone (+0.010, left/right classes) but is not additive with DiceDet"],
              None,
              "Losses. DiceDet, a hinge on the maximum probability of every class absent from the reference, stops the network emitting stray classes: plus one point. AnatomyLoss penalises impossible contacts and exclusive structures from a graph written from anatomy, not from data: plus one point. Together plus two and a half, more than the sum. The same rules as post-processing changed nothing. Cross-entropy, size-weighted, focal-Tversky and Hausdorff losses were worse, and a loss that helps one architecture can hurt another.",
              size=15)

d.bullets_fig("Data: pre-training and pseudo-labels", "Public data only",
              ["**Worked:** pre-train **encoder + decoder** on CholecSeg8k / EndoVis 2018 at 1024×576, transfer everything but the heads → **+0.015 on every fold** (0.693 → 0.708–0.710); corpus choice barely matters",
               "**Did not (earlier):** encoder-only surgical pre-training (Cholec, SurgeNetXL): 0 to −0.056 — the decoder is what carries the domain",
               "**Pseudo-labels:** Wan2.1 first-last-frame video between two station views + SAM3 label propagation; propagated labels are good (Dice 0.76–0.79 vs 0.57 copy control; generated frames indistinguishable to the segmenter) — but training on them: +0.006 at best, **−0.016** on the final recipe → not used",
               "**ToolPaste** (instrument cut-outs from SAR-RARP50 / SurgToolLoc): mixed alone, kept as one diversity member",
               "**Did not:** stronger augmentation on the final recipe (±0)"],
              None,
              "Data. The largest late gain: pre-training encoder and decoder together on CholecSeg8k and EndoVis 2018, plus one and a half points on every fold. Encoder-only pre-training had given nothing, so the decoder carries the domain. We also generated pseudo-labels: a video model interpolates between two station views, SAM3 propagates the real masks from both ends. The labels were good, but training on them cost one and a half points, so they are not in the submission.",
              size=15)

d.bullets_fig("Post-processing and specialisation", "Cross-fitted on OOF predictions, applied in the container",
              ["**Island removal:** connected components < 0.4 % of the frame → surrounding majority class: **+0.032 fine / +0.027 merged**, better on every centre leave-one-centre-out",
               "**Class-wise probability scaling α** before the argmax (19 fine classes, 0.3–3.8): **+0.014** on Task 2; ≈ 0 on Task 1 → applied to fine only",
               "**Fine specialist:** fine-tune 8 epochs with the coarse loss off: **+0.0125 on 5/5 folds**; added to the Task 2 average only (weight 4)",
               "**Did not:** anatomy rules / component-count limits as post-processing (0 / −0.007), SWA / EMA (−0.006), a coarse specialist (0), the fine specialist without AnatomyLoss (diverges, −0.044)"],
              FIG / "v1_score_movers.png",
              "Post-processing. Island removal, components below zero point four percent of the frame relabelled with the surrounding majority class: plus three points fine, plus two point seven merged, better on every centre. A per-class probability scaling before the argmax: plus one point four on fine only. A fine specialist, fine-tuned with the coarse loss off: plus one point on every fold, fine average only. Checkpoint averaging and rule-based post-processing did not help.",
              fig_w=6.0, size=15)

d.bullets_fig("The ensemble: diversity first, then stop", "Selected by 5-fold CV with the official Dice + Hausdorff, averaged over folds",
              ["2 → 4 → 7 recipes: 0.675 → 0.687 → 0.695; **the two strongest recipes alone lose by 0.010 to seven ordinary ones**",
               "But Task 2 drops as soon as recipes outside the surgically pre-trained group are added → **16 full-data models**:",
               (1, "12 × DeepLabV3+ / ConvNeXt-L, CholecSeg8k / EndoVis18 / joint pre-training, DiceDet + AnatomyLoss — seeds, ToolPaste, no-flip, 960×544 and 896×512 variants"),
               (1, "1 × Unet++ / ConvNeXt-L, EndoVis18 pre-trained"),
               (1, "2 × ImageNet-only Unet++ (ConvNeXt-XL; ConvNeXt-L with DiceDet + AnatomyLoss)"),
               (1, "1 × fine specialist (Task 2 only)"),
               "Softmax average → α (fine) → bilinear to original size → argmax → island removal"],
              FIG / "v1_ensemble_size.png",
              "Ensemble. Diversity beats strength: the two strongest recipes alone lose by a full point to seven ordinary ones. But Task 2 drops as soon as recipes outside the pre-trained group are added. So: sixteen models, twelve pre-trained DeepLabV3-plus variants with different seeds, augmentations and input sizes, one pre-trained Unet-plus-plus, two ImageNet-only Unet-plus-plus for diversity, and the fine specialist.",
              fig_w=5.8, size=14)

d.fig_only("What did not work", "Each tested on at least two folds against the baseline of the same experiment",
           FIG / "v1_did_not_work.png",
           "In one picture, what did not work, each on at least two folds: pseudo-labels, DINOv3, surgical and transformer encoders, ViT and random-initialised decoders, cross-entropy and size-weighted losses, checkpoint averaging, multi-task training with Task 3, and losses that do not transfer.",
           fig_w=10.5)

d.bullets_fig("Take-aways", "",
              ["Match the loss and the post-processing to the metric's absent-class rule (DiceDet, island removal, α)",
               "Teach anatomy during training; do not patch outputs afterwards",
               "Pre-train the **decoder** as well as the encoder on public surgical data",
               "Ensemble for diversity, select by CV, and stop when the weaker recipes start to hurt",
               "Single-fold noise is ±0.007: decide on ≥ 2 folds, and never on one",
               "Code: github.com/JmeesInc/TigerSQAI-jmees"],
              None,
              "Take-aways: match loss and post-processing to the metric, teach anatomy during training, pre-train the decoder too, ensemble for diversity but stop early, and never decide on one fold. Thank you.",
              size=19)
d.save(HERE / "video1_task12_jmees.pptx")

# =============================================================== VIDEO 2 — Task 3
d = Deck(PURPLE)
d.title("Task 3: what worked and what did not",
        "Tiger SQ-AI Challenge 2026 · lymph-node station visibility (14 stations, multi-label)",
        ["Team Jmees — Shunsuke Kikuchi, Atsushi Kouno, Hiroki Matsuzaki (Jmees Inc.)",
         "All numbers: 5-fold OOF on the 518 labelled frames, case-grouped folds, official evaluate_cls",
         "No station name or file name is used anywhere · Final: weighted F1 0.789 · AUROC 0.917"],
        "Team Jmees on Task 3, station visibility. Three minutes: where we started, what worked, what did not. All numbers are five-fold out-of-fold with the official code, and nothing in the pipeline reads a file name.")

d.fig_only("The road: from an image head to a mask-reading classifier", "Weighted F1 at the fixed 0.5 threshold, 5-fold OOF",
           FIG / "v2_progression.png",
           "We started where most teams would: a classification head on the frozen features of our segmentation encoder. F1 zero point seven one, AUROC zero point eight nine. Training that head jointly inside the segmentation network was worse on both tasks. The turning point was to stop looking at the image and look at our own masks instead. That took us to zero point seven nine.")

d.bullets_fig("The finding: the signal is the whole scene, not the node", "Analysis on the annotated frames, case-grouped CV",
              ["Classifying the **local neighbourhood** of a lymph-node component: 0.269 vs chance 0.244",
               "An **inventory of the whole frame** (which structures are visible, where): 0.344",
               "179 of 518 frames have *no* lymph-node pixels at all, yet stations are visible",
               "→ predict the station from what the segmentation sees: our own predicted fine + coarse masks",
               "Masks only → LightGBM: **F1 0.756** — already above image-only (0.741 with the same honest protocol)"],
              FIG / "v2_where_signal.png",
              "The finding everything rests on. Classifying the neighbourhood of a lymph-node component identifies the station at chance level, and a third of the labelled frames contain no lymph node pixels at all. What carries the signal is the rest of the frame: which structures are visible and where. That is how a surgeon reads a station, from the aorta, the azygos vein, the bronchus and the nerve in view. So we classify from an inventory of our own predicted masks. A gradient-boosting model on that inventory alone already beats the image-only head.",
              fig_w=6.0, size=16)

d.bullets_fig("Features and models that worked", "Training features come from out-of-fold masks, so they look exactly like test-time features",
              ["**Anatomy inventory** (315): per class area, presence, centroid, components, extent",
               "**+ anatomy context** (202): contact matrix, what surrounds lymph nodes and fat, proximity · **+ 6×12 occupancy grid** (1224): restores layout → **+0.010**",
               "**Mask features fed into the CNN head** (encoder GAP ⊕ 1741 features): **+0.015** over the image-only head",
               "**Two branches blended** 0.4 MLP / 0.6 LightGBM: 45 MLPs (3 encoders × 3 seeds × 5 folds) + 420 LightGBMs (per station): **+0.013** over the best single model",
               "**Calibration for the fixed threshold:** piecewise-linear map so each station's OOF-optimal threshold lands on 0.5: **+0.028**, monotone → AUROC unchanged"],
              FIG / "v2_features_branches.png",
              "What worked on top. Adding context features, which classes touch which, what surrounds the nodes and the fat, and a coarse six-by-twelve occupancy grid that restores layout: one point. Feeding the same mask features into the CNN head: one and a half points. Blending an MLP branch with a LightGBM branch, forty-five networks and four hundred twenty boosters: one point three over the best single model. And the biggest single lever: F1 is binarised at zero point five, but the best threshold per station is not, so each station's probability is mapped piecewise-linearly so its out-of-fold optimum lands on zero point five. Almost three points, with AUROC untouched.",
              fig_w=6.0, size=14)

d.bullets_fig("What did not work", "Each against the alternative we kept",
              ["Classifying the lymph-node region itself: chance level",
               "Multi-task Task 3 head inside the segmentation network: worse on both tasks (AUROC −0.016, segmentation −0.032)",
               "Adding the 1536-d encoder features to LightGBM: **−0.021** — 518 frames cannot support 1 851 features in a tree model",
               "Logistic regression instead of boosting: −0.027",
               "Stacking the CNN probability as a feature instead of blending: −0.014",
               "Selecting the head by best validation epoch: optimistic OOF; final-epoch heads are used",
               "Checked, not tried: a model trained on predicted masks does **not** degrade on ground-truth masks → better segmentation cannot hurt Task 3"],
              FIG / "v2_did_not_work.png",
              "What did not work. The lymph-node region itself. Multi-task training inside the segmentation network. Adding the encoder features to the gradient boosting model, minus two points, because five hundred frames cannot support that many features in a tree model. Logistic regression instead of boosting, minus three. Stacking the CNN probability as a feature instead of blending, minus one and a half. And one thing we checked rather than tried: a model trained on predicted masks does not degrade when it is given ground-truth masks, so a better segmentation cannot hurt Task 3.",
              fig_w=6.0, size=14)

d.bullets_fig("Take-aways", "",
              ["Station visibility is about **what anatomy is on screen**, not texture → re-use the segmentation",
               "Calibrate for the fixed threshold; blend, do not stack",
               "Keep the tabular model small; keep training features out-of-fold",
               "Final: weighted F1 **0.789** · AUROC **0.917** (image-only start: 0.707 / 0.887)",
               "Code: github.com/JmeesInc/TigerSQAI-jmees"],
              FIG / "v2_results.png",
              "Take-aways. Station visibility is a question of what anatomy is on screen, so re-use the segmentation. Calibrate for the fixed threshold, blend rather than stack, keep the tabular model small, and keep training features out of fold. That took us from zero point seven one to zero point seven nine F1. Thank you.",
              fig_w=6.2, size=17)
d.save(HERE / "video2_task3_jmees.pptx")
