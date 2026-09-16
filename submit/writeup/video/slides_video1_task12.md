# Video 1 — Tasks 1 & 2 (segmentation): what worked, what did not

Team Jmees · deck: `video1_task12_jmees.pptx` (narration is also in each slide's notes pane) · narration ≈ 493 words → 197 s at 150 wpm.
Generated from the deck by `build_pptx.py`; edit the deck script, not this file.

## Slide 1 · Tasks 1 & 2: what worked and what did not  (26 words)

- Tiger SQ-AI Challenge 2026 · segmentation of thoracoscopic esophagectomy frames
- Team Jmees — Shunsuke Kikuchi, Atsushi Kouno, Hiroki Matsuzaki (Jmees Inc.)
- All numbers: 5-fold cross-validation, case-grouped folds, official evaluation code, original resolution
- Final: 16 dual-head ConvNeXt-L models · Task 1 Dice 0.717 / Task 2 Dice 0.732 (fold ensemble, OOF)

**SAY**
> Team Jmees, Tasks 1 and 2: where we started, what moved the score, and what did not. All numbers are five-fold cross-validation with the official code.

## Slide 2 · The road: from a plain Unet++ to the final ensemble  (68 words)

- Official weighted Dice, 5-fold out-of-fold, same folds throughout
- Each point is a 5-fold OOF evaluation of the full 526-frame training set.

**SAY**
> We started from a two-head Unet-plus-plus at fine Dice zero point six zero. Stronger augmentation, plus two points. MaxViT-Base at ten-twenty-four by five-seven-six, plus one. Then the first big step: a consistency loss that makes the fine probabilities, summed per merged class, agree with the coarse head, plus eight points on Task 1. Ensembling, a better backbone, metric-aligned losses and post-processing took us to zero point seven three.

## Slide 3 · Backbone and training recipe  (46 words)

- One encoder, two decoders; everything measured on ≥ 2 folds (single-fold noise σ ≈ 0.007)
- •  Worked: ConvNeXt-Large (IN-22k, 384) replaces MaxViT-B: +0.025 on the same recipe
- •  Worked: DeepLabV3+ as a second decoder family (≈ Unet++, different errors → ensemble value)
- •  Worked: 20 epochs, AdamW 2e-4, cosine; 40 epochs no better
- •  Did not: ConvNeXt-XL (≈ L, and −0.018 with DiceDet), ConvNeXt-V2, CAFormer, SE-ResNeXt, SegFormer-B5, DPT-L (−0.01 to −0.09); 1536×864 input (−0.011 / −0.028)
- •  Did not: random-init UPerNet / FPN / MAnet decoders (−0.07 to −0.16); ADE20k-pre-trained UPerNet-Swin-L wins fold 0 but loses on 3 of 5 folds
- •  Did not: frozen DINOv3-7B feature branch (+0 once the consistency loss is present)

**SAY**
> Backbone: ConvNeXt-Large replaced MaxViT for two and a half points, and DeepLabV3-plus matched Unet-plus-plus while making different errors. Twenty epochs were enough. Larger or newer encoders, SegFormer, a ViT decoder, random-initialised UPerNet and FPN, higher input resolution, and a frozen DINOv3 branch all failed to help.

## Slide 4 · Losses that match the metric  (78 words)

- The score gives 0 to a class present in only one of prediction and reference — so stray classes are the enemy
- •  DiceDet: hinge on the *maximum* probability of every class absent from the reference → +0.011
- •  AnatomyLoss: differentiable penalty on forbidden contacts and mutually exclusive structures → +0.009
- •  Together +0.025 (super-additive): one fixes *which* classes appear, the other *where*
- •  The graph written from 3-D mediastinal anatomy (23 classes, no data statistics) scores ≥ the one estimated from the masks: 0.693 vs 0.689
- •  Did not: the same rules as post-processing (±0.000), Dice+CE (−0.030), size-weighted Dice (−0.031), focal-Tversky, Hausdorff-DT; boundary loss helps ConvNeXt (+0.004) but costs −0.054 on Swin
- •  Did not: dropping horizontal flip helps alone (+0.010, left/right classes) but is not additive with DiceDet

**SAY**
> Losses. DiceDet, a hinge on the maximum probability of every class absent from the reference, stops the network emitting stray classes: plus one point. AnatomyLoss penalises impossible contacts and exclusive structures from a graph written from anatomy, not from data: plus one point. Together plus two and a half, more than the sum. The same rules as post-processing changed nothing. Cross-entropy, size-weighted, focal-Tversky and Hausdorff losses were worse, and a loss that helps one architecture can hurt another.

## Slide 5 · Data: pre-training and pseudo-labels  (76 words)

- Public data only
- •  Worked: pre-train encoder + decoder on CholecSeg8k / EndoVis 2018 at 1024×576, transfer everything but the heads → +0.015 on every fold (0.693 → 0.708–0.710); corpus choice barely matters
- •  Did not (earlier): encoder-only surgical pre-training (Cholec, SurgeNetXL): 0 to −0.056 — the decoder is what carries the domain
- •  Pseudo-labels: Wan2.1 first-last-frame video between two station views + SAM3 label propagation; propagated labels are good (Dice 0.76–0.79 vs 0.57 copy control; generated frames indistinguishable to the segmenter) — but training on them: +0.006 at best, −0.016 on the final recipe → not used
- •  ToolPaste (instrument cut-outs from SAR-RARP50 / SurgToolLoc): mixed alone, kept as one diversity member
- •  Did not: stronger augmentation on the final recipe (±0)

**SAY**
> Data. The largest late gain: pre-training encoder and decoder together on CholecSeg8k and EndoVis 2018, plus one and a half points on every fold. Encoder-only pre-training had given nothing, so the decoder carries the domain. We also generated pseudo-labels: a video model interpolates between two station views, SAM3 propagates the real masks from both ends. The labels were good, but training on them cost one and a half points, so they are not in the submission.

## Slide 6 · Post-processing and specialisation  (71 words)

- Cross-fitted on OOF predictions, applied in the container
- •  Island removal: connected components < 0.4 % of the frame → surrounding majority class: +0.032 fine / +0.027 merged, better on every centre leave-one-centre-out
- •  Class-wise probability scaling α before the argmax (19 fine classes, 0.3–3.8): +0.014 on Task 2; ≈ 0 on Task 1 → applied to fine only
- •  Fine specialist: fine-tune 8 epochs with the coarse loss off: +0.0125 on 5/5 folds; added to the Task 2 average only (weight 4)
- •  Did not: anatomy rules / component-count limits as post-processing (0 / −0.007), SWA / EMA (−0.006), a coarse specialist (0), the fine specialist without AnatomyLoss (diverges, −0.044)

**SAY**
> Post-processing. Island removal, components below zero point four percent of the frame relabelled with the surrounding majority class: plus three points fine, plus two point seven merged, better on every centre. A per-class probability scaling before the argmax: plus one point four on fine only. A fine specialist, fine-tuned with the coarse loss off: plus one point on every fold, fine average only. Checkpoint averaging and rule-based post-processing did not help.

## Slide 7 · The ensemble: diversity first, then stop  (58 words)

- Selected by 5-fold CV with the official Dice + Hausdorff, averaged over folds
- •  2 → 4 → 7 recipes: 0.675 → 0.687 → 0.695; the two strongest recipes alone lose by 0.010 to seven ordinary ones
- •  But Task 2 drops as soon as recipes outside the surgically pre-trained group are added → 16 full-data models:
- –  12 × DeepLabV3+ / ConvNeXt-L, CholecSeg8k / EndoVis18 / joint pre-training, DiceDet + AnatomyLoss — seeds, ToolPaste, no-flip, 960×544 and 896×512 variants
- –  1 × Unet++ / ConvNeXt-L, EndoVis18 pre-trained
- –  2 × ImageNet-only Unet++ (ConvNeXt-XL; ConvNeXt-L with DiceDet + AnatomyLoss)
- –  1 × fine specialist (Task 2 only)
- •  Softmax average → α (fine) → bilinear to original size → argmax → island removal

**SAY**
> Ensemble. Diversity beats strength: the two strongest recipes alone lose by a full point to seven ordinary ones. But Task 2 drops as soon as recipes outside the pre-trained group are added. So: sixteen models, twelve pre-trained DeepLabV3-plus variants with different seeds, augmentations and input sizes, one pre-trained Unet-plus-plus, two ImageNet-only Unet-plus-plus for diversity, and the fine specialist.

## Slide 8 · What did not work  (40 words)

- Each tested on at least two folds against the baseline of the same experiment

**SAY**
> In one picture, what did not work, each on at least two folds: pseudo-labels, DINOv3, surgical and transformer encoders, ViT and random-initialised decoders, cross-entropy and size-weighted losses, checkpoint averaging, multi-task training with Task 3, and losses that do not transfer.

## Slide 9 · Take-aways  (30 words)

- •  Match the loss and the post-processing to the metric's absent-class rule (DiceDet, island removal, α)
- •  Teach anatomy during training; do not patch outputs afterwards
- •  Pre-train the decoder as well as the encoder on public surgical data
- •  Ensemble for diversity, select by CV, and stop when the weaker recipes start to hurt
- •  Single-fold noise is ±0.007: decide on ≥ 2 folds, and never on one
- •  Code: github.com/JmeesInc/TigerSQAI-jmees

**SAY**
> Take-aways: match loss and post-processing to the metric, teach anatomy during training, pre-train the decoder too, ensemble for diversity but stop early, and never decide on one fold. Thank you.
