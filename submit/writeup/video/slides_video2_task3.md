# Video 2 — Task 3 (station visibility): what worked, what did not

Team Jmees · deck: `video2_task3_jmees.pptx` (narration is also in each slide's notes pane) · narration ≈ 445 words → 178 s at 150 wpm.
Generated from the deck by `build_pptx.py`; edit the deck script, not this file.

## Slide 1 · Task 3: what worked and what did not  (35 words)

- Tiger SQ-AI Challenge 2026 · lymph-node station visibility (14 stations, multi-label)
- Team Jmees — Shunsuke Kikuchi, Atsushi Kouno, Hiroki Matsuzaki (Jmees Inc.)
- All numbers: 5-fold OOF on the 518 labelled frames, case-grouped folds, official evaluate_cls
- No station name or file name is used anywhere · Final: weighted F1 0.789 · AUROC 0.917

**SAY**
> Team Jmees on Task 3, station visibility. Three minutes: where we started, what worked, what did not. All numbers are five-fold out-of-fold with the official code, and nothing in the pipeline reads a file name.

## Slide 2 · The road: from an image head to a mask-reading classifier  (65 words)

- Weighted F1 at the fixed 0.5 threshold, 5-fold OOF

**SAY**
> We started where most teams would: a classification head on the frozen features of our segmentation encoder. F1 zero point seven one, AUROC zero point eight nine. Training that head jointly inside the segmentation network was worse on both tasks. The turning point was to stop looking at the image and look at our own masks instead. That took us to zero point seven nine.

## Slide 3 · The finding: the signal is the whole scene, not the node  (92 words)

- Analysis on the annotated frames, case-grouped CV
- •  Classifying the local neighbourhood of a lymph-node component: 0.269 vs chance 0.244
- •  An inventory of the whole frame (which structures are visible, where): 0.344
- •  179 of 518 frames have *no* lymph-node pixels at all, yet stations are visible
- •  → predict the station from what the segmentation sees: our own predicted fine + coarse masks
- •  Masks only → LightGBM: F1 0.756 — already above image-only (0.741 with the same honest protocol)

**SAY**
> The finding everything rests on. Classifying the neighbourhood of a lymph-node component identifies the station at chance level, and a third of the labelled frames contain no lymph node pixels at all. What carries the signal is the rest of the frame: which structures are visible and where. That is how a surgeon reads a station, from the aorta, the azygos vein, the bronchus and the nerve in view. So we classify from an inventory of our own predicted masks. A gradient-boosting model on that inventory alone already beats the image-only head.

## Slide 4 · Features and models that worked  (108 words)

- Training features come from out-of-fold masks, so they look exactly like test-time features
- •  Anatomy inventory (315): per class area, presence, centroid, components, extent
- •  + anatomy context (202): contact matrix, what surrounds lymph nodes and fat, proximity · + 6×12 occupancy grid (1224): restores layout → +0.010
- •  Mask features fed into the CNN head (encoder GAP ⊕ 1741 features): +0.015 over the image-only head
- •  Two branches blended 0.4 MLP / 0.6 LightGBM: 45 MLPs (3 encoders × 3 seeds × 5 folds) + 420 LightGBMs (per station): +0.013 over the best single model
- •  Calibration for the fixed threshold: piecewise-linear map so each station's OOF-optimal threshold lands on 0.5: +0.028, monotone → AUROC unchanged

**SAY**
> What worked on top. Adding context features, which classes touch which, what surrounds the nodes and the fat, and a coarse six-by-twelve occupancy grid that restores layout: one point. Feeding the same mask features into the CNN head: one and a half points. Blending an MLP branch with a LightGBM branch, forty-five networks and four hundred twenty boosters: one point three over the best single model. And the biggest single lever: F1 is binarised at zero point five, but the best threshold per station is not, so each station's probability is mapped piecewise-linearly so its out-of-fold optimum lands on zero point five. Almost three points, with AUROC untouched.

## Slide 5 · What did not work  (92 words)

- Each against the alternative we kept
- •  Classifying the lymph-node region itself: chance level
- •  Multi-task Task 3 head inside the segmentation network: worse on both tasks (AUROC −0.016, segmentation −0.032)
- •  Adding the 1536-d encoder features to LightGBM: −0.021 — 518 frames cannot support 1 851 features in a tree model
- •  Logistic regression instead of boosting: −0.027
- •  Stacking the CNN probability as a feature instead of blending: −0.014
- •  Selecting the head by best validation epoch: optimistic OOF; final-epoch heads are used
- •  Checked, not tried: a model trained on predicted masks does not degrade on ground-truth masks → better segmentation cannot hurt Task 3

**SAY**
> What did not work. The lymph-node region itself. Multi-task training inside the segmentation network. Adding the encoder features to the gradient boosting model, minus two points, because five hundred frames cannot support that many features in a tree model. Logistic regression instead of boosting, minus three. Stacking the CNN probability as a feature instead of blending, minus one and a half. And one thing we checked rather than tried: a model trained on predicted masks does not degrade when it is given ground-truth masks, so a better segmentation cannot hurt Task 3.

## Slide 6 · Take-aways  (53 words)

- •  Station visibility is about what anatomy is on screen, not texture → re-use the segmentation
- •  Calibrate for the fixed threshold; blend, do not stack
- •  Keep the tabular model small; keep training features out-of-fold
- •  Final: weighted F1 0.789 · AUROC 0.917 (image-only start: 0.707 / 0.887)
- •  Code: github.com/JmeesInc/TigerSQAI-jmees

**SAY**
> Take-aways. Station visibility is a question of what anatomy is on screen, so re-use the segmentation. Calibrate for the fixed threshold, blend rather than stack, keep the tabular model small, and keep training features out of fold. That took us from zero point seven one to zero point seven nine F1. Thank you.
