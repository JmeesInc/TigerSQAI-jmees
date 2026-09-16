# Anatomy-aware ensemble segmentation and mask-inventory station classification for the Tiger SQ-AI Challenge (Tasks 1, 2 and 3)

> **v7 (2026-09-16 20:20) — describes the submitted image `v13` as built** (31 checkpoints in 7 recipe slots, flip-TTA on 5, class-wise α on the fine head and island removal at 0.4 % of the frame; Task 3 = XGBoost + MLP blend). Solution-only write-up; task, data and metric descriptions are omitted.

**Team name:** Jmees
**Authors:** Shunsuke Kikuchi, Atsushi Kouno, Hiroki Matsuzaki
**Affiliations:** Jmees Inc., Kashiwa, Chiba, Japan
**Contact:** shunsuke.kikuchi@jmees-inc.com
**Source code:** https://github.com/JmeesInc/TigerSQAI-jmees.git
**Docker image:** `docker.synapse.org/syn77311180/tigersqai26_shunsuke:v13` (one image serving Tasks 1, 2 and 3)

**Will you be able to make your submission public as part of the challenge archive?** Yes

All numbers in this document are 5-fold cross-validation (folds grouped by case, stratified by centre), scored out-of-fold with the official evaluation code at original resolution. Experiments are described by their content; the seven deployed recipes are listed in Appendix A and referred to as A1–A7.

---

## 1. Solution overview

**Tasks 1 & 2.** One network predicts both label sets: a ConvNeXt-Large encoder with two decoders (merged 15 + bg, fine 30 + bg). Seven training recipes (Appendix A) are averaged in softmax space from 31 all-data checkpoints (seed replicas averaged inside their recipe slot), with horizontal-flip TTA on 5 recipes; the averaged fine probabilities are scaled per class (α), upsampled and arg-maxed, and connected components smaller than 0.4 % of the frame are removed. The ingredients that moved the score most were: **fine→merged consistency loss (+0.08 on Task 1, early)**, **DiceDet + AnatomyLoss (+0.025)**, **encoder+decoder pre-training on public laparoscopic data (+0.015)**, and **recipe-diverse ensembling (+0.02)**. The two post-processing steps in the container (island removal, class-wise probability scaling) were developed on earlier ensembles and validated on Dice only; a re-check on both halves of the metric, completed after the image was built, shows that on the final ensemble they trade Dice for a larger Hausdorff loss, and that a class-dependent, much smaller island threshold without α would have been better (Section 2.5).

**Task 3.** The station is predicted from *our own predicted masks*, not from the image: 1741 features describing which structures are on screen, how much, where and next to what, go into a multi-output XGBoost; a small MLP on frozen encoder features plus the mask inventory is blended in (0.7 / 0.3) and each station is calibrated for the fixed 0.5 threshold. OOF weighted F1 0.806 / AUROC 0.925. No station name or file name is used anywhere.

**Design principles.** (1) Optimise what the metric measures: a class present in only one of prediction and reference scores Dice 0 and Hausdorff 1, so stray classes cost more than loose boundaries; DiceDet targets exactly that during training. (2) Teach anatomy during training instead of fixing outputs afterwards: anatomical rules as post-processing changed nothing, the same rules as a training penalty gave +0.009 and were super-additive with DiceDet. (3) Diversity over strength in the ensemble, chosen and stopped by cross-validation. (4) Decide nothing on a single fold (σ ≈ 0.007 per fold).

Figure 1 (`figures/fig1_pipeline.png`) shows the pipeline; Figure 2 (`figures/fig2_qualitative_oof.png`) shows out-of-fold predictions.

---

## 2. Tasks 1 & 2: segmentation

### 2.1 Network

| | |
|---|---|
| Encoder | ConvNeXt-Large (`convnext_large.fb_in22k_ft_in1k_384`, timm, ImageNet-22k→1k) in 6 of 7 recipes; ConvNeXt-XLarge in one (A2) |
| Decoders | two heads on one encoder: DeepLabV3+ (5 recipes) or Unet++ (2 recipes, decoder channels 256-128-64-32-16), segmentation-models-pytorch 0.5.0 |
| Input | 1024×576 (area resampling; predictions bilinearly upsampled to the original size before the argmax), ImageNet mean/std |
| Parameters | DeepLabV3+/ConvNeXt-L 201 M, Unet++/ConvNeXt-L 228 M, DeepLabV3+/ConvNeXt-XL ≈ 350 M |

### 2.2 Loss

`L = 0.5·L_fine + 0.5·L_merged + 0.25·L_fine→merged (+ AnatomyLoss)`, all Dice terms weighted with the official 3/2/1 class weights (background included).

* **Fine→merged consistency (`L_fine→merged`).** The fine softmax is summed over the fine classes of each merged class and scored with weighted Dice against the merged mask. +0.078 Task 1 / +0.019 Task 2 Dice (p ≤ 0.002), the largest single gain of the project.
* **DiceDet.** Dice plus a hinge per class and image: `relu(0.5 − Dice_c)` if the class is present in the reference, `relu(max_x p_c(x) − 0.5)` if it is absent. The absent-class term drives the *maximum* probability of a class that should not appear below the decision threshold. +0.011.
* **AnatomyLoss.** On the softmax `P`, an adjacency penalty `Σ_{a,b} W[a,b]·P_a(x)·P_b(x+d)` over 4-neighbour pixel pairs (normalised by the detached soft boundary mass) plus an exclusion penalty `Σ_{a<b} E[a,b]·s_a·s_b` on soft image-level presence `s_c = max(avgpool_16(P_c))`; weights 1.0 / 0.1, switched on at epoch 8 with a 5-epoch ramp. `W` (forbidden contacts) and `E` (structures that never co-occur) come either from training-mask statistics (fold-specific, validation cases excluded) or from a **hand-written 3-D anatomy graph** (23 classes: contact / proximity / distant, exclusion when the cranio-caudal gap is ≥ 5 cm; overlaying tissues such as instrument, fat, pleura and blood are excluded). The knowledge graph scores at least as well as the data-derived one (0.693 vs 0.689) and cannot leak. +0.009 alone, **+0.025 together with DiceDet** (one fixes *which* classes appear, the other *where*).

### 2.3 Training

* AdamW 2e-4, weight decay 0.01, 3 warm-up epochs + cosine, **20 epochs**, batch 2 × gradient accumulation 4, AMP. Fold models: best validation epoch; all-data models: final epoch (in 80 % of fold runs the best epoch lay in the last 5, spread 0.0055; SWA/EMA were worse, −0.006).
* **Data handling.** Two byte-identical duplicate frames removed (526 frames used); images that are identical under different station names are kept as annotation variability and always share a fold. RGB masks → class ids via `labelmap.csv`. Task 3 uses the 518 labelled frames.
* **Augmentation** (albumentations, at 1024×576): horizontal flip 0.5 (off in A7, because several classes are left/right-specific); affine scale 0.8–1.25, translate ±10 %, rotate ±25° (p 0.8); elastic (α 60, σ 8) or grid distortion (5 steps, 0.2) (p 0.3); colour jitter 0.3/0.3/0.3/hue 0.15 (p 0.7); gamma 70–140 (p 0.3); Gaussian or motion blur (p 0.3); Gaussian noise σ 0.02–0.08 (p 0.3).
* **Surgical-domain pre-training (encoder + decoder).** For 5 of the 7 recipes the whole network except the class heads is initialised from a model pre-trained at 1024×576 on public laparoscopic scene segmentation: **CholecSeg8k** (13 classes, every 6th frame, 1 232 / 182 frames), **EndoVis 2018** (12 classes, 975 / 150 frames) or both (25-class union); Dice loss, AdamW 2e-4, one-cycle, 12 epochs, best validation checkpoint. **+0.015 on every fold** (0.693 → 0.708–0.710) regardless of the corpus; encoder-only surgical pre-training had given nothing, so the decoder is what carries the domain.
* **Fine-specialised fine-tuning.** Recipe A3 continues the CholecSeg8k-pre-trained DeepLabV3+ with the merged-head loss switched off (lr 5e-5) and is used as a full member (both heads). In cross-validation the best epoch of this fine-tune was the first one on all five folds, so the deployed checkpoints are the 1-epoch versions.

### 2.4 Ensemble

* **Recipes.** Seven recipes were chosen on the cross-validated official score (Dice and normalised Hausdorff distance of both tasks, averaged over the five folds): five surgically pre-trained DeepLabV3+ variants (EndoVis18; CholecSeg8k with ConvNeXt-XL; CholecSeg8k fine-specialised; joint Cholec+EndoVis; CholecSeg8k without horizontal flip) and two ImageNet-only Unet++ models (DiceDet + AnatomyLoss; DiceDet only). Cross-validated with fold models, before TTA and post-processing: **Task 1 Dice 0.7133 / HD 0.2009, Task 2 Dice 0.7350 / HD 0.1888**. Two findings shaped the choice: recipe diversity beats individual strength (the two strongest recipes alone lose by 0.010 to seven ordinary ones; 2 → 4 → 7 recipes gave 0.675 → 0.687 → 0.695 mean Dice), and there is a limit (adding recipes outside the surgically pre-trained group hurt Task 2). A fine-only specialist added with extra weight to the fine head *reduced* Task 2 (0.7338 → 0.7210) and is not deployed.
* **Deployment.** Every recipe is deployed as all-data checkpoints (final epoch); fold models were used only for validation. Seed replicas of a recipe are averaged inside the recipe's slot, then the 7 slots are averaged with equal weight, so the cross-validated recipe weighting is unchanged and each slot only gets a lower-variance prediction (31 checkpoints in 7 slots, Appendix A). The seed replicas added last were trained after the selection and could not be cross-validated.
* **TTA.** Horizontal-flip TTA on 5 of the 7 recipes: Task 2 Dice 0.7350 → 0.7371, HD 0.1888 → 0.1876, Task 1 unchanged (HD +0.001). It is off for A7 (trained without flips) and A2 (ConvNeXt-XL, where TTA hurt in CV).

### 2.5 Post-processing (as submitted, and what we learned too late)

The submitted container applies, in this order: **class-wise probability scaling α** on the fine head before the argmax (19 fine-class factors 0.3–3.8, cross-fitted on out-of-fold predictions of an earlier ensemble, where it gave +0.014 Task 2 Dice), bilinear upsampling to the original resolution, argmax, and **island removal**: for every class, connected components smaller than **0.4 % of the frame** are relabelled with the majority class of their surrounding ring (+0.032 / +0.027 Dice on that earlier ensemble).

Both steps had been validated on Dice only. Re-measured on the final seven recipes with the official Dice **and** normalised Hausdorff distance (fold models, TTA on) on the day of the deadline:

| post-processing | Task 1 Dice / HD | Task 2 Dice / HD |
|---|---|---|
| none | 0.7134 / 0.2021 | 0.7371 / 0.1876 |
| **α + islands < 0.4 % of the frame (submitted)** | **0.7188 / 0.2242** | **0.7383 / 0.2101** |
| islands < 0.4 % of the frame, no α | 0.7188 / 0.2242 | 0.7465 / 0.1974 |
| islands < 0.05 % of the frame, all classes | 0.7177 / 0.2043 | 0.7429 / 0.1869 |
| islands < 1 % of the class's mean area | 0.7179 / 0.2029 | 0.7427 / 0.1851 |
| islands < 2 % of the class's mean area | 0.7193 / 0.2053 | 0.7453 / 0.1848 |
| α only | 0.7138 / 0.2018 | 0.7354 / 0.1933 |

* **Island removal** buys Dice at every threshold, but with a frame-relative threshold the Hausdorff cost grows quickly: the official HD is 1.0 for a class present in only one of prediction and reference, so deleting the only component of a small weight-3 structure that *is* in the reference costs more than any stray island gains. The final ensemble (DiceDet, pre-training) produces few false islands, so at 0.4 % the survivors are mostly true small structures. A threshold scaled to each class's mean area (2 %: right recurrent laryngeal nerve 0.009 %, lymph node 0.07 %, pleura 0.35 % of the frame) keeps the Dice gain with the Hausdorff at or below the no-post-processing value.
* **α** was fitted on an older ensemble and no longer helps on the final one (Task 2 −0.002 Dice, +0.006 HD).

The class-scaled variant was implemented and packaged, but the container that was pushed and submitted still carries the earlier setting (α + 0.4 %). We report the submitted configuration here; the expected difference on the cross-validated Dice − HD objective is about −0.02 (Task 1) and −0.03 (Task 2) relative to the class-scaled setting.

### 2.6 What moved the score, and what did not

**Progress of the segmentation solution.** Official OOF weighted Dice (Task 1 / Task 2) where a full 5-fold OOF evaluation was run; the internal validation score (mean of fine and merged weighted Dice at 1024×576, 5-fold mean unless marked f0) for the loss/pre-training ablations, which were gated on that score.

| Step | Task 1 Dice | Task 2 Dice | val score |
|---|---|---|---|
| EfficientNet-B7 + Unet++, two heads, weighted Dice | 0.576 | 0.604 | |
| + stronger augmentation | 0.579 | 0.625 | |
| MaxViT-B + Unet++ at 1024×576 | 0.589 | 0.634 | |
| + fine→merged consistency loss | 0.666 | 0.653 | |
| 5-recipe ensemble of the MaxViT era | 0.673 | 0.682 | |
| ConvNeXt-L + Unet++, Dice + consistency loss (new baseline) | | | 0.665 |
| + DiceDet | | | 0.676 (f0) |
| + AnatomyLoss, data-derived graph | | | 0.679 (f0) |
| + DiceDet + AnatomyLoss, data-derived graph | | | 0.689 |
| + DiceDet + AnatomyLoss, 3-D knowledge graph | | | 0.693 |
| DeepLabV3+, DiceDet + AnatomyLoss, CholecSeg8k encoder+decoder pre-training | | | 0.708 |
| same with EndoVis18 / both corpora / ConvNeXt-XL / ToolPaste / boundary term | | | 0.706–0.710 |
| 7-recipe ensemble (ConvNeXt-L era, fold models) | 0.690 | 0.701 | |
| + α (fine) + island removal (Dice only; later found to raise HD, see 2.5) | 0.717 | 0.732 | |
| final 7 recipes A1–A7 (fold models) | 0.713 (HD 0.201) | 0.735 (HD 0.189) | |
| + horizontal-flip TTA on 5 recipes | 0.713 (HD 0.202) | 0.737 (HD 0.188) | |
| + α + islands < 0.4 % = **submitted configuration** | **0.719 (HD 0.224)** | **0.738 (HD 0.210)** | |
| + class-scaled island removal, no α (implemented, not in the submitted image) | 0.719 (HD 0.205) | 0.745 (HD 0.185) | |

**Did not work (each on ≥ 2 folds):** pseudo-labelled frames generated by a first-last-frame video model (Wan2.1-FLF2V) between two station views with SAM3 label propagation — the propagated labels were good (Dice 0.76–0.79 against the real opposite-end mask) but training on them cost −0.016; a frozen DINOv3-7B feature branch (−0.005); SurgeNetXL encoder (−0.056); CAFormer, SE-ResNeXt, SegFormer-B5, DPT-L (−0.01 to −0.09); random-init UPerNet / FPN / MAnet decoders (−0.07 to −0.16); the ADE20k-pre-trained UPerNet-Swin-L (best on fold 0, worse on 3 of 5); 1536×864 input (−0.011 / −0.028); Dice+CE (−0.030), size-weighted Dice (−0.031), focal-Tversky, RMI, Hausdorff-DT; a boundary loss that helps ConvNeXt (+0.004) but costs −0.054 on Swin; anatomy rules or component-count limits as post-processing (0 / −0.007); the frame-relative 0.4 % island threshold and class-wise scaling on the final ensemble, which the submitted image nevertheless uses (Section 2.5); SWA / EMA (−0.006); multi-task training with the Task 3 head (−0.032); ToolPaste instrument-pasting augmentation (helps large organs, hurts thin weight-3 structures); dropping the horizontal flip is +0.010 alone but not additive with DiceDet.

---

## 3. Task 3: lymph-node station visibility

### 3.1 Idea

Classifying the neighbourhood of a lymph-node component identifies the station at chance level (0.269 vs 0.244), and 179 of 518 labelled frames contain no lymph-node pixels at all. An inventory of the *whole frame* — which structures are visible, how large, where and next to what — does carry the signal (0.344), which is how a surgeon reads a station. So the predictor reads our own Task 1/2 masks; at training time these are out-of-fold masks, so the feature distribution matches test time.

### 3.2 Features (1741, from the predicted fine + merged masks, sub-sampled ×4)

* **Anatomy inventory (315):** for each of the 30 fine + 15 merged foreground classes: area ratio, presence, centroid x/y, log(1 + connected components), bounding-box width/height.
* **Anatomy context (202):** class-to-class contact matrix, what surrounds the lymph-node and fatty-tissue regions, pairwise proximity.
* **Coarse occupancy grid (1224):** 6×12 cells × classes — restores layout at a resolution too coarse to memorise frames.
* plus the global-average-pooled (1536-d) last-stage features of five frozen fold models of one segmentation recipe (DeepLabV3+ / ConvNeXt-L, ImageNet-only, Dice loss; not one of the deployed segmentation recipes).

### 3.3 Models

| Branch | Input | Model | Count |
|---|---|---|---|
| GBDT | 1741 mask features | XGBoost multi-output tree (`multi_strategy="multi_output_tree"`: one tree with a 14-d leaf shared by all stations), depth 6, 1000 trees, lr 0.02, colsample 0.5, subsample 0.8, λ₂ 1 | 3 seeds × 5 folds = 15 |
| MLP | encoder GAP ⊕ 315 inventory | 256 → ReLU → Dropout 0.3 → 14 sigmoid; AdamW 1e-3, wd 1e-2, cosine, 60 epochs, BCE, standardised features, mirrored frame as a second sample | 3 seeds × 5 folds = 15 |

Probability = 0.7·XGBoost + 0.3·MLP (the weight 0.7 is chosen in every fold when selected nested on the other four). Each branch uses the feature set that suits it: the 1741 features help trees and hurt the MLP (AUROC 0.9005 with 315 vs 0.8834 with 1741 features), and adding the GAP features to the tree model hurts it (−0.021 F1). **Calibration:** because F1 is binarised at 0.5, each station's probability is mapped piecewise-linearly so that its optimal OOF threshold (one per station) lands on 0.5; the map is monotone, so AUROC is unchanged.

### 3.4 Results (OOF, 518 frames, official `evaluate_cls`)

| Predictor | F1 @ 0.5 | AUROC |
|---|---|---|
| Frozen encoder GAP → MLP (image only) | 0.7405 | 0.8885 |
| Anatomy inventory → LightGBM (masks only) | 0.7560 | 0.8982 |
| MLP on GAP ⊕ inventory | 0.7700 | 0.9015 |
| MLPs + per-station LightGBM on 1741 features, blended (earlier image) | 0.7890 | 0.9165 |
| XGBoost multi-output tree on 1741 features alone (3 seeds) | 0.7917 | 0.9208 |
| **Final: 0.7·XGBoost + 0.3·MLP[GAP ⊕ 315], calibrated** | **0.8056** | **0.9253** |
| same with 2–6 encoders in the MLP branch | 0.804–0.807 | 0.925–0.926 |

The final blend beats the earlier LightGBM configuration on every fold (+0.022 / +0.016 / +0.029 / +0.021 / +0.045 on F1 + AUROC). A model trained on predicted masks does not degrade when given ground-truth masks, so better segmentation cannot hurt Task 3.

**Did not work:** classifying the lymph-node region itself (chance); a Task 3 head trained jointly inside the segmentation network (AUROC −0.016, segmentation −0.032); encoder features added to the tree model (−0.021 F1); logistic regression instead of boosting (−0.027); stacking the CNN probability as a feature instead of blending (−0.014); 29 XGBoost hyper-parameter variants (all within seed noise ±0.004); dropping the occupancy grid (−0.011 F1 + AUROC); free greedy selection over 78 individual members (worse than the fixed blend: 518 frames do not support it).

---

## 4. Container and runtime

* Base `pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime`; `segmentation-models-pytorch 0.5.0`, `timm 1.0.22`, `xgboost 3.2.0`, OpenCV headless; offline (`--network none`, HF offline flags); all weights fp16; image 34.9 GB.
* **Frame-outer, chunked inference:** frames in chunks of 12 (mini-batch 4); each PNG decoded once and reused by all members; softmax accumulated on the GPU for the chunk only; PNGs and `task3.csv` rows written and flushed after every chunk. Networks are built once per architecture (3 distinct) and members swapped by copying fp16 weights; weights cached in host RAM up to min(30 % of free memory, 12 GB).
* **Wall-clock budget controller:** the container measures its own throughput on the first frames and, before each chunk, sizes the ensemble to the time remaining (`K = (remaining time / remaining frames − post-processing) / per-member cost`); members are dropped from the end of the CV-ranked priority list (newest seed replicas first, every slot keeps weight 1). The submitted image applies α (`alpha.json`) and island removal at `ISLAND_PPM=4000`; the per-class threshold support in `process.py` is present but no `island_ppm.json` is packaged. The number of Task-3 encoders is fixed for the whole run so that probabilities are on one scale across frames. Without a GPU the same mechanism shrinks to one member and still completes.
* **Runtime:** 139 frames at the training-set resolution mix on one Quadro RTX 8000: 42.5 min for 25 checkpoints with TTA on 23 and 15 Task-3 encoders (previous image); the final image runs 31 checkpoints (TTA on 27) and 5 encoders, about 20 % more segmentation work. The organisers' hardware ran our first image 1.7–2.4× slower than ours, giving roughly 90–120 min against the 417 min budget; the controller's internal deadline is 250 min.
* Validation before submission: container test on a GPU host with `--network none`, read-only `/input`, a 4K frame included; file count, names, sizes, colour tables and the Task 3 CSV checked; the official `01_evaluate_challenge.py` run on the container output; container-vs-host pixel agreement ≥ 99.99 %; each exported Task 3 fold model reproduces its out-of-fold probabilities.

## 5. External data and pre-trained weights (all public)

ImageNet-22k/1k ConvNeXt-L/XL (timm). **CholecSeg8k** (Hong et al., 2020) and **EndoVis 2018 Robotic Scene Segmentation** (Allan et al., 2020) frames and masks for encoder+decoder pre-training. Instrument cut-outs from **SAR-RARP50** and **SurgToolLoc 2022** were used only in an unselected augmentation candidate. No private data and no weights pre-trained on private data; generated/pseudo-labelled data were evaluated but are not part of the submission.

## 6. Compute

4 × Quadro RTX 8000 (48 GB) and one node with RTX 4090 / RTX A4000. One 20-epoch all-data run ≈ 85 min on an RTX 8000 shared by two jobs; surgical pre-training 30–40 min; Task 3 heads train in seconds on cached features. About 260 fold-level runs were used for the ablations above.

---

## 7. Discussion

**What we learned.** Match the loss to the metric's absent-class rule (DiceDet), and validate every post-processing step on *both* halves of the metric: island removal at 0.4 % of the frame looked like +0.03 on Dice and turned out to cost more on Hausdorff, because the same absent-class rule punishes a deleted true structure with the worst possible distance; a threshold scaled to each class's typical size keeps the gain without the cost, but this was found only after the final image was built, and the submitted container still uses the frame-relative threshold with α. Teach anatomy during training; do not patch outputs. Pre-train the decoder as well as the encoder on public surgical data; the corpus matters less than having decoder weights that have seen laparoscopic scenes. Ensemble for diversity, validate by cross-validation, and stop when weaker recipes start to hurt. For Task 3, station visibility is a question of *what anatomy is on screen*, so re-use the segmentation, keep the tabular model small, and calibrate for the fixed threshold. Never decide on one fold.

**Expected behaviour.** Robust to the unseen centre (centre-stratified folds, surgically pre-trained members); weakest on the thin weight-3 structures (recurrent laryngeal nerves, ligaments, thoracic duct, bronchial artery: per-class Dice 0.25–0.30) and on heavily bleeding or very close-up frames. Task 3 inherits systematic segmentation errors, and its calibration assumes a test-set station prevalence similar to training.

**What we would do differently.** Verify the container on a GPU-enabled Docker runtime from the first test submission (our first image silently fell back to CPU on the evaluation machine and took 9 hours); start surgical pre-training and the metric-aligned loss on day one; measure Hausdorff-oriented terms with the official implementation from the beginning; and spend the compute used on generative pseudo-data on more diverse real-data recipes instead.

---

## 8. References
1. Liu Z. et al. A ConvNet for the 2020s (ConvNeXt). CVPR 2022.
2. Chen L.-C. et al. Encoder-decoder with atrous separable convolution for semantic image segmentation (DeepLabV3+). ECCV 2018.
3. Zhou Z. et al. UNet++: A nested U-Net architecture for medical image segmentation. DLMIA/ML-CDS 2018.
4. Tu Z. et al. MaxViT: Multi-axis vision transformer. ECCV 2022.
5. Tan M., Le Q. EfficientNet: Rethinking model scaling for convolutional neural networks. ICML 2019.
6. Liu Z. et al. Swin Transformer: Hierarchical vision transformer using shifted windows. ICCV 2021.
7. Xiao T. et al. Unified perceptual parsing for scene understanding (UPerNet). ECCV 2018.
8. Xie E. et al. SegFormer: Simple and efficient design for semantic segmentation with transformers. NeurIPS 2021.
9. Ranftl R. et al. Vision transformers for dense prediction (DPT). ICCV 2021.
10. Yu W. et al. MetaFormer baselines for vision (CAFormer). IEEE TPAMI 2024.
11. Siméoni O. et al. DINOv3. arXiv:2508.10104, 2025.
12. Jaspers T. J. M. et al. Scaling up self-supervised learning for improved surgical foundation models (SurgeNetXL). arXiv:2501.09436, 2025.
13. Wan Team. Wan: Open and advanced large-scale video generative models (Wan2.1). arXiv:2503.20314, 2025.
14. Carion N. et al. SAM 3: Segment Anything with Concepts. 2025.
15. Chen T., Guestrin C. XGBoost: A scalable tree boosting system. KDD 2016.
16. Ke G. et al. LightGBM: A highly efficient gradient boosting decision tree. NeurIPS 2017.
17. Hong W.-Y. et al. CholecSeg8k: A semantic segmentation dataset for laparoscopic cholecystectomy based on Cholec80. arXiv:2012.12453, 2020.
18. Allan M. et al. 2018 Robotic Scene Segmentation Challenge (EndoVis 2018). arXiv:2001.11190, 2020.
19. Psychogyios D. et al. SAR-RARP50: Segmentation of surgical instrumentation and action recognition on robot-assisted radical prostatectomy challenge. arXiv:2401.00496, 2023.
20. Zia A. et al. Surgical tool classification and localization: results and methods from the MICCAI 2022 SurgToolLoc challenge. arXiv:2305.07152, 2023.
21. Tiger SQ-AI Challenge evaluation code (NCT/TSO Dresden). https://www.synapse.org/Synapse:syn74209386.

## 9. Authors' statement
S.K.: conception, data analysis, method development, training, container engineering, evaluation, write-up and video. A.K., H.M.: method development.

## 10. Acknowledgements
This project was financially and computationally supported by Jmees Inc. We thank the Tiger SQ-AI Challenge organisers (NCT/UCC Dresden, TSO) for the dataset, the public evaluation code and the early Docker compatibility feedback. Parts of the experiment management and code were developed with the help of AI coding assistants (Anthropic Claude Code); all design decisions and results were verified by the authors.

---

## Appendix A. Segmentation ensemble in the submitted container

Every checkpoint is trained on all 526 frames (final epoch) at 1024×576 with the loss of Section 2.2 (`L_fine→merged` weight 0.25 throughout, DiceDet in every recipe). Checkpoints of the same recipe are averaged inside the slot; the 7 slots are averaged with equal weight. TTA = horizontal-flip test-time augmentation. The code name is the directory name in the public repository.

| # | encoder | decoder | surgical pre-training | AnatomyLoss graph | augmentation / note | TTA | checkpoints (seeds) | code name |
|---|---|---|---|---|---|---|---|---|
| A1 | ConvNeXt-L | DeepLabV3+ | EndoVis18 | 3-D knowledge | standard | on | 17 (42, 43, 44, 45, 46, 48, 49, 50, 52, 53, 55, 56, 57, 62, 64, 66, 68) | `q_endovis18_dlv3` |
| A2 | ConvNeXt-XL | DeepLabV3+ | CholecSeg8k | 3-D knowledge | standard | off | 2 (42, 43) | `r_xl` |
| A3 | ConvNeXt-L | DeepLabV3+ | CholecSeg8k (via base) | 3-D knowledge | 1-epoch fine-specialised fine-tune (merged loss off) of the CholecSeg8k DeepLabV3+ base of the same seed | on | 2 (42, 43) | `r_ft_fine_ep1` |
| A4 | ConvNeXt-L | DeepLabV3+ | CholecSeg8k + EndoVis18 (25-class union) | 3-D knowledge | standard | on | 2 (42, 43) | `q_both_dlv3` |
| A5 | ConvNeXt-L | Unet++ | – (ImageNet only) | data-derived (training-mask statistics) | standard | on | 4 (42, 43, 43 second run, 44) | `k_dicedet_rules` |
| A6 | ConvNeXt-L | Unet++ | – (ImageNet only) | – | standard | on | 2 (42, 43) | `l_dicedet` |
| A7 | ConvNeXt-L | DeepLabV3+ | CholecSeg8k | 3-D knowledge | standard, **no horizontal flip** | off | 2 (42, 43) | `r_nohflip` |

Total: 31 checkpoints in 7 slots. Task 3 package: 15 XGBoost boosters, 15 MLPs, the 5 fold encoders of the DeepLabV3+ / ConvNeXt-L feature model, `meta.json` (feature lists, blend weight, 14 calibration thresholds).

## Appendix B. Figures (originals uploaded to the Synapse "Files" section)
* `figures/fig1_pipeline.png` — pipeline overview (Figure 1).
* `figures/fig2_qualitative_oof.png` — out-of-fold predictions of a 7-recipe fold ensemble on three held-out frames, fine and merged (Figure 2).
