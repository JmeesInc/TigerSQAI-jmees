# TigerSQAI — Team Jmees solution for the Tiger SQ-AI Challenge (EndoVis @ MICCAI 2026)

Code for our submission to the [Tiger SQ-AI Challenge](https://www.synapse.org/Synapse:syn74209386/wiki/639462)
(Tasks 1 and 2: merged / fine anatomical segmentation of thoracoscopic esophagectomy frames; Task 3: lymph-node
station visibility). The method write-up is in [`submit/writeup/writeup_draft.md`](submit/writeup/writeup_draft.md);
the submitted container is `docker.synapse.org/syn77311180/tigersqai26_shunsuke:v13`.

**Authors:** Shunsuke Kikuchi, Atsushi Kouno, Hiroki Matsuzaki (Jmees Inc.)

## Solution in one paragraph

Tasks 1 & 2 are solved by one ConvNeXt-Large network with two decoders (merged 15 + bg, fine 30 + bg). Seven training
recipes are averaged in softmax space (31 all-data checkpoints, seed replicas averaged inside their recipe slot) with
horizontal-flip TTA on five of them. The recipes combine a fine→merged consistency loss, a metric-aligned hinge on absent
classes (**DiceDet**), an anatomical-plausibility penalty from a hand-written 3-D anatomy graph (**AnatomyLoss**), and
encoder+decoder pre-training on public laparoscopic segmentation data (CholecSeg8k, EndoVis 2018). Task 3 reads our own
predicted masks: 1741 features describing which structures are on screen, how large, where and next to what feed a
multi-output XGBoost, blended with a small MLP on frozen encoder features, and calibrated for the fixed 0.5 threshold.
No station name or file name is used. Five-fold case-grouped cross-validation with the official evaluation code:
Task 1 Dice 0.719 / HD 0.224, Task 2 Dice 0.738 / HD 0.210, Task 3 F1 0.806 / AUROC 0.925 (see the write-up for the
post-processing caveat).

## Repository layout

| Path | Content |
|---|---|
| `submit/writeup/` | Method write-up (`writeup_draft.md`), figures, video decks scripts, organiser e-mail draft |
| `submit/v006_a23/` | **Submitted container**: `Dockerfile`, `process.py` (inference: chunked, budget-controlled, TTA, island removal), `model_def.py`, `t3_features.py`, `export.py`, `build.sh`, `test.sh` |
| `submit/v001…v005/` | Earlier container versions (kept for the record) |
| `submit/SUBMISSIONS.md` | History of every container version and its cross-validated scores |
| `workspace/preprocess/` | RGB mask → class-id conversion and 1024×576 caching |
| `workspace/fold/` | Case-grouped, centre-stratified 5-fold assignments (`v2` is the one used) |
| `workspace/expA23_sweep/` | **Final segmentation code**: `train.py`, `losses.py` (DiceDet, boundary, …), `anatomy_rules.py` (AnatomyLoss), `model.py` (dual-head smp models), `pretrain_ext.py` (CholecSeg8k / EndoVis18 pre-training), `configs/` (every recipe), ensemble selection (`ens_greedy_mgpu.py`, `ens_select_cv.py`), post-processing evaluation (`island_postproc.py`, `score_final_postproc.py`), official-metric evaluation (`eval_official_hd.py`) |
| `workspace/anatomy_graph/` | Anatomy graphs for AnatomyLoss: data-derived (`build_graph.py`, `rules.py`) and the hand-written 3-D knowledge graph (`knowledge_graph_3d.py`) |
| `workspace/expT04_task3_sweep/` | **Final Task 3 code**: mask features (`features_from_masks.py`, `features_anatomy.py`, `features_grid.py`), models and search (`t3_search.py`, `t3_search2.py`, `t3_xgb_sweep.py`), export (`export_t3_v12.py`) |
| `workspace/expA00…expA22`, `expB*`, `expS*`, `expT01…T03` | Earlier experiments in chronological order (baselines, MaxViT, DINOv3, pseudo-labels, …); see `claudeSummary.md` for the scoreboard |
| `daily_reports/` | Day-by-day experiment log (Japanese) |
| `reference/tigersqai_challenge/` | Not included: the organisers' official evaluation code, fetched from Synapse |

Weights, data, checkpoints, generated masks and Docker images are not in the repository (see `.gitignore`).

## Reproducing the submission

Requirements: Python ≥ 3.10, PyTorch 2.5, `segmentation-models-pytorch 0.5.0`, `timm 1.0.22`, `albumentations`,
`lightgbm`, `xgboost 3.2`, `opencv-python`, and the official evaluation code in `reference/tigersqai_challenge/`.
Set `data/` to the challenge data directory (images, `masks_fine/`, `masks_coarse/`, `labelmap.csv`,
`lymph_node_station_visibility.csv`).

```bash
# 1. labels and 1024x576 cache
python workspace/preprocess/build_labels.py
# 2. folds (already committed as workspace/fold/v2/folds.csv)
# 3. surgical-domain pre-training (public CholecSeg8k / EndoVis 2018 must be downloaded separately)
python workspace/expA23_sweep/pretrain_ext.py --data cholec   --arch deeplabv3plus
python workspace/expA23_sweep/pretrain_ext.py --data endovis18 --arch deeplabv3plus
# 4. fold models of a recipe (for validation) and its all-data model (for deployment)
python workspace/expA23_sweep/train.py --fold 0 --config workspace/expA23_sweep/configs/expA23_q_endovis18_dlv3.yaml
python workspace/expA23_sweep/train.py --fold 0 --config workspace/expA23_sweep/configs/expA23_full_q_endovis18_dlv3.yaml
# 5. Task 3 features from out-of-fold masks, model search and export
python workspace/expT04_task3_sweep/export_t3_v12.py
# 6. package and build the container
python submit/v006_a23/export.py --prefer-latest --folds 0 --members A23:expA23_full_q_endovis18_dlv3 ...
cd submit/v006_a23 && VER=v13 ./build.sh && VER=v13 ./test.sh 6
```

The seven deployed recipes and their configuration files are listed in Appendix A of the write-up.

## External data and pre-trained weights (all public)

ImageNet-22k/1k ConvNeXt weights (timm); CholecSeg8k [Hong et al., 2020] and EndoVis 2018 Robotic Scene Segmentation
[Allan et al., 2020] for encoder+decoder pre-training; SAR-RARP50 and SurgToolLoc instrument cut-outs were evaluated in an
augmentation candidate that is not part of the submission. No private data were used.

## Acknowledgements

We thank the Tiger SQ-AI Challenge organisers (NCT/UCC Dresden, TSO) for the dataset, the public evaluation code and the
early Docker compatibility feedback. Parts of the experiment management and code were developed with the help of AI
coding assistants; all design decisions and results were verified by the authors.
