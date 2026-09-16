# TIGER SQ-AI Challenge — Training Data Quality Report

**From:** Shunsuke Kikuchi (participant)
**Date:** 2026-09-09
**Data version:** the current training release (40 cases, 524 annotated frames,
`lymph_node_station_visibility.csv` with 518 rows), downloaded from Synapse syn74209386 after your latest
update announcement.

We ran an integrity check over the training release before our final training runs. Three things came up
that we believe are unintended, and two questions we would like to ask. Everything your announcement already
covers is excluded, except where our findings appear to contradict it (Issue 2).

**Scope.** Our local copy contains 528 PNG files: your 524 annotated frames plus four extra files
(`center_3_case_3_na_station_1.png`, `center_7_case_3_na_station_1.png`,
`center_2_case_8_12L_frame_84.png`, `center_2_case_8_6L_frame_1733.png`). We believe we have the correct
version.

**Metrics used below.** *Pixel Accuracy* is the fraction of pixels on which two published annotations assign
the same class. *macro Dice* is the unweighted mean of the per-class Dice over every class present in either
annotation. Both are symmetric — neither annotation is treated as ground truth. Pixel Accuracy is dominated
by the largest classes, so macro Dice is the more informative of the two here, and the closer analogue of the
challenge's weighted Dice. No model or prediction is involved anywhere in this report.

---

## Summary

| # | Issue | Severity |
|---|-------|----------|
| 1 | **Five pairs of byte-identical images are published under two different station names, and each pair carries two conflicting segmentation masks.** Both members of every pair are fully valid station frames (both have a visibility row). Mean macro Dice between the two annotations: **0.687 fine / 0.774 coarse** | **High** |
| 2 | **The `na_station_1` note appears inconsistent with the files as shipped.** Each `na_station_1` frame is byte-identical to a station-named frame in the same case, and that station-named frame is still in the release | **High** |
| 3 | **`center_7_case_2_13R.png` — the segmentation mask appears incomplete** (93.7 % background, one annotated class) | **High** |

Checks that came back clean: every mask pixel uses a colour present in `labelmap.csv` (0 unknown colours
across all 528 fine and coarse masks); `lymph_node_station_visibility.csv` has no duplicated
`(case, station)` keys and no rows without a matching image; in all 518 rows the annotated station itself is
listed as visible.

---

## Issue 1 — Byte-identical images with conflicting masks

**Method.** MD5 over the raw bytes of every file in `images/`, then per-pair comparison of `masks_fine/` and
`masks_coarse/` after converting RGB to class IDs through `labelmap.csv`.

**Result.** Of the 524 station-named frames, 517 are distinct images. Seven pairs are byte-identical. Two of
those seven involve a frame with no visibility row and are covered by Issue 2 / Issue 3. The remaining
**five pairs are two fully valid station frames each — both members have a row in
`lymph_node_station_visibility.csv` — but the same image carries two different masks.**

| Pair (identical image bytes) | fine: Pixel Acc. | fine: macro Dice | coarse: Pixel Acc. | coarse: macro Dice |
|---|---|---|---|---|
| `center_1_case_11_6L` = `center_1_case_11_7L` | 0.954 | 0.900 | 0.959 | 0.913 |
| `center_1_case_3_11L` = `center_1_case_3_13R` | 0.871 | 0.722 | 0.963 | 0.890 |
| `center_1_case_10_6R` = `center_1_case_10_7R` | 0.915 | 0.651 | 0.942 | 0.745 |
| `center_1_case_15_6R` = `center_1_case_15_7R` | 0.880 | 0.613 | 0.932 | 0.680 |
| `center_1_case_12_6R` = `center_1_case_12_7R` | 0.892 | 0.547 | 0.923 | 0.644 |
| **Mean** | **0.902** | **0.687** | **0.944** | **0.774** |

Merging the fine classes into the 16 coarse groups recovers part of the disagreement (macro Dice 0.687 →
0.774), so some of it is confusion between fine classes inside one coarse group. Most of it survives the
merge, i.e. **the two annotations also disagree at the Task 1 level.**

The differences are not boundary jitter — they are class-level disagreements on the same pixels. Taking
`center_1_case_12_6R` vs `center_1_case_12_7R` as an example (figure `center_1_case_12_6R__VS__7R.png`):

| Fine class | weight | area in the `6R` mask | area in the `7R` mask | Dice |
|---|---|---|---|---|
| 25 Resection area | 1 | 4.58 % | 0.00 % | 0.000 |
| 20 Fatty tissue | 1 | 1.49 % | 4.61 % | 0.409 |
| 4 Right main bronchus | 2 | 0.00 % | 3.17 % | 0.000 |
| 14 Right vagal nerve | **3** | 1.18 % | 0.00 % | 0.000 |
| 15 Aorta | 2 | 0.00 % | 0.52 % | 0.000 |
| 24 Pool of blood | 1 | 0.00 % | 0.25 % | 0.000 |
| 3 Trachea | 2 | 5.33 % | 1.99 % | 0.525 |

The same central structure is labelled *Resection area* in one file and *Fatty tissue* in the other, and the
weight-3 class *Right vagal nerve* is present in only one of the two annotations. In these five cases the
visibility rows of the two members are identical, so the two annotations were made from the same image with
the same set of visible stations.

We have attached one figure per byte-identical group. Each figure has one row per published filename; each
row shows the image (identical bytes in both rows), the Task 1 coarse mask and the Task 2 fine mask published
under that filename, with Pixel Accuracy and macro Dice between the two rows given per task, and both
visibility rows printed at the bottom.

---

## Issue 2 — The `na_station_1` note does not match the files as shipped

Your announcement states:

> Two training frames are published as **extra frames** rather than as a station's selected frame, **because
> the chosen frame did not in fact show the station it was selected for**: `center_3_case_3_na_station_1.png`,
> `center_7_case_3_na_station_1.png`. These carry no row in `lymph_node_station_visibility.csv`.

As distributed, each of those two files is **byte-identical to a station-named frame in the same case, and
that station-named frame is still present in the release**:

| Extra frame | Byte-identical to | Does that station-named file have a visibility row? |
|---|---|---|
| `center_3_case_3_na_station_1.png` | `center_3_case_3_12R.png` | **No** |
| `center_7_case_3_na_station_1.png` | `center_7_case_3_10L.png` | **No** |

The station-named copies also have no visibility row — which is exactly the property the note attributes to
the extra frames. Read together, the most consistent explanation is that the frame was **demoted from its
station, its visibility row removed, and republished as `na_station_1`, but the original station-named copy
was never withdrawn from the release.**

If that is what happened, then `center_3_case_3_12R.png` and `center_7_case_3_10L.png` are frames you have
determined **do not show 12R and 10L**, yet a participant parsing filenames — which the note explicitly
anticipates — will treat them as valid 12R and 10L frames. That is the failure mode the note was written to
prevent.

**The same pattern appears to extend to four more files.** Exactly six of the 524 station-named frames have
no visibility row (524 − 6 = 518, matching the row count exactly):

| File | Byte-identical to another frame? |
|---|---|
| `center_3_case_3_12R.png` | yes — `center_3_case_3_na_station_1.png` |
| `center_7_case_3_10L.png` | yes — `center_7_case_3_na_station_1.png` |
| `center_7_case_3_12L.png` | yes — `center_7_case_3_12R.png` (masks differ, macro Dice 0.751) |
| `center_7_case_2_13R.png` | yes — `center_7_case_2_9.png` (see Issue 3) |
| `center_7_case_2_13L.png` | no |
| `center_2_case_7_10L.png` | no |

Every one of the six is a frame whose station assignment looks withdrawn. Could you confirm whether these six
should be treated as demoted frames whose filename station is not to be trusted? If so, we would suggest
either removing them from the release or listing them explicitly, since the current note names only two of the
six. Note that this is not simply the pending Task 3 annotation work you described: the six form a coherent
group, all of them either byte-identical duplicates or otherwise anomalous.

---

## Issue 3 — `center_7_case_2_13R.png` mask appears incomplete

`masks_fine/center_7_case_2_13R.png` is **93.7 % background** and contains exactly one non-background class
(*Instrument*). Across the training set the background fraction has a median of 18.0 % and a 90th percentile
of 41.6 %; the next-highest file is 43.7 %. This file is an isolated outlier.

The same image is also published as `center_7_case_2_9.png`, and that copy carries a complete annotation with
15 classes, including three weight-3 classes (*Right inferior pulmonary ligament*, *Right vagal nerve*,
*Lymph node*). See figure `center_7_case_2_13R__VS__9.png`.

Our reading is that the `13R` annotation was never completed. Consistently with Issue 2, this file also has
no visibility row.

---

## Two questions

**1. Is the segmentation ground truth a function of the image alone, or is it conditioned on the station the
frame was selected for?**

This is what Issue 1 really raises. In those five pairs, identical pixels receive different class labels
depending on which station the file is named after, with a macro Dice between the two annotations of 0.687
(fine) and 0.774 (coarse), while their visibility rows are identical. If the annotation scope is
intentionally station-dependent — for example if annotators label the structures relevant to the named
station — then an image-only model has an irreducible error floor, and it would help all participants to know
that. If it is not intentional, these five pairs are inconsistent labels; we would suggest reconciling them
and checking whether the same duplication exists in the test set, where it would directly affect scoring.

**2. May the station identifier in the filename be used as a model input for Task 1 and Task 2?**

The rules state clearly that it must not be used for Task 3, and we have complied with that. The Task 1 / 2
case is not addressed, and the answer changes our modelling approach given question 1.

---

## Attachments

11 figures (one per byte-identical group), `summary.csv` with the per-pair Pixel Accuracy / macro Dice and
visibility rows, and `mask_quality_scan.csv` with the background fraction and annotated-class count for all
528 masks.
