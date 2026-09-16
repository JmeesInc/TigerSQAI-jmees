**Title:** Duplicate images with conflicting masks in the training release

Hi Max and team,

Before our final training runs we checked the integrity of the training release and found three things we
think are unintended. Figures are attached, one per affected pair.

### 1. Five byte-identical images are published under two station names, with conflicting masks

By MD5, the 524 station-named frames are only **517 distinct images**. In five of the duplicate pairs, both
members are valid station frames — both have a visibility row, and the two rows are *identical* — yet the
same image carries two different masks:

- `center_1_case_11_6L` = `center_1_case_11_7L`
- `center_1_case_3_11L` = `center_1_case_3_13R`
- `center_1_case_10_6R` = `center_1_case_10_7R`
- `center_1_case_15_6R` = `center_1_case_15_7R`
- `center_1_case_12_6R` = `center_1_case_12_7R`

Scoring one annotation against the other with the challenge metric, the five pairs average **Task 1 Dice
0.830 / normalised HD 0.121** and **Task 2 Dice 0.839 / normalised HD 0.110**. These are class-level disagreements rather than boundary jitter —
whole structures are assigned to different classes, including weight-3 classes present in only one of the two
annotations. The attached figures show this directly.

### 2. Two `na_station_1` frames duplicate a station frame that is still in the release

`center_3_case_3_na_station_1.png` is byte-identical to `center_3_case_3_12R.png`, and
`center_7_case_3_na_station_1.png` to `center_7_case_3_10L.png`. Both station-named copies are still in the
release and, like the extra frames, have no visibility row — so it looks as if the frame was demoted from its
station but the station-named copy was never withdrawn.

More generally, exactly six station-named frames have no visibility row (524 − 6 = 518), and all six are
either byte-identical duplicates or otherwise anomalous: `center_3_case_3_12R`, `center_7_case_3_10L`,
`center_7_case_3_12L`, `center_7_case_2_13R`, `center_7_case_2_13L`, `center_2_case_7_10L`. Should all six be
treated as frames whose filename station is not to be trusted?

### 3. `center_7_case_2_13R.png` — the mask looks incomplete

93.7 % background with one non-background class (*Instrument*), against a dataset median of 18.0 % and a
next-highest file of 43.7 %. The same image, published as `center_7_case_2_9.png`, has a complete 15-class
annotation.

---

### Two questions

**a) Is the segmentation ground truth a function of the image alone, or is it conditioned on the station the
frame was selected for?** In those five pairs the pixels and the visibility rows are identical, yet the labels
differ by station name. If this is intentional, an image-only model has an irreducible error floor and it
would help to know. If not, it would be worth checking whether the same duplication exists in the test set.

**b) May the station identifier in the filename be used as a model input for Tasks 1 and 2?** The rules are
explicit for Task 3 and we have complied; the Task 1 / 2 case is not addressed, and the answer depends on (a).

Thanks,
Shunsuke Kikuchi
