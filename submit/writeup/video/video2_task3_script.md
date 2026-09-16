# 3-min video #2 — Task 3 (lymph-node station visibility)

**Team Jmees.** 目標 3:00。

---

## Slide 1 — Title and the rule (0:00–0:20, 50 words)
（タイトル + 「ファイル名の station を使わない」制約を明示）

> Lymph-node station visibility for the Tiger SQ-AI challenge. Team Jmees.
> Fourteen stations, multi-label, one score per frame. The rule that shapes the
> method: the station name in the file name may not be used. Our model never sees
> a file name.

## Slide 2 — The finding that set the design (0:20–1:00, 95 words)
（比較図: リンパ節成分の局所パッチ 0.269（ほぼチャンス） vs フレーム全体の構造在庫 0.344）

> We started from the obvious approach: look at the lymph-node region and classify
> it. That does not work. A local patch around a lymph-node component identifies
> the station at chance level. What does carry the signal is the *rest* of the
> frame: which anatomical structures are visible at all, and where. A surgeon
> knows the station from the aorta, the azygos vein, the bronchus and the nerve in
> view, not from the node itself. So we classify the station from an inventory of
> the whole scene.

## Slide 3 — Features (1:00–1:50, 115 words)
（3 ブロック図: 在庫 315 / 解剖文脈 202 / 粗グリッド 1224 = 1741 次元 + encoder GAP）

> The input is our own predicted masks, not the ground truth, and at training time
> they are out-of-fold masks, so the feature distribution matches test time.
> From each mask we take an inventory: for every class, area fraction, a presence
> flag, centroid, number of connected components, and bounding-box size.
> On top of that, anatomy context: which classes touch which, what surrounds the
> lymph-node and the fatty-tissue regions, and proximity between structures.
> And a coarse six-by-twelve occupancy grid, which restores the spatial layout the
> inventory throws away. Seventeen hundred features in total, plus the pooled
> encoder features of the segmentation backbones.

## Slide 4 — Models and calibration (1:50–2:30, 90 words)
（MLP 45 + LightGBM 420 → ブレンド → 区分線形較正の図。閾値 0.5 に最適点を寄せる）

> Two branches. A small multilayer perceptron on the pooled encoder features
> combined with the inventory, five folds by three encoders by three seeds. And
> LightGBM on the tabular features alone, per station, per fold, per seed.
> We blend the two.
> One detail matters for the metric: F1 is binarised at a fixed zero point five,
> but the optimal threshold per station is not zero point five. We map each
> station's probability piecewise-linearly so its out-of-fold optimum lands on
> zero point five. The mapping is monotone, so AUROC is untouched.

## Slide 5 — Results and robustness (2:30–3:00, 70 words)
（F1/AUROC の推移表 + 「GT マスクを入れても劣化しない」検証結果）

> Out of fold, on five hundred eighteen frames, this reaches an F1 around zero
> point seven nine and an AUROC around zero point nine two, against zero point
> seven one and zero point eight nine for our earlier encoder-only baseline.
> We also checked that a model trained on predicted masks does not degrade when it
> is given better masks, so improving segmentation cannot hurt Task 3.
> Thank you.
