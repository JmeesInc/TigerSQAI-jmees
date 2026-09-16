# 3-min video #1 — Tasks 1 & 2 (segmentation)

**Team Jmees.** 目標 3:00。読み上げ原稿は英語、1 スライド 1 メッセージ。
括弧内は画面に出すもの。図は `submit/writeup/figures/` を使う。

---

## Slide 1 — Title (0:00–0:12, 30 words)
（タイトル、チーム名、3 タスク担当、コンテナ 1 本）

> Anatomy-aware ensemble segmentation for the Tiger SQ-AI challenge. Team Jmees.
> One container covers all three tasks, sharing a single encoder between the
> merged and the fine-grained head.

## Slide 2 — The problem the metric creates (0:12–0:45, 80 words)
（公式 Dice の「片方だけ存在＝0点」規約の図解。小さな偽陽性 1 個でクラス全体が 0 点になる例）

> The official score is a weighted Dice plus a normalised Hausdorff distance,
> aggregated per image, then per case. The part that drives our design is the
> absent-class convention: if a class is missing from the ground truth but our
> mask contains even a few pixels of it, that class scores zero for the frame.
> Ten small, high-weight structures carry weight three. So a handful of stray
> pixels costs more than a slightly loose boundary.

## Slide 3 — Architecture (0:45–1:15, 75 words)
（dual-head 図: 共有 encoder → fine decoder / coarse decoder。ConvNeXt-L, DeepLabV3+ / Unet++）

> Every member is a dual-head network: one ConvNeXt-Large encoder, two decoders,
> one for the fifteen merged classes and one for the thirty fine classes. Sharing
> the encoder means the container pays for one forward pass per member and gets
> both tasks. Decoders are DeepLabV3-plus or Unet-plus-plus, at one-zero-two-four
> by five-seven-six.

## Slide 4 — What actually moved the score (1:15–2:05, 120 words)
（棒グラフ 3 本: 外部データ事前学習 +0.015 / island removal +0.030 / α +0.014）

> Three things moved the score, and the ranking surprised us.
> First, pre-training the encoder *and decoder* on public laparoscopic
> segmentation data, CholecSeg8k and EndoVis 2018, is worth about plus one and a
> half points of Dice, on every fold. More pre-training data was not better: about
> one thousand frames beat three and a half thousand.
> Second, island removal. Connected components smaller than zero point four
> percent of the frame are relabelled with the majority class around them. That
> is plus three points on the fine task, and it improves every centre in a
> leave-one-centre-out check. It is a direct answer to the absent-class rule.
> Third, a per-class probability scaling fitted out of fold, worth another
> one point four points.

## Slide 5 — Ensemble selection (2:05–2:35, 70 words)
（表: メンバー数 1/4/8/14 と Task1/Task2 の Dice・HD。6本目と14本目で T2 が下がる点を強調）

> We picked the ensemble on five-fold cross-validation, scored with the official
> Dice and Hausdorff distance and averaged over folds. More members is not better.
> Task 2 gets worse as soon as we add recipes from outside the surgically
> pre-trained group. The submitted ensemble is sixteen full-data models: the
> pre-trained family with seed and input-size replicas, two architecture-diverse
> members, and one fine-only specialist.

## Slide 6 — The container (2:35–3:00, 60 words)
（タイムライン図: 139 枚 33 分 / 予算 417 分。予算制御のループ図）

> The container processes frames in chunks, decodes each frame once, accumulates
> on the GPU, and writes results after every chunk. It measures its own speed at
> start-up and sizes the ensemble to the time remaining, so it cannot exceed the
> budget. One hundred thirty-nine frames take thirty-three minutes on one GPU
> against a budget of four hundred seventeen. Thank you.
