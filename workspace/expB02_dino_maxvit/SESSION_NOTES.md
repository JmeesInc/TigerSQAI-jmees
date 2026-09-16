# expB02_dino_maxvit

**DINOv3-7B (凍結) + MaxViT (学習) の融合 + dual Unet++**。expA05 (現ベスト) に DINOv3 特徴を注入する形。

## 構成

- MaxViT-Base tf_512 (in21k_ft_in1k, img_size=(576,1024)) が通常の5段ピラミッド（expA05 と同一・学習対象）
- 凍結 DINOv3-7B (fp16) の stride-16 特徴 (4096ch) を射影して **MaxViT の stride-16 (384ch) / stride-32 (768ch) ステージへ加算注入**
  - 射影後 BatchNorm を **γ=0 初期化** → 初期状態は expA05 と等価（注入ゼロ）から学習で開く。事前学習表現を壊さない
- checkpoint から凍結 ViT を除外（expB01 と同じ仕組み）
- batch 2 × accum 4（実効8）、他は expA01/A05 と同一

## Ablation プロトコル

- **fold0 のみ → expA05 fold0 (監視 val 0.6090) と比較**。γ=0 初期化なので「expA05 相当から DINOv3 注入でどれだけ上乗せできるか」を直接測る
- ~15-20min/epoch 見込み（7B forward + MaxViT 学習）→ fold0 ≈ 15-20時間

## 結果

| fold | best val/score | expA05 同 fold | 差 |
|------|---------------|---------------|-----|
| 0 | **0.6320** | 0.6090 | **+0.023 → ゲート強通過（全実験 fold0 最高）** |

- ep20 で早くも 0.6286（expA05 の最終 best 超え）。γ=0 注入 + 7B 特徴の上乗せが明確
- expB01 (凍結のみ, 0.6168) をも +0.015 上回る → 「MaxViT の学習可能ピラミッド + DINOv3 意味特徴」の組み合わせが最良
- fold1〜4 は expA06 fold0 (GPU1) の後に自動開始（2026-08-25 仕込み）
