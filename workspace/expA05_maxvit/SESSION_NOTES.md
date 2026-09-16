# expA05_maxvit

**expA01（強aug）レシピのまま encoder を `tu-maxvit_base_tf_512.in21k_ft_in1k` に変更**した実験。

## 構成

- encoder: MaxViT-Base (tf_512, ImageNet-21k→1k ft)。**5ステージ**（stride 2,4,8,16,32）なので Unet++ とそのまま噛み合う
- **`img_size=(576,1024)` を timm に渡して window/grid attention の分割サイズを再計算**
  （tf_512 既定 window=16 では 576×1024 の特徴マップが割り切れず assert で落ちるため）
- batch 4→**2** × accum 2→**4**（実効バッチ 8 は expA01 と同一）。smoke 実測 VRAM ~22GB
- decoder/loss/aug/その他ハイパラは expA01 と同一

## 動機

- CNN (EfficientNet) → hybrid attention (MaxViT) のアーキ比較。局所 window + 大域 grid attention は
  大臓器の一貫性（speckle 抑制）に効く可能性
- in21k 事前学習は公開（ルール OK）

## Ablation プロトコル

- 5fold 全学習 → OOF 全量公式評価。比較対象 = expA01 (T1 0.5792/0.3349, T2 0.6245/0.2913)

## 結果

| fold | best val/score | expA01 同 fold | 差 |
|------|---------------|---------------|-----|
| 0 | 0.6090 | 0.5971 | +0.012 |
| 1 | 0.6110 | 0.5978 | +0.013 |
| 2 | 0.5987 | 0.5937 | +0.005 |
| 3 | 0.6165 | 0.5971 | +0.019 |
| 4 | 0.6176 | 0.6221 | −0.005 |
| **平均** | **0.6106** | 0.6016 | **+0.009** |

- 4/5 fold で改善。coarse が一貫して強い（0.64 台）— window+grid attention の大域一貫性が効いている可能性
- GPU1 共有時 ~5-7min/epoch。公式 OOF 全量評価は実行中（確定値はそちら）
