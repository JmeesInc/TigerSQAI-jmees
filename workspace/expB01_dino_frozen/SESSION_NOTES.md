# expB01_dino_frozen

**DINOv3-7B (凍結) + ViTDet 風 pyramid + dual Unet++**。expA01 強aug レシピベース。

## 構成

- encoder: timm `vit_7b_patch16_dinov3.lvd1689m`（6.72B params, **完全凍結・fp16・eval 固定**）
  - RoPE 位置符号化 → 576×1024 可変解像度 OK（grid 36×64 = 2304 tokens, dim 4096）
  - 学習・推論とも毎 step forward（強 aug で入力が変わるため特徴キャッシュ不可）
- ネック (学習対象): stride-16 特徴 → ViTDet 方式で p4 (convT×2) / p8 (convT) / p16 (1×1) / p32 (s2 conv)、各 256ch。stride-2 skip は学習可能 conv stem (3→32ch)
- 学習対象 = ネック + Unet++ decoder ×2 + head のみ
- **checkpoint から凍結 ViT を除外**（on_save_checkpoint で `.encoder.vit.` を削除、`strict_loading=False`。ckpt 491MB）。predict_oof は timm cache から ViT を再ロードし、非 ViT キーの欠落ゼロを assert
- batch 4 × accum 2、他ハイパラは expA01 と同一

## Ablation プロトコル

- **fold0 のみ → expA05 fold0 (監視 val 0.6090) と比較**して 5fold 化を判断（expA04 と同じゲート方式）
- ~7-8min/epoch 見込み（7B forward が支配的）→ fold0 ≈ 7時間

## 結果

| fold | best val/score | expA05 同 fold | 差 |
|------|---------------|---------------|-----|
| 0 | 0.6168 | 0.6090 | +0.008 |
| 1 | 0.6183 | 0.6110 | +0.007 |
| 2 | 0.6148 | 0.5987 | +0.016 |
| 3 | 0.6178 | 0.6165 | +0.001 |
| 4 | 0.6232 | 0.6176 | +0.006 |
| **平均** | **0.6182** | 0.6106 | +0.008 |

- expA05 には安定して勝つが、**expA06 (0.6562) には −0.038 と大差** → 単体採用はなし。OOF 全量評価は実行中（アンサンブル素材・記録として完遂）
- 傾向: coarse 寄り（fine 0.58-0.62 / coarse 0.61-0.66）。凍結特徴+軽ネックでは fine の細粒度分離が不足
