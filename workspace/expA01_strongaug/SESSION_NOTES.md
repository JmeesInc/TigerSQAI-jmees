# expA01_strongaug

expA00 からの変更点は **augmentation の増強のみ**（効果を単独で確認するため、モデル・loss・ハイパラは expA00 と同一）。

## 変更内容 (dataset.py)

expA00 の弱 aug → 以下に増強:
- Affine: scale ±10%→±20-25%, translate ±5%→±10%, rotate ±10°→±25°, p 0.5→0.8
- 追加: ElasticTransform / GridDistortion (OneOf, p=0.3)
- ColorJitter 強化 (0.3/0.3/0.3/0.15, p=0.7) + RandomGamma (p=0.3)
- 追加: GaussianBlur / MotionBlur (OneOf, p=0.3), GaussNoise (p=0.3)

## 動機

- expA00 fold0 で epoch 12 に early peak → 以降 fine が緩やかに劣化（過学習の兆候）
- train 40 case と少なく、hidden test は未知センター（center_5）→ 汎化が効く見込み

## Ablation プロトコル

- fold0 のみ学習（60ep, GPU0, ~1.1h）→ 公式コードで fold0 OOF 採点
- 比較対象: expA00 fold0 = **task1 Dice 0.5973 / HD 0.3358, task2 Dice 0.5731 / HD 0.3515**

## 結果

| fold | best val/score (自前監視) | expA00 同 fold | 差 |
|------|--------------------------|---------------|-----|
| 0 | 0.5971 | 0.5904 | +0.0067 |
| 1 | 0.5978 | 0.5905 | +0.0073 |
| 2 | 0.5937 | 0.5948 | −0.0011 |
| 3 | 0.5971 | 0.5886 | +0.0085 |
| 4 | 0.6221 | 0.5746 | +0.0475※ |
| **平均** | **0.6016** | 0.5878 | **+0.0138** |

※ expA00 fold4 は中断→再開の影響あり。それを除いても 4/5 fold で改善
- 終盤（ep40〜60）も伸び続け、expA00 の early peak → fine 劣化が緩和（過学習抑制が効いている）
- **公式 OOF 全量（524枚/42case, 2026-08-23 確定）**:
  - task1: **Dice 0.5792 / HD 0.3349**（expA00: 0.5759 / 0.3415 → +0.0033 / −0.0066）
  - task2: **Dice 0.6245 / HD 0.2913**（expA00: 0.6041 / 0.3148 → **+0.0204 / −0.0235**）
  - **4指標すべて改善 → 採用（新ベスト）**。特に coarse と HD に効く（過学習抑制で境界が安定した可能性）
- 注: fold3 は初回起動が2分で中断され `fold3_001` に退避 → 完走後 `fold3` へスワップ済み（旧 = `fold3_aborted`）
