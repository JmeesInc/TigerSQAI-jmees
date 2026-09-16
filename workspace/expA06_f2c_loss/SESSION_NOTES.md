# expA06_f2c_loss

**expA05（MaxViT + 強aug）+ fine→coarse 階層一貫性 loss** のみ追加（ユーザー指示 2026-08-25）。

## 変更内容

- fine decoder の logits を softmax 後、`labelmap.csv` の fine_id→merged_id で **merged 単位に確率合算**
  （einsum with 31×16 の 0/1 行列、buffer 登録）→ coarse GT に対する weighted Dice loss を第3項として追加
- `loss = 0.5*L_fine + 0.5*L_coarse + 0.25*L_f2c`（`loss.fine2coarse_weight: 0.25`, config で調整可）
- L_f2c は MONAI DiceLoss(softmax=False = 確率入力, include_background=True, coarse の 3/2/1 重み)

## 動機

- fine2coarse ablation で「fine 予測の写像 (0.6181/0.3033) < 専用 decoder (0.6342/0.2780)」
  → fine decoder は coarse 粒度で見ると劣る = fine の断片化誤り。coarse 粒度の直接監督で
  fine decoder の大域一貫性を改善し、fine 側スコア（現状の弱点）の向上を狙う

## Ablation プロトコル

- fold0 のみ → expA05 fold0 (監視 val 0.6090) と比較。効けば B 系の勝者にマージ

## 結果

| fold | best val/score | expA05 同 fold | 差 |
|------|---------------|---------------|-----|
| 0 | 0.6575 | 0.6090 | +0.049 |
| 1 | 0.6474 | 0.6110 | +0.036 |
| 2 | 0.6435 | 0.5987 | +0.045 |
| 3 | 0.6737 | 0.6165 | +0.057 |
| 4 | 0.6591 | 0.6176 | +0.042 |
| **平均** | **0.6562** | 0.6106 | **+0.046（5/5 fold 全勝）** |

- fine が 0.57 台→0.65-0.67 台に大幅改善（f2c ablation で特定した「fine の断片化」への直接対策が的中）
- 推論コスト増ゼロ（loss のみ）で全実験最大の改善幅。7B DINOv3 融合 (expB02 fold0 0.6320) をも上回る
- OOF 全量公式評価 実行中 → 確定値はそちら。**次: expB02 (DINOv3+MaxViT) + f2c loss の合成が本命**
