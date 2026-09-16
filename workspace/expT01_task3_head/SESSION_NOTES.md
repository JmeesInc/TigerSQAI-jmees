# expT01_task3_head

Task3 特化モデル（凍結版）: expA06 の fold 対応 encoder（凍結）+ GAP + station one-hot(15) + MLP→14 sigmoid。
BCE 30ep, batch8, lr1e-3 cosine。fold v2。

## 結果（公式 evaluate_cls, OOF 518行, 2026-09-06）

- **Weighted F1@0.5 = 0.7143 / AUROC = 0.8830**（Task3 初 CV）
- fold 別 val mAUROC: 0.862 / 0.884 / 0.917 / 0.897 / 0.873（平均 0.887）
- 学習コスト: 全 5fold で ~25 分（encoder 凍結）

## 検証マトリクスでの位置

- vs expT01-ft（encoder finetune, dl2 A4000 で実行中）→ 特化の上限
- vs expT02（MTL, ローカル GPU1）→ 「特化 vs multi-task」の本題
- 前提知見: station one-hot は 518/518 で自ステーション可視のため必須特徴

## 追加結果（2026-09-06 確定）

| 構成 | Weighted F1@0.5 | AUROC |
|------|----------------|-------|
| T01 凍結 | 0.7143 | 0.8830 |
| T01 finetune (dl2 A4000) | 0.7097 | 0.8897 |
| **T01 凍結+ft 平均アンサンブル** | **0.7255** | **0.8979** ← **Task3 採用構成** |

- finetune 単体は凍結と実質同等（518 枚では encoder 更新の利得なし）だが、
  **アンサンブルは両指標で単体超え** → 提出は 2 系統 × 5fold = 10 モデル平均
- 特化 vs MTL: expT02 (MTL) は seg/cls 両方で劣化 → 特化の勝ち（expT02 SESSION_NOTES 参照）
