# Fold 設計

## v1 (2026-08-22, 現行)

- **切り方**: StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
  - **group = case_id**（正規表現 `center_\d+_case_\d+` で抽出。例外2枚の `_frame_N` 形式にも対応）
  - **stratify = center**（center_1 が 16/40 case と偏っているため）
- **理由**:
  - 公式評価は 画像→case→全体 の階層平均。同一 case が train/val に跨るとリークし、CV が case 平均と乖離する → case でグループ化必須
  - hidden test には train に無い center_5 が含まれる（汎化評価）。center 層化で各 fold の center 構成を揃え、fold 間分散を抑える
- **データ**: 524 枚 / 40 case / 6 センター（第3バッチまで、Task1/2 は完全）

### fold 分布

| fold | imgs | cases | center 内訳 (cases) |
|------|------|-------|---------------------|
| 0 | 105 | 8 | c1:2, c2:2, c3:1, c6:1, c7:2 |
| 1 | 121 | 9 | c1:4, c2:1, c4:2, c6:1, c7:1 |
| 2 |  91 | 7 | c1:3, c2:2, c3:2 |
| 3 | 101 | 8 | c1:4, c2:2, c3:1, c7:1 |
| 4 | 106 | 8 | c1:3, c2:1, c3:1, c4:1, c6:1, c7:1 |

### 使い方

- config の `cv.folds_csv: workspace/fold/v1/folds.csv` で参照
- 列: `filename, case_id, center, station, fold`

### 更新ポリシー

- Task 3 ラベル最終版（8月末予定）が来たら、Task3 用に v2 を検討（seg 用は v1 のまま有効）
- 旧バージョンは削除しない
