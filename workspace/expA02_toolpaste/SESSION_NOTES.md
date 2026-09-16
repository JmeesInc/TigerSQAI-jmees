# expA02_toolpaste

expA00 からの変更点は **器具貼り付け augmentation の追加のみ**（base aug は expA00 の弱 aug のまま。効果を単独で確認するため）。

## 変更内容

- `toolpaste.py`: 自前 ToolPaste 実装の Tiger 適応版
  - cutouts: 同リポジトリの `tool_instances/instances.csv`（296 個, RGBA）。**出典 = SAR-RARP50 + SurgToolLoc（いずれも公開データセット。write-up で開示・引用必須）**
  - v10.41 との違い: ラベルをゼロにせず **Instrument クラスを書き込む**（fine=1 / coarse=12。Tiger は器具自体が採点対象）
  - 配置: コンテンツ bbox（レターボックス黒帯を輝度で検出して除外）の辺のアンカー → 解剖組織上の狙い点へ base→tip を写す相似変換。tip が必ず解剖上に乗る。長さは bbox 対角の 0.15〜0.85 に制限
  - Dataset.__getitem__ 内で albumentations の**前**に適用（幾何 aug は貼り付け後に一貫して掛かる）
  - DataLoader worker fork の RNG 複製対策: worker 毎に numpy global（pl.seed_everything が worker 別シード済み）から再シード
- config: `toolpaste: {enabled: true, p: 0.5, max_tools: 2}`
- プレビュー検証済み（レターボックス回避・挿入角度・サイズ感 OK）

## 動機

- 器具まわりの誤分類と、器具による解剖のオクルージョンへの頑健性向上
- Instrument クラス自体の強化（w=1 だが面積が大きい）

## Ablation プロトコル

- fold0 のみ学習（60ep）→ 公式コードで fold0 OOF 採点
- 比較対象: expA00 fold0 = **task1 Dice 0.5973 / HD 0.3358, task2 Dice 0.5731 / HD 0.3515**

## 結果

| fold | best val/score (自前監視) | expA00 同 fold | 差 |
|------|--------------------------|---------------|-----|
| 0 | 0.5906 | 0.5904 | +0.0002 |
| 1 | 0.5832 | 0.5905 | −0.0073 |
| 2 | 0.6027 | 0.5948 | +0.0079 |
| 3 | 0.6060 | 0.5886 | +0.0174 |
| 4 | 0.6116 | 0.5746 | +0.0370※ |
| **平均** | **0.5988** | 0.5878 | **+0.0110** |

※ expA00 fold4 は中断影響あり。参考: expA01 平均は 0.6016
- **公式 OOF 全量（524枚/42case, 2026-08-23 確定）**:
  - task1: **Dice 0.5715 / HD 0.3406**（expA00: 0.5759 / 0.3415 → **Dice −0.0044** / HD −0.0009）
  - task2: **Dice 0.6236 / HD 0.2851**（expA00: 0.6041 / 0.3148 → +0.0195 / **−0.0297**）
- **判断: 単独では不採用**（fine が悪化）。ただし **coarse への効果は大きく、HD 0.2851 は expA01 (0.2913) をも上回る全実験ベスト** → expA01 (強aug) と ToolPaste の併用 (expA04 候補) を試す価値あり。fine 悪化の仮説: 貼り付け器具が微細構造 (w=3 の神経・血管) を隠して学習機会を奪う → p を下げる / 小構造上への貼り付け回避も選択肢
