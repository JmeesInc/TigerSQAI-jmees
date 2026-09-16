# expA22_anatomy_rules — expA06 + 解剖ルール loss (隣接・排他)

2026-09-11 開始。expA06_f2c_loss (maxvit_base, dual Unet++, f2c loss) のレシピをそのまま使い、
`anatomy_rules.py` の `AnatomyRuleLoss` を **epoch 20 から 10 epoch かけて線形に投入** する。

## ルール loss

- 行列は `workspace/anatomy_graph/rules.py` が **fold 別 (val 除外) の GT 統計** から生成 (`train.py` の `ensure_rules` が無ければ自動生成)。
- `adj` = 4 近傍の画素対で `Σ W[a,b] P_a(x) P_b(x+d)` を soft 境界質量で正規化 (= 境界のうち禁止ペアの割合, [0,1])。正規化項は detach。
- `excl` = soft presence `s_c = max avgpool16(P_c)` で `Σ_{a<b} E[a,b] s_a s_b`。
- 重み: adj 1.0 / excl 0.1、coarse ヘッドにも同じ (coarse_scale 1.0)。fine 83 ペア (完全禁止 25) + 排他 24、coarse 18 + 2 (fold0)。
- 覆い被さる組織 (脂肪・胸膜・血液・切除面・器具・Other) はルールから除外 (ユーザー指示 2026-09-11)。
- val では argmax 予測の硬い違反率 (`val/viol_adj_fine` など) も記録 → A06 と比較できる。

## 実行

```
CUDA_VISIBLE_DEVICES=0 ./run.sh 0      # fold0 (2026-09-11 15:49 開始, GPU0)
./run.sh oof                           # 5fold 後に OOF + 公式評価
```

smoke (`./run.sh smoke`, 2 epoch × 5%) は 2026-09-11 15:47 に通過 (rules 読み込み・loss 計算・val 違反率ログを確認)。

## 期待値と判断基準

- ens5 OOF の違反率は fine adj 0.0152 (GT 0.0020) と小さく、後処理での Dice 改善はゼロだった (`workspace/anatomy_graph/README.md`)。
  → 本実験の勝ち筋は「学習中の正則化で気管の誤検出 (気管–心膜/下肺静脈) が減るか」と HD。
- fold0 の A06 = val/score 0.6575 (監視 val)。これを下回れば不採用。

## 結果

### fold0（2026-09-11 15:49 → 19:14, 60 epoch, GPU0）

| | expA22 | expA06 | 差 |
|---|---|---|---|
| val best score | **0.6637** (ep54) | 0.6575 (ep36) | **+0.0062** |
| 終盤 10 epoch 平均 | **0.6610** | 0.6489 | **+0.0121** |
| val dice_fine (best 時) | 0.6690 | 0.6589 | +0.0101 |
| val dice_coarse (best 時) | 0.6583 | 0.6561 | +0.0022 |

**ルール違反率の推移（val, argmax 予測）**: ramp 開始前 ep15 で 0.040 → ep19 で 0.0117 → ep25 で 0.0017 →
ep30 以降 **0.0004 前後で安定**。GT 自身の違反率 0.0020 より低い。排他違反は 24 ペア → **0**。

epoch 24 以降はほぼ全 epoch で A06 を上回る。**ルール loss は Dice を犠牲にせず違反を消している**。

### 判断
- **採用**。5fold へ（fold1 GPU0 / fold2 GPU1 / fold3 GPU3 を 22:31 投入、fold4 は fold1 の後に自動実行）
- 次: `./run.sh oof` で公式 OOF（Dice + **HD**）。遠方の誤検出が減るので HD の改善が本命
- ens5 に追加した ens6 の評価まで行う
