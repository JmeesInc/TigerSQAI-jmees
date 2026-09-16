# expA10_toolmask

**器具を専用モデルに委譲し、解剖モデルは解剖だけを学ぶ**（ユーザー TODO「tool masked dice」）。

## 前提検証（先に実施。これが成立しないと意味がない）

STIR `convnext-unet-best.pth` と expA06 の器具 Dice を、公式と同じ case 階層集約で比較（526 枚）:

| 対象 | STIR tool model | expA06 | 差 |
|------|----------------|--------|-----|
| **Instrument (fine_id=1)** | **0.8927** | 0.8132 | **+0.0795** |
| Non-anatomical (id 1 or 2) | 0.8133 | 0.8535 | −0.0402 |

→ **器具単体では tool model が明確に強い**ので委譲は成立。ただし **Other (id=2) を含めると逆転**する
（tool model は「器具」しか知らない）ため、**上書きは Instrument のみ**、Other は解剖モデルに残す。

## 構成

1. **マスク事前生成**: `workspace/data_proc/tool_masks_1024/`（528 枚, thr 0.5）。器具画素率 平均 5.5% / 最大 22.5%
2. **学習**: `MaskedDiceLoss`（自前 soft dice）で **器具画素を loss から完全除外**。
   fine / coarse / f2c の 3 損失すべてに同じ keep マスクを適用。
   マスクは albumentations の `masks` に渡すので**幾何 aug 後も画像・ラベルと整合**する
   （keep が全 1 のとき MONAI DiceLoss と一致する実装）
3. **推論**: 解剖モデルの argmax に対し、tool model のマスク領域を Instrument (fine 1 / coarse 12) で上書き

## expA03 との違い

expA03 は「学習はそのまま、推論時だけ上書き」で +0.001 に留まった。
本実験は**学習時に器具画素を除外して解剖クラスの学習を純化する**のが本命で、上書きはその帰結。

## プロトコル

- 5-fold（fold v2）→ OOF 公式評価。比較 expA06: T1 0.6663/0.2646, T2 0.6530/0.2574
- 注目点: 全体スコアに加え **Instrument の per-class Dice が 0.8132 → 0.89 付近に上がるか**、
  および器具周辺の解剖クラス（Fatty_Tissue, Pleura 等）が改善するか

## 結果

（学習後に記入）
