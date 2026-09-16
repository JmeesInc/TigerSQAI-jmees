# expA08_dino_distill

**expA06 + DINOv3-7B 特徴空間の蒸留**（ユーザー提案 2026-09-06）。
「B03 で分かった DINOv3 の表現の良さを、推論コストとトレードオフなしで取り込む」試み。

## 動機（B03 の per-class 分析から）

- expB03（推論時に 7B 特徴を融合）は **Fatty_Tissue_Esoph +0.136 / Instrument +0.022 と
  テクスチャ系を改善する一方、Lymph_Node −0.042 / R_Vagal −0.028 / Aorta −0.034 と w=3/2 を犠牲**
  → 加重指標では純損（T1 −0.005）
- 仮説: **表現の知識だけを蒸留で移せば、推論アーキ（= expA06）の w=3 性能を壊さずに
  テクスチャ系の改善だけを得られる**のではないか

## 構成

- ベース: expA06 と完全同一（MaxViT-Base tf_512 + dual Unet++ + 強aug + f2c loss）
- 追加（**学習時のみ**）:
  - 教師 = 凍結 DINOv3-7B fp16 の stride-16 特徴 (4096ch, 36×64)
  - 生徒 = MaxViT encoder の stride-16 特徴 (384ch, 36×64) → 1×1 conv で 4096ch へ射影
  - **チャネル方向 cosine 類似度損失**: `L_distill = 1 - cos(proj(student), teacher)`
  - `loss = 0.5·L_fine + 0.5·L_coarse + 0.25·L_f2c + 0.25·L_distill`（config `distill_weight`）
- **teacher / proj は checkpoint から除外** → 保存 ckpt は **expA06 とキー完全一致（検証済み）**
  = 推論コード・提出コンテナ（submit v002/v003 の model_def.py, process.py）をそのまま流用可能
- 学習コストのみ増（7B forward: ~2.5min/epoch → fold0 ≈ 2.5h）。**推論は expA06 と同一**

## プロトコル

- fold0 → expA06 fold0 (0.6575) と比較。上回れば 5-fold（vast / 空き GPU 活用）
- 空間解像度が 576/16=36, 1024/16=64 で教師・生徒とも一致するため、リサイズ不要

## 結果

| fold | expA08 (蒸留) | expA06 (参照) | 差 |
|------|--------------|--------------|-----|
| 0 | **0.6591** | 0.6575 | **+0.0016（わずかに勝ち → 5-fold 展開）** |
| 1-4 | 学習中 (GPU3) | | |

- 学習曲線: ep10 で 0.5711（A06 +0.031）と立ち上がりが速いが、ep20-30 で A06 に追いつかれ、
  最終的に僅差勝ち。**蒸留は収束を早めるが最終到達点への寄与は小さい**という傾向
- 差が誤差レベル（+0.0016）なので、5-fold OOF の公式評価（Dice + HD）で最終判断する
