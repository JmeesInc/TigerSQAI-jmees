# expG01_mask2image — マスク条件付き術野画像生成（ControlNet）

## 目的
1. **本命**: expS01 の合成ラベルマップ（Blender/アトラス由来）に実データ由来の見た目を与え、
   擬似リアルな術野画像 + 正確な GT を得る
2. **副次**: 実マスクを条件に「同じレイアウト・別の施設スタイル」を生成し、
   未知センター（center_5 相当）への見た目汎化を狙う aug 素材にする

## 位置づけ
- ユーザー判断: 合成の統計を実データに完全一致させるのは困難かつ不要。
  この段階で一旦画像を作り、実物を見て判断する
- 器具(1)・Other(2) は**レンダリングせず**、expA02 の cutout 資産で貼り付ける方針
- 内視鏡の円形視野（黒縁）も**生成せず**、学習時 augmentation で付与する方針

## 構成
```
実 528 ペア（data/images + workspace/data_proc/labels_fine_1024）
  → fine ラベルを公式パレット（data/labelmap.csv）で着色 = 条件画像
  → SD1.5 (runwayml) + ControlNet (lllyasviel/control_v11p_sd15_seg で初期化)
     UNet/VAE/TextEncoder 凍結、**ControlNet のみ学習**
  → 合成ラベルマップに適用
```
- プロンプトに `center_N` をスタイルトークンとして埋め込み → 生成時に施設スタイルを指定可能
- `--holdout-center` で特定施設を除外した汎化チェックが可能
- 学習: 768×448, batch4×accum2, AdamW 1e-5, fp16 AMP, 6000 step

## 実行
```bash
# 学習（GPU1）
CUDA_VISIBLE_DEVICES=1 .venv/bin/python workspace/expG01_mask2image/train_controlnet.py \
  --output workspace/expG01_mask2image/results/cn_v1 --steps 6000 --batch 4 --accum 2

# 生成（実ラベルから = 再構成の忠実度チェック）
.venv/bin/python workspace/expG01_mask2image/generate.py \
  --controlnet workspace/expG01_mask2image/results/cn_v1/controlnet \
  --labels workspace/data_proc/labels_fine_1024 \
  --output workspace/expG01_mask2image/results/cn_v1/gen_real --n 12

# 生成（合成ラベルから = 本命）
.venv/bin/python workspace/expG01_mask2image/generate.py \
  --controlnet workspace/expG01_mask2image/results/cn_v1/controlnet \
  --labels workspace/expS01_atlas_synth/outputs/r4_cond_1024/label \
  --output workspace/expG01_mask2image/results/cn_v1/gen_synth --n 12 --center center_1
```

## 環境
torch 2.9.1+cu128 / diffusers 0.40.0 / transformers 5.15.1 / peft 0.20.0（追加導入）
Quadro RTX 8000 ×4。学習は GPU1 で 1.03 秒/step（6000 step ≈ 1h45m）、VRAM 18GB

## 判定基準（画像を見て決める）
1. 実ラベル → 生成 が実画像に似ているか（再構成の忠実度）
2. **合成ラベル → 生成 が術野として破綻していないか（本命）**
   - 破綻するなら「合成レイアウトが実データ分布から外れすぎ」= expS01 の較正に戻る判断材料
3. center トークンでスタイルが変わるか（未知施設 aug の実現性）
4. 生成画像を既存 seg モデルに通し、条件マスクとの Dice で忠実度フィルタ（未実装）

---

## v1 結果（2026-09-09）: 見た目は成立、解剖忠実度は不足

### 忠実度フィルタ（expA06 fold0 で生成画像を推論 → 条件マスクとの weighted Dice）

| 条件 | 平均 Dice | 中央値 | 採択率(≥0.35) |
|---|---:|---:|---:|
| **実画像そのもの（ベースライン）** | **0.4912** | 0.5094 | 18/20 |
| 実ラベル条件の生成画像 | 0.2673 | 0.2681 | 4/20 |
| 合成ラベル条件の生成画像 | **0.1364** | 0.1205 | **0/20** |

**ベースラインを取ったのが重要**。実画像でも 0.49（fold0 単体・TTA なし）なので、
生成は「ベースラインの 54%」、合成条件は「28%」という位置づけ。

### クラス別（合成条件）— 大きいクラスだけ従い、小構造は完全に無視
Pleura 0.685 / Esophagus 0.355 / Azygos 0.223 / Trachea 0.221 / Pericardium 0.187 /
**L main bronchus 0.000 / Instrument 0.000 / Fatty tissue esophagus 0.007**

### スイープ
- **施設スタイルトークンは無効**。center_1/3/6/7 でほぼ同一画像 → v2 で削除
- 条件強度は上げるほど良い（0.6→0.084, 0.8→0.105, 1.0→0.113）が 1.0 で頭打ち

### 診断
**ControlNet だけ学習し UNet を凍結したことが原因**。SD1.5 の UNet は手術画像を知らないため、
ControlNet が空間条件を渡しても解剖として解釈できず、UNet の事前分布
（それらしい生体組織テクスチャ）が支配する。

## v2（実行中）: UNet LoRA 追加 + 施設条件削除
- `train_v2.py`: UNet に LoRA (rank 32, 6.4M params, lr 1e-4) を追加。ControlNet は lr 1e-5 のまま
- プロンプトは固定キャプション 1 種（施設トークンは v1 で無効と判明したため削除）
- step 12000（v1 の倍）、768×448、batch4×accum2
- 判定: 合成条件の weighted Dice が **0.1364 から有意に上がるか**。
  実画像ベースライン 0.4912 の半分（≈0.25）に届けば学習データとして検討の余地

---

## v2 結果（2026-09-10）: UNet LoRA は実条件で大幅改善、合成条件は横ばい

### 途中で見つけたバグ（重要）
`unet.save_lora_adapter()` が保存するキーには `unet.` 接頭辞が無く、
**`pipe.load_lora_weights()` が黙って何も読み込まずに成功を返す**（画素差分 0.0000 で確認）。
警告 `No LoRA keys associated to UNet2DConditionModel found with the prefix='unet'` は出るが
例外にならない。→ `{f'unet.{k}': v}` に詰め替えて渡し、`get_active_adapters()` で assert するよう修正。
**LoRA 未適用のまま「v2 は悪化」と誤判定しかけた**（0.0677 という数値が出ていた）。

### 最終スコア（expA06 fold0 で生成画像を推論 → 条件マスクとの weighted Dice, n=20）

| 構成 | 実ラベル条件 | 合成ラベル条件 |
|---|---:|---:|
| 実画像そのもの（ベースライン） | **0.4912** | — |
| v1（ControlNet のみ, 6000 step） | 0.2673 | 0.1364 |
| **v2（+UNet LoRA r32, 12000 step）** | **0.3510** | **0.1570** |
| ベースライン比 | 54% → **71%** | — |

- **実条件は +0.084 の大幅改善**、採択率 4/20 → **11/20**
- **合成条件は +0.021 でほぼ横ばい**（0.1364 → 0.1570）
- クラス別（実条件, v1→v2）: Fatty tissue +0.190 / R main bronchus +0.127 /
  L main bronchus +0.126 / Azygos +0.122 / Trachea +0.105 / Esophagus +0.105 / Pleura +0.078
  → **UNet LoRA で手術ドメインを教えると条件追従性が上がる**という仮説は実証された

### 最重要の知見: 実条件 0.351 vs 合成条件 0.157 のギャップ
生成器は**実データらしいマスクなら条件に従える**が、**合成マスクでは従えない**。
v1 では両方低くて切り分け不能だったが、v2 で実条件だけが改善したことで
**合成ラベルマップが実データ分布から外れていることが定量的に確認された**。
→ expS01 の較正（Pleura 過剰・上縦隔バイアス・IPV 面積不足）は「詰めても意味がない」のではなく、
**実際に下流性能を制限している**。較正を続ける根拠になる。

### 判定
- 合成条件 0.157 は**学習データとしては使えない水準**（マスクと画像が不一致）
- 実条件 0.351（ベースラインの 71%）は、**実マスクを条件にした見た目 augmentation** としてなら
  検討の余地がある。ただし締切 9/15 まで 5 日で、A/B 検証込みでは間に合わない
