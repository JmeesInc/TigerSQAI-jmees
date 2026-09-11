# expG03_organwise — 先行研究3本を統合したマスク条件付き生成

## 動機
expG01 v2 の失敗（合成条件での忠実度 0.157、小構造が Dice 0）を受けて先行研究を調査した結果、
**主催者（NCT/TSO Dresden）自身が同じ問題を解いていた**ことが判明。3 本を統合する。

| 出典 | 要点 | 我々への適用 |
|---|---|---|
| [NCT anatomy-aware](https://arxiv.org/abs/2410.07753) WACV25 (Apache-2.0, `gitlab.com/nct_tso_public/muli-class-image-synthesis`) | 臓器別 inpainting → 合成。実画像併用で **seg +15%** | 臓器別・段階生成 |
| [SimuScope](https://github.com/SanoScience/SimuScope) WACV25 | シミュレータ → SD+LoRA+ControlNet(SoftEdge/Depth) で sim-to-real | **Depth 条件** |
| [CASDM](https://arxiv.org/abs/2410.23962) HTL25 | class-aware MSE で小さく重要なクラスを優先 | **クラス重み付き loss** |

補足: [SAADi](https://arxiv.org/abs/2509.18796) は「リアルに見えるが下流性能を害する」問題を明示。我々の症状と一致。
**汎用の「手術特化 SD ベースモデル」は存在しない**（公開されているのはタスク別 LoRA のみ）→ SD1.5 + 自前 LoRA は方向として妥当。

## 統合設計（1+2 を同一パイプラインに）
両手法の弱点が相殺される:
- 臓器別生成は**小構造が無視されない**が、独立生成のため光源・遠近が不整合 → **Depth 条件が解決**
- Depth 条件は 3D 整合を与えるが単発生成では小クラスが埋もれる → **臓器別生成が解決**

我々は先行研究より有利な材料を持つ: **Blender の深度真値**（推定でなく metric、30〜141mm）と**厳密な z 順序**。

### 学習 (`train_v3.py`)
- 条件を **6ch**（seg 着色 3ch + 深度 3ch）に拡張し、**ControlNet 1 本**で受ける（2 本並列だと競合する）
- **v2 の学習済み ControlNet から conv_in を 3→6ch に拡張し、深度側を zero-init**
  → 初期挙動は v2 と完全同一、そこから深度の使い方を学習する
- UNet LoRA も v2 から継承（`unexpected 0` で確認）
- **class-aware loss**: 公式 weight 3/2/1 を潜在解像度に落として MSE に乗算（0.51〜1.52 に正規化）

### 深度の調達 (`extract_depth.py`)
- 実データ: **Depth-Anything-V2-Small** で推定（実画像に深度は無い）、528 枚
- 合成: Blender の EXR 真値、2000 枚
- 両者を**逆深度で per-image 正規化**して分布を揃えた:
  実 p5/p50/p95 = 9/103/232 / 合成 = 9/77/226。
  横方向勾配は 実 1.24 / 合成 1.88（**合成のほうが鋭い** → 必要なら平滑化で調整）

### 推論 (`generate_organwise.py`)
- 深度バッファから**奥→手前の順序**を決定（明るい=近い）
- その順に **masked latent blending（RePaint 方式）で段階生成**。
  塗り終えた領域は以降のステップで固定する
- **inpainting 専用チェックポイントに移らないので、学習した LoRA をそのまま使える**
- 最終段は全画面パスで残りを仕上げる

## 実行
```bash
# 深度条件の作成
.venv/bin/python workspace/expG03_organwise/extract_depth.py --mode real \
  --src data/images --dst workspace/expG03_organwise/depth_real
.venv/bin/python workspace/expG03_organwise/extract_depth.py --mode synthetic \
  --src workspace/expS01_atlas_synth/outputs/pretrain_2k/depth --dst workspace/expG03_organwise/depth_synth

# 学習（v2 から継承）
CUDA_VISIBLE_DEVICES=3 .venv/bin/python workspace/expG03_organwise/train_v3.py \
  --output workspace/expG03_organwise/results/cn_v3 --steps 12000 --batch 4 --accum 2 \
  --class-aware --init-controlnet workspace/expG01_mask2image/results/cn_v2/controlnet \
  --init-unet-lora workspace/expG01_mask2image/results/cn_v2/unet_lora

# 臓器別・奥→手前生成
.venv/bin/python workspace/expG03_organwise/generate_organwise.py \
  --controlnet workspace/expG03_organwise/results/cn_v3/controlnet \
  --unet-lora workspace/expG03_organwise/results/cn_v3/unet_lora \
  --labels workspace/expS01_atlas_synth/outputs/pretrain_2k/label \
  --depth workspace/expG03_organwise/depth_synth \
  --output workspace/expG03_organwise/results/cn_v3/gen_organwise --n 20
```

## 判定
`workspace/expG01_mask2image/fidelity_filter.py` で合成条件の weighted Dice を測る。
- v1 (ControlNet のみ): 0.1364
- v2 (+UNet LoRA): 0.1570
- **v3 (+depth +class-aware +臓器別): 目標は 0.25 以上**（実画像ベースライン 0.4912 の半分）

## 状態 (2026-09-11)
- 深度条件: 実 528 / 合成 2000 とも作成済み
- v3 学習: GPU3 で走行中（1.56 秒/step、12000 step ≈ 5.2h）
- 臓器別生成スクリプト: 実装済み、v3 チェックポイント待ちで未検証
