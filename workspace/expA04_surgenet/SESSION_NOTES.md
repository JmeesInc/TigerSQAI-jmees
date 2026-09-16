# expA04_surgenet

**expA01（強aug）レシピのまま encoder を SurgeNet-Public (CAFormer-S18) に変更**した実験。

## 構成

- encoder: timm `caformer_s18` を **SurgeNet 変種（StarReLU→素の ReLU）に置換**して
  SurgeNet-Public の DINO 事前学習重みをロード（`reference/SurgeNet/weights/SurgeNetPublic_teacher.pth`, 150 tensors, missing=0 を assert）
  - キー変換: 公式 sail-sg 命名 → timm `checkpoint_filter_fn` → features_only 用 `stages.N.`→`stages_N.`
- CAFormer は4ステージ（stride 4,8,16,32）で smp は stride-2 に 0-ch ダミーを入れる → Unet++ が壊れる
  → **stride-2 の学習可能 conv stem（3→32ch）を追加**して実 skip に置換。encoder_channels=[3,32,64,128,320,512]
- decoder/loss/aug/ハイパラは expA01 と同一（強aug, weighted Dice, AdamW 2e-4, 60ep, batch4×accum2）

## ルール関連（重要・経緯）

- 当初は「非公開データ事前学習は禁止」の字義どおり **SurgeNet-Public**（公開データのみ ≈200万frames）で開始
- **2026-08-23 ユーザー判断で SurgeNetXL に切替**: 「公開されている重み自体は OK（organizer が DINOv3 = LVD-1689M 非公開データ事前学習を OK した前例あり）」の解釈。
  SurgeNetXL は RAMIE-UMCU（**食道切除 = 本コンペと同ドメイン**, 非公開）+ RARP-AvL + YouTube 470万 frames を含む最良 variant
  - Public 重みで ~10分学習した fold0 は `fold0_public_discarded` に退避して XL でやり直し
  - **TODO: organizer へ正式に問い合わせて確認を取る**（write-up 前必須）。NG なら Public 版に差し替え（config 1行 + 再学習）
- write-up では SurgeNet 論文 (Jaspers+, MedIA 2025) と重み URL を引用・開示

## Ablation プロトコル

- 5fold 全学習 → OOF 全量公式評価。比較対象 = expA01 (T1 0.5792/0.3349, T2 0.6245/0.2913)
- 仮説: 手術ドメイン事前学習で少データ (40case) への転移が改善。smoke でも epoch0 fine 0.56 と立ち上がり速い

## 結果（fold0 のみで打ち切り）

| fold | best val/score | expA01 同 fold | 差 |
|------|---------------|---------------|-----|
| 0 | **0.5415** | 0.5971 | **−0.056** |

- 終盤 (ep55-59) は 0.535 前後で収束済み（伸びしろなし）。立ち上がりは速かったが天井が低い
- **判断: 不採用・fold1 以降は中止**（2026-08-24）。CAFormer-S18 (26M) と EfficientNet-B7 encoder (64M) の
  容量差が手術ドメイン事前学習の利点を上回った。SurgeNet が提供する最大アーキが S18 なのが制約
- 復活条件があるとすれば: 蒸留・アンサンブル要員、または SurgeNet 重みを init に使った
  より大きいアーキ（ただし公式提供なし）。優先度低
