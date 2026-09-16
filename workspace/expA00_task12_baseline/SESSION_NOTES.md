# expA00_task12_baseline

Task1 (fine 31c) + Task2 (coarse 16c) の同時学習ベースライン。

## 構成

- **モデル**: shared encoder + dual decoder
  - encoder: `tu-tf_efficientnet_b7.ns_jft_in1k`（smp `get_encoder`、timm pretrained）
    - ⚠️ NS 重みは JFT-300M（非公開データ）由来。「非公開データで事前学習した重みは禁止」ルールに抵触の恐れあり → ユーザー判断で採用。config 1行で `ap_in1k` 等へ差し替え可能（要再学習）
  - decoder: `UnetPlusPlusDecoder` ×2（fine 31ch / coarse 16ch の `SegmentationHead`）
  - 計 70.2M params
- **入力**: 576×1024（16:9 厳密維持、÷32）。学習は `workspace/data_proc/*_1024` キャッシュから
- **loss**: MONAI DiceLoss(softmax, to_onehot_y, include_background=True, weight=クラス重み) を task 別に構築し `0.5*L_fine + 0.5*L_coarse`。クラス重みは公式評価と同じ 3/2/1（背景=1、coarse は構成 fine の max）= `data_proc/class_weights.json`。config で dicece / dicefocal に切替可
- **ハイパラ**: AdamW lr 2e-4 / wd 0.01 / warmup 3ep + cosine / 60 epochs / batch 4 × accum 2 / 16-mixed
- **augmentation**（弱め開始の原則）: HFlip(0.5), Affine(scale±10%, shift±5%, rot±10°, p=0.5), ColorJitter(0.2/0.2/0.2/0.1, p=0.5)

## Fold 設計

- `workspace/fold/v1/folds.csv` = StratifiedGroupKFold(5, group=**case**, stratify=**center**), seed=42
- 理由: 公式評価が 画像→case→全体 の階層平均のため case リーク厳禁。center_1 が 16/40 case と偏るため center 層化。分布は `workspace/fold/README.md` 参照
- 全 5 fold 学習する

## 監視メトリクス

- 毎 epoch: GPU 上で per-image weighted Dice（公式規約: 両方空=1.0 / 片方空=0.0）→ case 平均 → 全体平均。`val/score` = (fine+coarse)/2 を monitor に best.ckpt 保存
- HD は毎 epoch 計算しない（重い）。最終評価は `predict_oof.py` で **公式 `metrics.evaluate_seg.evaluate`** を使用（元解像度の RGB PNG に復元して採点。提出物と同一形式）

## wandb

- project `tigersqai`。**run 名は全 fold 共通 = experiment.name、fold 分けは `group=str(fold)`**（ユーザー指定の運用）
- config.yaml 全体を wandb config に記録。smoke 時は無効

## 実行

```bash
CUDA_VISIBLE_DEVICES=0 ./run.sh smoke   # 疎通確認 (_smoke に分離)
CUDA_VISIBLE_DEVICES=0 ./run.sh 0       # fold0 学習 (last.ckpt あれば自動再開)
./run.sh oof                            # 全 fold OOF 推論 + 公式評価
```

## 環境メモ（ハマりどころ）

- **プロジェクト venv 必須**: `.venv`（--system-site-packages）。user site の `transformers 4.57.2` が `huggingface_hub 1.7.1` と非互換で、lightning → torchmetrics → transformers の import 連鎖が死ぬ。venv 内で `transformers>=4.58`（5.15.1）を上書きして解決。run.sh が自動で activate する
- **公式 HD は総当たり実装で 4K では計算不能**（6枚が40分でも終わらない）→ `fast_hd.py` の EDT ベース厳密等価実装で `_binary_normalized_hausdorff` のみモンキーパッチ（適用前に公式実装と <1e-6 一致を自動検証）。6枚×2タスク=47秒。predict_oof.py が自動適用する

- **Lightning 2.x の `save_last` は「best 保存時のコピー」でしかない** → 毎 epoch レジューム用に `monitor=None` の rolling `latest.ckpt` を別 ModelCheckpoint で保存。レジュームは latest.ckpt 優先（train.py `pick_resume`）

## 実験ログ

| 日付 | 内容 | 結果 |
|------|------|------|
| 2026-08-22 | smoke (2ep, 5%) | 疎通 OK |
| 2026-08-22 | fold0 完走 (60ep) | **best = epoch 12, val/score 0.5904**（fine 0.57 台 / coarse 0.59 台）。以降 0.53–0.59 で停滞、cosine 終盤でも更新なし。early peak → LR 2e-4 が高すぎる可能性 or 過学習。fold1〜4 の曲線と合わせて判断 |
| 2026-08-22 | OOF→公式評価の配線検証 (smoke ckpt, 6枚) | 通し OK。fast HD パッチで 47 秒 |
| 2026-08-22 | fold1 完走 (60ep) | **best = epoch 30, val/score 0.5905**。fold0 (0.5904) と同水準。中盤 peak → fine 低下の形は fold0 と同様 |
| 2026-08-22 | fold2 完走 (60ep) | **best = val/score 0.5948**（終盤まで単調に改善。fold ごとに曲線の形は違うが best は 0.59 前後で揃う） |
| 2026-08-22 | fold3 完走 (60ep) | **best = val/score 0.5886** |
| 2026-08-22 | fold4 完走 (60ep, 中断→latest.ckpt 再開あり) | **best = val/score 0.5746**（最弱 fold） |

- 全 fold best: 0.5904 / 0.5905 / 0.5948 / 0.5886 / 0.5746（平均 0.588）

## OOF スコア（公式コード評価）

| task | Dice | HD | 範囲 | 備考 |
|------|------|----|------|------|
| **task1 (fine)** | **0.5759** | **0.3415** | **5fold OOF 全量 (524枚/42case)** | **確定 CV** |
| **task2 (coarse)** | **0.6041** | **0.3148** | **5fold OOF 全量 (524枚/42case)** | **確定 CV** |
| task1 (fine) | 0.5973 | 0.3358 | fold0 のみ (8 case) | fold0 は coarse 弱・fine 強の fold |
| task2 (coarse) | 0.5731 | 0.3515 | fold0 のみ (8 case) | 〃 |

- 42 case = 実 40 case + 例外ファイル名 2 枚が公式 case パース（rsplit）で擬似 case 化したもの（公式仕様どおり）

- 自前監視 val/score 0.5904 vs 公式 fine Dice 0.5973 → 監視メトリクスは公式とよく整合
- 全 fold OOF は fold1〜4 完走後に実施

## エラー分析（fold0, 24 枚目視 2026-08-22）

1. **断片化・speckle が最大の問題**: 大きな臓器領域の内部に小さな誤クラス島が多発。散在 FP は HD を直撃（max-min 距離）→ **クラス別小連結成分除去の後処理で HD 大幅改善の余地**
2. **テクスチャ類似クラスの混同**: Esophagus↔Fatty_Tissue(_Esophagus)、Pericardium↔Inferior_Pulmonary_Vein、Pleura/Lung（支配的緑クラス）への過剰割当
3. **Lymph_Node (w=3) の散在 FP** が多い。weight 3 なのでスコア影響大
4. Instrument / Aorta / Azygos など高コントラスト構造は良好。letterbox 黒枠・UI オーバーレイは正しく Background 扱い
