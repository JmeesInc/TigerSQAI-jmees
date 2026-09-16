# expT03_task3_noleak — Task3 を station 入力なしで再学習

> 学習自体は別スレッド（dl2 / A4000 16GB）で実施。本ノートは v005 提出時に
> 成果物を回収・検証した側（2026-09-09）がまとめたもの。

## 背景

公式 Docker Instructions（wiki 639935, 2026-08-11 版）に
**"For Task 3: Do not use the station information to get a prediction."** と明記されている。

expT01 のヘッドは入力に **ファイル名由来の station one-hot(15)** を連結していた。
学習データ 518 行では「自 station は必ず可視」が 518/518 で成立するため、これは
**ルール違反であると同時に強力なリーク**でもある。→ 画像特徴のみで再学習したのが本実験。

## 構成

- 入力: 1024×576 キャッシュ画像のみ（`workspace/data_proc/images_1024`）
- backbone: **expA06 の fold 対応 `best.ckpt` の encoder のみをロードして凍結**
  （別マシン用に `enc_weights/fold{N}_encoder.pt` の軽量 encoder-only 重みも読める）
- head: encoder 最終特徴 GAP **(768)** → MLP(256) → Dropout0.3 → 14 sigmoid、BCE
  - **expT01 との差分は「one-hot(15) を入力から外した」ことだけ**
- fold: `workspace/fold/v2/folds.csv`（case グループ）、30ep、AdamW lr 1e-3、warmup3 + cosine

## 結果（公式 `metrics/evaluate_cls`, 5fold OOF 518行）

| 構成 | Weighted F1@0.5 | AUROC |
|------|----------------:|------:|
| **expT03 凍結ヘッド ×5（本実験・ルール準拠）** | **0.7070** | **0.8867** |
| expT01 凍結ヘッド ×5（station one-hot＝違反） | 0.7143 | 0.8830 |
| expT01 凍結+ft 10 モデル平均（違反） | 0.7255 | 0.8979 |

- **リーク除去のコストはほぼゼロ**: 凍結ヘッド同士で F1 −0.007 / AUROC **+0.004**。
  「自 station は必ず可視」という情報は、画像特徴だけでほぼ代替できていた
- **ft（encoder も更新）版は dl2 A4000 16GB で CUDA OOM のため未完**（`results_ft/` は空、`train_ft.log` 参照）。
  expT01 では ft を足すと F1 +0.011 だったので、**48GB GPU で回せば上積みの余地がある**

## 成果物と利用先

- `results/fold{0..4}_head.pt`（head 部分のみ、fp32 804KB）、`results/oof_task3.csv`
- **v005 提出コンテナで採用**: `submit/v005_ens5_t3noleak/model_t3/frozen_head_fold{N}.pt`（fp16 変換）
  - 変換は `submit/v005_ens5_t3noleak/export_t3.py`
  - 提出重み + `model/fold0.pt`（expA06 fold0）での推論が OOF 確率を **max|Δp| = 0.0014** で再現することを確認済み
  - 原寸 PNG → `cv2.resize(INTER_AREA)` が学習時の 1024×576 キャッシュと一致することも同時に確認できた

## 注意

- ヘッドは **expA06 の fold 対応 encoder 特徴**に対して学習されている。
  推論時は必ず同じ fold の encoder と 1:1 で組むこと（他レシピの encoder には乗らない）
- 学習ラベルとして station を使うこと自体は禁止されていない（禁止は **推論時の入力としての使用**）。
  現構成は推論時に station を一切参照しない
