# vast_ops — TigerSQAI の学習を Vast.ai に farm out する

MVAA の `workspace/vast_ops/` の方式を移植。**学習だけを box で行い、OOF 推論・公式評価は
重みを pull してローカルで行う**（元解像度 data/ 1.5GB と reference/ を上げないため）。

## ワークフロー

```bash
cd workspace/vast_ops
bash package_data.sh          # -> tiger_train_data.tar.zst (~450MB, 1024x576キャッシュ+fold+labelmap)
bash package_code.sh          # -> tiger_code.tgz (workspace の py/sh/yaml)

bash search_and_create.sh search 40 0.8      # VRAM>=40GB, $0.8/h 以下の候補
bash search_and_create.sh create <offer_id> 90
vastai show instances                         # ssh host/port を確認

bash provision.sh <port> <host> expA06_f2c_loss 2   # upload + deps + tmux 学習開始
bash collect_one.sh <instance_id> <port> <host> expA06_f2c_loss 2  # 監視->pull->destroy (要バックグラウンド実行)
```

pull 後は通常どおりローカルで `./run.sh oof` (predict_oof) → 公式評価。

## 不変条件（MVAA での事故から）

- **pull 成功が destroy の前提**。checkpoint は box にしか無い。destroy を忘れると課金が続く
- **ssh は短い呼び出しのみ**。長時間 ssh セッションは切断時に box の tmux まで巻き添えにした事故あり
- 学習は必ず box 上の **detached tmux** 内（bootstrap.sh が行う）

## GPU 要件

| 実験系 | VRAM | disk | 備考 |
|--------|------|------|------|
| A系 (EffNet/MaxViT ± f2c) | 24GB (推奨40GB) | 40GB | batch2 実測 ~22GB |
| B系 (DINOv3-7B 凍結/融合) | 40GB+ | 90GB | DINOv3 26GB を HF から自動DL (ungated)。bf16 対応 GPU なら高速化余地 |

## 状態管理

`experiments.yaml` に box/fold の実行状態を記録（machine-readable）。スコアの正本は
従来どおり `claudeSummary.md`。

## 注意

- **アカウント残高**: `vastai show user` で確認。2026-08-26 時点で残高 $0 → **入金が必要**
- wandb: provision.sh が ~/.netrc から API key を取り box に渡す（run 名/group は従来どおり）
- checkpoint は best/latest とも凍結 ViT 除外済み (~0.5-1.4GB) なので pull は軽い
