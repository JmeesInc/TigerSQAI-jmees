# 提出履歴

> ✅ **2026-09-09: 下記の 2 件を修正した v005_ens5_t3noleak が完成**（本ファイル末尾）。**提出はこれを使う**。
>
> 🚨 **2026-09-08 判明: v001〜v004 はこのままでは提出不可**（公式 Docker Instructions 2026-08-11 版・最新評価コードで確認）
> 1. **task1/ と task2/ の中身が逆**: 公式は Task1=merged(15+bg) / Task2=fine(30+bg)。全バージョンが fine→task1/, merged→task2/ に出力しており、色テーブル不一致で大幅減点になる
> 2. **Task3 の station one-hot 入力は禁止**（「Do not use the station information to get a prediction」）: v002〜v004 の task3 ヘッドが該当 → 画像のみ入力で再学習が必要
> 3. tar.gz 命名は `tigersqai_<team>_task123.tar.gz` 形式厳守（`_v3` 等のサフィックスは避ける）。提出フォームで各タスクの担当イメージを申告
> → 修正版（フォルダ入替 + task3 再学習）を作成してから提出すること

## v001_expA06_task12 — 初回テスト提出（Task1+2、未アップロード）

- **日付**: 2026-08-27 作成（アップロードはユーザー手動 → Synapse evaluation queue）
- **モデル元**: `workspace/expA06_f2c_loss/results/expA06_f2c_loss/fold{0..4}/best.ckpt`
  （fp16・model-only に変換して `model/fold{N}.pt`、計 1.3GB）
- **構成**: MaxViT-Base tf_512 (in21k_ft_in1k, img_size 576×1024) + dual Unet++（fine 31 / coarse 16）
  + 強 augmentation + f2c 階層一貫性 loss（学習時のみ）
- **fold 定義**: `workspace/fold/v1/folds.csv`（StratifiedGroupKFold 5, group=case, stratify=center, seed42）
- **学習データ**: 第3バッチまでの 524 枚 / 40 case（1024×576 キャッシュ）
- **学習パラメータ**: AdamW 2e-4, wd 0.01, warmup3+cosine, 60ep, batch2×accum4, 16-mixed,
  loss = 0.5·Dice_fine + 0.5·Dice_coarse + 0.25·Dice_f2c（weight 3/2/1, 背景込み）
- **推論**: 5-fold 平均 softmax アンサンブル。1024×576 で forward → 平均確率を元解像度へ
  bilinear → argmax → labelmap の正確な RGB で PNG。task3 は出力しない（task12 提出）
- **CV（公式 5fold OOF 全量, 524枚/42case）**: **Task1 Dice 0.6663 / HD 0.2646、Task2 Dice 0.6530 / HD 0.2574**
- **LB**: 未提出
- **検証**: ローカル回帰テスト合格（--network none, RO /input, 4枚〔4K 含む〕で
  ファイル数・命名・解像度・RGB 色仕様準拠を assert。コンテナ CPU vs ホスト GPU の画素一致率 99.96%+）
  ⚠️ 本ホストは nvidia-container-toolkit 不在のためコンテナ内 GPU は未検証（コード同一性で担保）。
  実行時間見込み: RTXA5000 で 140 枚 ≈ 20 分（予算 280 分）
- **提出物**: `submit/v001_expA06_task12/tigersqai_shunsuke_task12.tar.gz`（docker save）
  - image: `tigersqai26_shunsuke:v1`。**TEAM 名は Synapse 登録名に要変更**（build.sh/export.sh の TEAM 環境変数 → 再ビルド）
- **アップロード手順**: Synapse evaluation queue へ tar.gz を提出し、TigerSQ-AI Challenge Admin に共有（wiki 639935）

## v002_expA06T01_task123 — Task1+2+3 フル提出（未アップロード）

- **日付**: 2026-09-06 作成（アップロードはユーザー手動）
- **Task1/2**: v001 と同一（expA06 5-fold 平均 softmax。CV T1 0.6663/0.2646, T2 0.6530/0.2574）
- **Task3**: expT01 凍結ヘッド ×5 + finetune 版 ×5 の **10 モデル平均**（station one-hot 入力）
  - CV（公式 evaluate_cls, OOF 518行）: **Weighted F1@0.5 0.7255 / AUROC 0.8979**
  - 凍結ヘッドは seg encoder の特徴を共有（追加コストほぼゼロ）、ft は encoder 5 本追加（fp16 1.2GB）
- **task3.csv**: 全入力フレーム分、列順 `case_id,6L,...,13L`、値 [0,1]
- **検証**: 本番同等コンテナテスト合格（--network none, RO /input, 4K 含む。task3.csv の列順・行対応・値域も assert）
- **提出物**: `submit/v002_expA06T01_task123/tigersqai_shunsuke_task123.tar.gz`（image: tigersqai26_shunsuke:v2）
  - **TEAM 名は Synapse 登録名に要変更**（TEAM 環境変数で再ビルド）
- **モデル元**: expA06 (workspace/expA06_f2c_loss) + expT01 (workspace/expT01_task3_head, results/ + results_ft/)

## v003_alldata_t3 — Task3 を全データ学習版に更新（推奨提出版・未アップロード）

- **日付**: 2026-09-06
- **v002 との差分**: Task3 の 10 モデルを **518 枚フル・固定 30ep・warmup3・最終 epoch 重み**で再学習
  （5-fold 学習版から置換。Task1/2 は expA06 のまま）
- **根拠（全データ学習の採用判断）**: 学習曲線の安定性検証 —
  ft の final-ep と best の差が全 fold ≤0.006、EMA 実験2本でも final≒best（差 ≤0.001）
  → ユーザー決定ルール「安定 → 全データ学習」を適用。EMA 自体は 0.999=underfit / 0.99=中立で不採用
- **参考 CV**（5-fold 学習版, 同構成）: F1 0.7255 / AUROC 0.8979。全データ版は
  held-out 評価不能だが、+20% のデータで同等以上を期待
- **検証**: コンテナ回帰テスト合格（v2 と同一手順）
- **提出物**: `submit/v003_alldata_t3/tigersqai_shunsuke_task123_v3.tar.gz`（image: tigersqai26_shunsuke:v3）

## v005_ens5_t3noleak — ✅ 仕様修正版（task1/2 スワップ + Task3 リーク除去）**推奨提出版・未アップロード**

- **日付**: 2026-09-09
- **v004 からの差分（v001〜v004 の 2 大問題を両方解消）**
  1. **出力先スワップ修正**: 公式定義どおり **merged/coarse → `/output/task1/`、fine → `/output/task2/`**
     - 公式 wiki 639935 の色表（Task1 16 ID / Task2 31 ID）と `labelmap.csv` の
       `merged_*` / `fine_*` が **完全一致**することをスクリプトで照合（不一致 0 件）
  2. **Task3 の station 入力を撤廃**: expT01（station one-hot 15 次元連結＝ルール違反）→
     **expT03（画像 GAP 768 のみ）** に差し替え
- **Task1 (merged) / Task2 (fine)**: expE01 **ens5** = expA06 + expA09 + expA10 + expA11 + expA05
  の 5 レシピ × 5 fold = **25 モデルの softmax 平均**（v004 と同一）
  - CV（公式 5fold OOF, 526枚/42case）: **Task1 Dice 0.6731 / HD 0.2351、Task2 Dice 0.6820 / HD 0.2383**
- **Task3**: `workspace/expT03_task3_noleak/results/fold{0..4}_head.pt` の **凍結ヘッド 5 モデル平均**
  - CV（公式 evaluate_cls, OOF 518行）: **Weighted F1@0.5 0.7070 / AUROC 0.8867**
  - expT01 凍結（違反構成）0.7143/0.8830 との差は F1 −0.007 / AUROC +0.004。**リーク除去のコストはほぼゼロ**
  - **ft 版は dl2 (A4000 16GB) で CUDA OOM のため未完** → v002〜v004 の「凍結+ft 10 モデル」構成から
    凍結 5 モデルのみに縮小している（ft が完走すれば上積みの余地あり）
- **fold 定義**: seg = `workspace/fold/v1/folds.csv`、Task3 = `workspace/fold/v2/folds.csv`
- **学習データ**: 第3バッチまでの 524〜528 枚 / 40 case（Task3 は GT のある 518 行）
- **推論**: 1024×576 で forward → 平均確率を元解像度へ bilinear → argmax → labelmap の正確な RGB で PNG。
  Task3 は seg encoder（expA06 fold 対応）の GAP 特徴に凍結ヘッドを乗せ、5 fold 平均の確率を出力
- **検証（すべて合格）**
  - コンテナ回帰テスト（`--network none`, `/input` RO, 4 枚〔4K 1 枚含む〕）:
    ファイル数・命名・解像度・**色表の取り違え検出**・task3.csv の列順/行対応/値域をすべて assert
  - **公式評価スクリプト `metrics/01_evaluate_challenge.py` がコンテナ出力に対しエラーなく完走**
    （4 枚は多くのメンバーの学習内フレームのためスコアは参考外: T1 0.8597/0.1140, T2 0.8049/0.1555,
    T3 0.7714/0.9722。**I/O 互換性の確認が目的**）
  - コンテナ(CPU) vs ホスト(GPU) の画素一致率 **99.99%+**、task3 確率 max|Δp| = 0.0011
  - 提出重み `model_t3/frozen_head_fold0.pt` + `model/fold0.pt` の推論が expT03 の OOF 確率を
    **max|Δp| = 0.0014** で再現（配線ミス・前処理不一致がないことの担保）
  - ⚠️ 本ホストは nvidia-container-runtime のバイナリ不在でコンテナ内 GPU 実行は不可 →
    テストは CPU 実行。GPU 実行時の挙動はホスト直実行（上記の 99.99% 一致）で担保
- **実行時間見込み**: ホスト GPU 直実行で **25 モデル × 3 タスク = 約 4.7 秒/フレーム**（+ モデルロード ~7 分）
  → 140 フレームで **約 20 分**（予算 420 分）。CPU 実行だと 3.7 分/フレームなので **GPU 必須**
- **提出物**: `submit/v005_ens5_t3noleak/tigersqai_shunsuke_task123.tar.gz`（docker save + pigz、**8.9GB**。イメージ実体は 13.1GB）
  - image: `tigersqai26_shunsuke:v5` / push 用タグ `docker.synapse.org/syn77311180/tigersqai26_shunsuke:v5`
  - ⚠️ **TEAM 名は Synapse 登録名に要変更**（`TEAM=<team> bash build.sh` で再ビルド → ファイル名も追従）
- **アップロード手順（ユーザー手動）**: tar.gz を Synapse へアップロード、または
  `docker login docker.synapse.org` → `docker push docker.synapse.org/syn77311180/tigersqai26_shunsuke:v5`。
  提出フォームで **task1/task2/task3 すべてをこのイメージが担当**すると申告し、method description
  （外部データ・事前学習重みの申告を含む）を添える。提出後は主催者にメール連絡が必要

## v006_a23 — ✅ 全データ 17 本アンサンブル + fine 特化 + α + Task3 マスク在庫融合（2026-09-13 20:10 ビルド・検証済み、未アップロード）

- **実験フォルダ**: `workspace/expA23_sweep`（seg）、`workspace/expT04_task3_sweep`（Task3）、`workspace/anatomy_graph`（AnatomyLoss）
- **Task1/2 モデル** (`model/`, 8.98 GB, fp16): 全 526 枚で学習した 17 本（fold v2 で検証したレシピ、最終 epoch）
  a_nohflip / a_toolpaste / d_deeplabv3p / e_convnext_xl_384 / h_upernet_swin_l / k_dicedet_rules (seed42, 43) / k_rules_boundary /
  l_boundary / l_dicedet / l_rules / rw1_detbd_rules / rw2_detbd_dlv3 / rw3_dlv3_f2c05 / rw4_rules_scse_strong / rw6_rules_scse_toolpaste_f2c05 /
  **ft_t2_fine**（fine 特化、`fine_only: true, weight: 4` → Task2 側にのみ、share ≈19%）
  推論: メンバーを 1 本ずつ GPU に載せ替え、softmax をモデル解像度で CPU 累積（fine 分母 20 / coarse 分母 16）→ 原寸へ bilinear → Task2 のみ α（`model/alpha.json`, cross-fit +0.0136）→ argmax → labelmap RGB
- **Task3** (`model_t3/` 223 MB + `model_t3enc/` 9.6 GB): 予測マスクの解剖在庫 315 次元 + fold encoder（l_dicedet / d_deeplabv3p / e_convnext_xl_384 × 5 fold）の GAP → MLP 45 本 + LGBM 420 本、w_lgb 0.4、クラス別閾値。OOF F1 0.7765 / AUROC 0.9145（v005: 0.7070 / 0.8867）
- **CV（fold v2, 原寸・公式規約）**: 7 recipe fold アンサンブル (候補 B) T1 coarse 0.6900 / T2 fine 0.7007（α で +0.0136、ft 追加で +0.004）。全データ 17 本は検証不可（データ量曲線からの外挿 +0.005 前後）
- **検証（すべて合格）**: コンテナ回帰テスト（CPU `runc`, `--network none`, `/input` RO, 4 枚〔4K 含む〕）ファイル数・命名・解像度・RGB 色・表の取り違え・task3.csv 列順/行対応/値域 OK。
  公式 `metrics/01_evaluate_challenge.py` がコンテナ出力で完走（4 枚は学習内フレームなので参考値: T1 0.8855/0.0781, T2 0.8282/0.1121）。
  ホスト GPU 直実行（同 process.py, 17+15 モデル）で fine_only 分母・Task3 配線を確認済み。
- **実行時間**: CPU で 4 枚 8.5 分（ロード込み）。GPU ホスト直実行で 2 枚 3.5 分（ロード込み、32 モデル）→ 140 枚で 15 分前後の見込み
- **提出物**: image `tigersqai26_shunsuke:v6`（26.6 GB）/ push 用 `docker.synapse.org/syn77311180/tigersqai26_shunsuke:v6`。tar.gz は `export_image.sh`（未実行）
- 保留: anat3d（純粋 3D 解剖知識グラフ版 AnatomyLoss）と rw5（detbd + AnatomyLoss + XL, fold0 0.7028）の全データ版を追加した v6b を検討中

## v007（= submit/v006_a23 の再ビルド） — ✅ 実行時間問題を修正した最終提出候補（2026-09-15 19:40 ビルド・検証済み）

- **背景**: 主催者が `tigersqai26_shunsuke:v6` を実行したところ **139 フレームに約 9 時間**かかった。
  公式予算は **1 分 / フレーム / タスク**（3 タスク・139 枚 = **417 分**）で、超過すると
  そのイメージが担当する **全タスクが無効**になる。
- **原因（実測）**
  1. **GPU が使われていなかった可能性が最も高い**。`process.py` は `torch.cuda.is_available()` が
     偽なら黙って CPU に落ちる。実測 1 メンバー 1 フレーム: **GPU 0.10〜0.14 s / CPU 4.01 s（27 倍）**。
     32 本 x 139 枚 x 4.0 s = 4.9 時間で、ロードと後処理を足すと 9 時間に整合する。
     自ホストは `nvidia-container-runtime` のバイナリが無く、コンテナ GPU 実行を一度も検証できていなかった。
  2. メンバー外側ループのため **1 枚の PNG を 32 回デコード**していた。
  3. 全フレームの確率を CPU に保持（576x1024 x 47ch fp32 = 110MB/枚、**139 枚で 15.3GB**）。
  4. 全処理後に一括書き出しのため、時間切れ kill で **出力ゼロ = 全タスク失格**。
- **修正 (process.py v007)**: フレーム外側チャンク(16枚)ループ / デコード 1 回 / GPU 上でチャンク分だけ累積 /
  チャンクごとに PNG と task3.csv を flush / 構造キャッシュ（75 本の構造は 7 通り）と重み RAM キャッシュ /
  **起動時に実機較正して時間予算からメンバー数 K を毎チャンク決定**（GPU 無しなら K=1 まで縮んで完走）。
  Task3 encoder 本数は AUROC のスケールを揃えるため全チャンク共通で固定。
- **等価性**: 旧実装と同一構成で **画素一致率 99.9965% / task3 max|Δp| 0.0013**。
- **Task1/2 モデル** (`model_sub/`, 11.7 GB fp16): 全 526 枚学習の **16 本**。
  5-fold CV（公式 Dice+HD の fold 平均）での構成比較:

  | メンバー数 | T1 Dice | T1 HD | T2 Dice | T2 HD |
  |---|---|---|---|---|
  | 1 | 0.6943 | 0.2096 | 0.7185 | 0.1988 |
  | 4 | 0.7062 | 0.2045 | **0.7310** | **0.1911** |
  | 6 | 0.7083 | **0.2012** | 0.7264 | 0.1967 |
  | 8 | **0.7109** | 0.2019 | 0.7293 | 0.1946 |
  | 10 | 0.7098 | 0.2040 | 0.7321 | 0.1933 |
  | 14 | 0.7097 | 0.2024 | 0.7247 | 0.2004 |

  弱いレシピを足すと T2 が下がるため、外部データ事前学習 dlv3 系 + seed/解像度違い + T1 に効いた
  アーキ多様性 2 本 + fine 特化 `ft_t2_fine` (fine_only, weight 4) の 16 本構成。
  順序は `model_sub/priority.json`（CV 順）。予算制御で実際に使う本数は実行時に決まる。
- **Task3** (`model_t3/` 223MB + `model_t3enc/` 9.6GB): 変更なし（在庫 315 次元 + fold encoder GAP、MLP 45 + LGBM 420）。
- **実行時間実測**: Quadro RTX 8000 1 枚、139 枚（1080p 101 / 4K 29 / 720p 9）、seg 16 本 + encoder 10 本で
  **約 32 分**（うち 12 分が初回の重み cold read）。予算 417 分。
- **検証**: コンテナ回帰テスト（`--network none`, `/input` RO, 4K 含む 6 枚, CPU `runc`）
  ファイル数・命名・解像度・RGB 色・表の取り違え・task3.csv 列順/行対応/値域すべて OK。
  公式 `metrics/01_evaluate_challenge.py` が完走（6 枚は学習内なので参考値: T1 0.8305/0.1248, T2 0.8036/0.1521）。
  コンテナ内で `device=cpu` と明示ログが出ることも確認（主催者側ログでここを確認してもらう）。
- **提出物**: image `tigersqai26_shunsuke:v7` (28.7 GB) / push 用 `docker.synapse.org/syn77311180/tigersqai26_shunsuke:v7`
  / `submit/v006_a23/tigersqai_shunsuke_task123.tar.gz`
- **write-up**: `submit/writeup/writeup_draft.md` を v2 に改訂（75 本 → 16 本、Appendix A 再生成、
  推論・実行時間の節を書き直し）。主催者返信ドラフトは `submit/writeup/reply_to_organizers_runtime.md`。

### v007 追記（2026-09-15 21:25 再ビルド）— Task3 の学習/推論 不整合を修正

19:40 にビルドした v7 の `model_t3` は 9/12 書き出しのままで、以下 2 点の不整合があった。
**この 2 点は提出済み v6 にも入っている。**

1. 学習側の在庫特徴が fine/coarse 取り違え（`f_*` が coarse マスク由来）。コンテナ側は正しい。
2. LightGBM の `num_leaves` が OOF(常に7) と書き出し(7/15) で不一致 → 未検証設定の出荷。

修正して `model_t3` を作り直し、同時に**解剖文脈 + 粗グリッド**まで入れた 1741 次元に更新。

| 構成 | F1@0.5 | AUROC |
|---|---|---|
| v6 同梱（取り違えあり 315 次元） | 0.7765 | 0.9145 |
| **v7 採用（修正 1741 次元, w_lgb 0.6）** | **0.7890** | **0.9165** |

検証: コンテナ側 `t3_features.py` と学習側 3 スクリプトが 1741 特徴で最大差 **0.0**。
書き出したモデルを fold0 重みだけで回した確率が OOF 生確率と **max|Δp| 0.000000** で一致。

再ビルド後の最終値: seg 16 本 + Task3 encoder 15 本、イメージ **28.8 GB**、
139 枚 GPU 実測 **22.7 分**（cold start 33 分、予算 417 分）、コンテナ回帰テスト合格。
提出物 `submit/v006_a23/tigersqai_shunsuke_task123.tar.gz` (22.8 GB)。

## v008 — CV 貪欲選択どおりの構成に差し替え（2026-09-16 ビルド）

- v7 の 16 本は CV 評価済み 6 本 + 手選び 9 本だったため、**公式 Dice+HD の 5-fold 平均で貪欲前向き選択**をやり直した
  （候補 13 レシピ、全員重み 1）。採用 7 本: `q_endovis18_dlv3, r_xl, r_ft_fine, q_both_dlv3, s_kdr_seed43, l_dicedet, r_detbd`
  → **T1 Dice 0.7114 / HD 0.1989、T2 Dice 0.7338 / HD 0.1897**（fold モデル, α・island removal なし）。
  v7 で CV 評価できた最良構成 (10 本) は T1 0.7098/0.2040, T2 0.7321/0.1933 だったので 4 指標すべて改善。
- 全データ版が無かった `r_xl, r_ft_fine, r_detbd` 等は当日学習して書き出した。`ft_t2_fine` は CV で悪化するため外した。
- Task3 (`model_t3` 1741 次元) と推論コード (`process.py` v007) は v7 と同一。
- 詳細: `daily_reports/20260916.md`

## v010 — 最終提出候補（2026-09-16 12:58 ビルド）: CV 貪欲 + 局所探索の 7 枠 x シード違い 25 本

- **枠（レシピ, 全員重み 1）**: `q_endovis18_dlv3, r_xl, r_ft_fine, q_both_dlv3, s_kdr_seed43, l_dicedet, r_nohflip`
  - 貪欲前向き選択（公式 Dice+HD の 5-fold 平均, 候補 13）で 7 本 → leave-one-out + 1 本入替の局所探索で
    `r_detbd → r_nohflip`（公式 CPU 実装で T1 0.7133/0.2009, T2 0.7350/0.1888）→ 入替後からの再探索で改善なし
- **シード違い**: 同じレシピの全データ版シード違いは **枠内で 1/k 平均**（`process.py` の group 正規化）。
  `q_endovis18_dlv3` x17, `s_kdr_seed43` x3, 他 x1 → 25 チェックポイント。予算で削られても枠の重みは 1 のまま。
- Task3 (`model_t3`, 1741 次元) と推論コードは v7 と同じ系統。較正時に encoder もウォームアップし、Task3 encoder 15 本を全部使う。
- v8（7 本のみ）は `r_detbd` 版で push 前に v10 に差し替え。v9（25 本, r_detbd 版）は局所探索前の中間版。
- イメージ 36.2 GB。詳細は `daily_reports/20260916.md` §6–8。

## v011 — v10 + 水平反転 TTA（5 枠）（2026-09-16 14:40 ビルド）

- v10 と同じ 7 枠 25 本。`r_xl`（TTA で悪化）と `r_nohflip`（反転なし学習）以外の 5 枠に反転 TTA。
- CV（公式 CPU 実装）: T1 0.7134/0.2021、T2 0.7371/0.1876（v10: 0.7133/0.2009、0.7350/0.1888）。
- 推論時間: 139 枚 GPU 実測 42.5 分（予算 417 分）。詳細は `daily_reports/20260916.md` §9。

## v12（2026-09-16 16:00 ビルド、dl2）— v11 + Task3 差し替え
- **イメージ**: `docker.synapse.org/syn77311180/tigersqai26_shunsuke:v12`（30.2 GB、dl2 でビルド、push 待ち）
- **Task1/2**: v11 と同一（seg 25 本 7 枠 + group 正規化 + 5 枠 TTA、`model_v11/`）
- **Task3**（`workspace/expT04_task3_sweep/export_t3_v12.py` → `model_t3_v12/`）:
  - GBDT 枝: XGBoost vector-leaf（multi_output_tree, depth 6, n=1000, lr 0.02, colsample 0.5, subsample 0.8, λ=1）× 3 seed、入力 1741 次元（在庫 315 + 解剖文脈 202 + 粗グリッド 1224、候補 E の OOF 予測マスク由来）
  - MLP 枝: d_deeplabv3p fold encoder GAP 1536 + 在庫 315 → hidden 256 × 3 seed（`model_t3enc_v12/`、5 fold 3.8 GB）
  - 合成 w_gbdt = 0.7、閾値較正 λ=1.0（全 OOF から station ごとに 1 組）、推論は 5 fold 平均
  - **CV（v2 5fold OOF 518 行、公式 evaluate_cls、nested 較正）: F1@0.5 0.8056 / AUROC 0.9253**（v7〜v11 の構成 0.7890 / 0.9165、5/5 fold 改善）
- 検証: 書き出し ↔ 探索 OOF max|Δp| 0、GAP 経路 cos 1.00000、dl2 回帰テスト合格（6 枚 1.7 分、seg 25/25、t3enc 5/5）
- process.py 変更点: `load_t3` に XGBoost 読み込み、`task3_row` を枝別特徴列（`features` / `features_mlp`）と `w_gbdt` 対応。LGBM 経路は後方互換で残置

## v013 — v12 + 全データ seed 違いを 6 枠に追加（2026-09-16 18:42 ビルド・回帰テスト合格）

- seg: **31 checkpoint / 7 枠**（枠の重みは全て 1、枠内で 1/k 平均）
  `q_endovis18_dlv3` 17 / `s_kdr_seed43` 4 / `r_xl`・`q_both_dlv3`・`l_dicedet`・`r_nohflip`・`r_ft_fine` 各 2
- 追加学習した全データ seed 違い 5 本: `r_xl_s43, q_both_dlv3_s43, l_dicedet_s43, r_nohflip_s43, k_dicedet_rules_s44`（16:16〜18:13）
- **`r_ft_fine` はエポック不整合を修正**: fold 検証の best_epoch が 5/5 fold とも 0 だったため、
  8 epoch 版 1 本 → **1 epoch 版 2 seed** に差し替え（`full_r_ft_fine_ep1`, `full_r_ft_fine_s43_ep1`）
- Task3・`process.py` は v12 と同一。新規 seed は priority 末尾（予算超過時に最初に削られる）
- イメージ 34.9 GB。dl2 4090 で `--network none` 回帰テスト合格（`device=cuda`, 31/31 本, 6 枚 1.8 分）
- push 用: `docker push docker.synapse.org/syn77311180/tigersqai26_shunsuke:v13`

## v014 — v13 + 後処理修正（2026-09-16 20:28 ビルド、dl2）: クラス別島除去・α なし

- seg 31 本 / 7 枠、Task3、`process.py` は v13 と同一。差分は後処理のみ:
  `alpha.json` を `.dockerignore` で除外（α 不使用）、`model/island_ppm.json`（クラス別しきい値 = GT 平均面積の 2%）を追加、`ENV ISLAND_PPM=500`（json 欠落時の保険）
- 根拠（最終 7 レシピ TTA の fold OOF、公式 Dice/HD）: v13 の α + 0.4% は T1 0.7188/0.2242, T2 0.7383/0.2101、v14 のクラス別 2% は T1 0.7193/0.2053, T2 0.7453/0.1848
- イメージ検査: `/app/model` に alpha.json なし・island_ppm.json あり（fine 30 / coarse 15 クラス）、31 checkpoint、ISLAND_PPM=500。34.9 GB
- push: 20:29 開始（`push_v14.log`）。Synapse の提出差し替えはユーザー手動
- **結果**: push 完了は 21:01 JST（締切 20:59 の約 2 分後、digest sha256:bd4ad0c4…）。締切時点の有効提出は **v13**。v14 を採点対象にできるかは主催者への確認次第
