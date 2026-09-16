# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## このリポジトリについて

**コンペ**: [Tiger SQ-AI-Challenge](https://www.synapse.org/Synapse:syn74209386/wiki/639462)
**プラットフォーム**: Synapse Sage Bionetworks（胸腔鏡下食道切除術の手術フレーム）

## プロジェクト固有情報（このコンペの具体）

> このセクションがコンペ固有の事実。以降の汎用テンプレと矛盾したらここを優先する。

### データ

- `data/` は symlink → `<challenge data root>`（読み取り専用扱い。中身を書き換えない）
- 詳細仕様は **`data/README.md`** と **`survey/competition/overview.md`** が正。要点のみ以下に再掲する
- 全体は **700 フレーム/50ケース/7センター**（train 40 / test 10、1センターを train から除外）。**第3バッチ（2026/8中旬）で Task1/2 の訓練データは完全**: ローカルに **524 枚 / 40 ケース / 6 センター**（center_1〜4, 6, 7。**center_5 欠番 = held-out センターとみられる**）
- **Task 3 のラベルは未完**: `lymph_node_station_visibility.csv` は **308 行のみ**（最終更新 8月末予定。Task1/2 より少ないまま確定の可能性あり）→ Task 3 はラベルのある行だけで学習する設計に
- 解像度は **1920×1080 ×379 / 3840×2160 ×107 / 1280×720 ×38** の混在（**4K あり**。リサイズ方針と提出時の原寸復元に注意）
- ファイル名 = `{case_id}_{station}.png` 例: `center_1_case_10_6L.png`。ただし **例外 2 枚**（`center_2_case_8_{12L,6L}_frame_{n}.png`）→ case 抽出は正規表現 `center_\d+_case_\d+` で。**case 単位でグループ構造あり** → fold は **GroupKFold（case でグループ化）**。同一 case が train/val に跨るとリーク
- マスクは `masks_fine/`（31クラス）と `masks_coarse/`（16クラス）。画像と同名で各 524 枚

### タスク（3つ）⚠️ 2026-09-08 に公式が番号を確定（従来の当リポジトリ表記と逆）

1. **Task1 = Coarse（merged）セグメンテーション** — 16 クラス（`masks_coarse/`）→ 提出は `/output/task1/`
2. **Task2 = Fine セグメンテーション** — 31 クラス（`masks_fine/`）→ 提出は `/output/task2/`
3. **Task3 = リンパ節ステーション可視性** — 14 ステーションの **マルチラベル分類**（`lymph_node_station_visibility.csv`）
   - **禁止事項（公式明記）**: ファイル名に含まれる station 情報を予測に使ってはならない（station one-hot 入力は違反）

古い日報・claudeSummary・v001〜v004 の「T1=fine / T2=coarse」表記は歴史的なもの。**提出コードのフォルダ割当てとドキュメントは必ず新定義に従う**（一次情報: `survey/competition/docker_instructions_wiki639935_20260811.md` と最新の `reference/tigersqai_challenge/`）。

### 評価指標（公式コードで確定）

- 一次情報は **`reference/tigersqai_challenge/`**（公式評価コード, NCT/TSO）。**Dice/HD/F1/AUROC は自前で書かず公式関数を呼ぶ**（`metrics.evaluate_seg.evaluate` / `metrics.evaluate_cls.evaluate`）。
- Task1/2 = **Weighted Dice ＋ 正規化 Hausdorff**、Task3 = **Weighted F1(@0.5) ＋ AUROC**。
- 集約は **画像→case→全体の階層平均**。**CV は必ず case 単位で集約**して測る（最終スコアが case 平均のため）。
- クラス重み 3/2/1 が効く（weight=3 は小さく重要な10クラス）。詳細・提出フォーマットは `survey/competition/overview.md` §4・§5。
- クラス数の数え方に注意: `labelmap.csv` は fine 31 ID / merged 16 ID（**背景 ID=0 を含む**）。公式評価は背景を除いた **30 / 15 クラス**を採点する（公式 README の表記はこちら）。

### コマンド（公式評価コード）

`reference/tigersqai_challenge/` で実行（Python ≥ 3.11、`uv sync` または `pip install -r requirements.txt`）:

```bash
# 合成データで評価パイプライン一式を動作確認（GT 不要）
uv run python metrics/demo.py --clean

# 実データで評価（gt/ = task1/ task2/ task3.csv、pred/ は method 名のサブディレクトリ）
uv run python metrics/01_evaluate_challenge.py --gt path/to/gt/ --pred path/to/predictions/ --out report.md --save-json
```

API 詳細・色テーブル・集約式は `reference/tigersqai_challenge/metrics/00_README.md`。

**プロジェクト本体の Python 環境**: リポジトリ直下の `.venv`（`--system-site-packages` で作成）を必ず使う。user site の transformers 4.57.2 が huggingface_hub 1.7.1 と非互換で lightning の import が死ぬため、venv 内で transformers>=5 を上書きしている。各実験の `run.sh` が自動 activate する（安定したら `requirements.txt` をコミットする）。

### 最重要の落とし穴

- **マスクは RGB PNG であってクラス ID ではない**。各ピクセルの (R,G,B) を `data/labelmap.csv` で引いてクラス ID に変換する必要がある（`fine_r/g/b`→`fine_id`、`merged_r/g/b`→`merged_id`）。前処理で RGB→ID 変換テーブルを作って一度ラベル画像化しておくこと（公式の `rgb_mask_to_label_mask` も利用可）
- 提出マスクは **正確な RGB 色で塗った PNG（JPEG 禁止）**。認識外の色のピクセルは採点から除外される
- `labelmap.csv` には **`weight` 列**（値 1/2/3）がある。weight=3 のクラス（小さく重要な解剖構造）を軽視しない
- fine ↔ coarse のマッピングも `labelmap.csv` に入っている（`fine_id` 行に対応する `merged_id`）。fine を学習して coarse へ集約することも可能

### 現状の足場

- このリポジトリは **テンプレート初期状態**。実験・fold 定義（`workspace/fold/`）はまだ無い。日報は `daily_reports/20260822.md`（データ棚卸し・改訂タイムライン）から開始
- `reference/tigersqai_challenge/` = **公式評価コード**（NCT/TSO）、`survey/competition/overview.md` = タスク把握メモ、は作成済み
- git は初期コミット済み（テンプレート・ドキュメント類のみ）。**`data/`（symlink）・`reference/`・モデル重みは git 管理外**。別マシンへの移設手順は `SETUP.md`（データ/reference は rsync で取得）

### 提出・締切・ルール（確定。詳細は `survey/competition/overview.md` §5–§7）

- **提出は Docker コンテナ**（zip ではない）。image `tigersqai26_<team>:v<n>` を Synapse evaluation queue へ。**オフライン動作・完全自動**必須。コンテナは公式 evaluator が食う形式（`task1/ task2/ task3.csv`）を出力する。→ `submit/` は **「提出パイプライン (B) Docker コンテナ提出型」** を使う
- **締切 2026/09/15**（評価フェーズ開始 **9/06** に延期、AoE）。**最終提出のみ評価**。**Docker 提出受付は開始済み（8月中旬〜）で、早期テスト提出を主催者が強く推奨** → 互換性フィードバックが貰えるので、モデルが未完成でも動く提出を早めに 1 本通す
- **write-up + 3 分ビデオが必須**（Task1+2 で 1 本、Task3 で別途 1 本）。write-up / Docker instructions は Synapse に公開済み。チャレンジ発表は 2026/09/27（EndoVis セッション、MICCAI はストラスブール開催に変更）
- **外部データは公開のもののみ可**。非公開データ・非公開データで事前学習した重みは **禁止**（ImageNet 等の公開事前学習は可）。使った外部データは開示・引用

### ドキュメントの役割分担

- **`survey/competition/overview.md`** = タスク内容のまとめ（把握フェーズ 7 項目、確定事実 vs 要確認TODO）。タスク仕様を確認するときはまずここ。
- **`claudeSummary.md`** = 横断スコアボード。**実験のスコア（CV/LB）は必ずここに記録する**。学習完了・スコア変化のたびに待たず即追記し、数値で書く（「上がった」ではなく値）。詳細経緯は `daily_reports/`、実験単位の詳細は各 `SESSION_NOTES.md`。

# Competition Workspace

このリポジトリはデータ分析コンペ用の実験管理テンプレートです。
Kaggle だけでなく、grand-challenge.org / CodaBench / 独自プラットフォームなど **Kaggle 以外のコンペにも対応** する想定で運用します。
**判断に迷ったら `competition_DIRECTION.md` の設計意図を確認すること。**

## コンペ開始時の把握フェーズ（学習コードを書く前に必ず実施）

新しいコンペに取り組むときは、いきなり学習コードに入らず、まず以下を **`survey/competition/` 配下に整理**してから実装に着手する:

1. **プラットフォーム特定**: Kaggle か / grand-challenge.org か / CodaBench か / 独自サイトか
2. **タスク定義**: 入出力、クラス数、評価単位（画像単位/ピクセル単位/患者単位 など）
3. **データの所在**: ダウンロード元 URL、サイズ、フォーマット、ライセンス、配置先のローカルパス
4. **評価指標**: 正確な定義（per-class/macro/micro、背景クラスの扱い、など実装上の曖昧さを潰す）
5. **提出形式**: CSV か / 予測ファイル（画像・JSON）の zip か / Docker コンテナか
6. **タイムライン**: Validation phase / Test phase / 最終締切、提出回数制限
7. **ルール**: チーム人数、外部データ可否、事前学習モデル可否、商用ライセンス

この 7 項目を埋めないまま実装を始めない。

## アイデア提案の原則（堅実＋爆発）

**アプローチやアイデアを提案するときは、必ず「堅実案」と「爆発案」の両方を出すこと。**

- **堅実案**: 既知の手法、定石、段階的改善。確実にスコアが上がる見込みがあるもの
- **爆発案**: 常識外れ、異分野からの転用、誰もやらなそうなアプローチ。失敗リスクは高いが当たれば大きいもの

例:
```
堅実: encoder を efficientnet_b0 → b4 に変更（+0.5%程度の改善見込み）
爆発: セグメンテーションを捨てて物体検出で解く / 全く別のモダリティの事前学習済みモデルを転用
```

局所解に陥らないために、爆発案は「それは普通やらないだろう」くらいがちょうどいい。

## 基本ルール

- 実験は `workspace/` 以下で行う
- Claude用: `expA00_baseline` (アルファベット+数字2桁)、人間用: `exp200_name` (数字3桁)
- 各実験フォルダには必ず `SESSION_NOTES.md` を作成する
- 大きく方針が変わる時だけ新しいexp番号にする。微調整は同じフォルダ内で
- 結果・知見・戦略はすべて日報 `daily_reports/YYYYMMDD.md` に集約する（個別planファイルは作らない）

## 学習コードの鉄則

- **AMP (Mixed Precision) は常にON** (`precision: 16-mixed`)
- **チェックポイント再開は必須** (`save_last=True` + `ckpt_path`)。ただし Lightning 2.x の `save_last` は「best 保存時のコピー」でしかなく毎 epoch 保存ではない → `monitor=None, save_top_k=1, every_n_epochs=1` の rolling checkpoint（例: `latest.ckpt`）を別 ModelCheckpoint で必ず併設し、レジュームはそちらを優先する
- **シード固定** (`pl.seed_everything(seed, workers=True)`)
- ハイパーパラメータはすべてconfigで管理（ハードコーディング禁止）
- **ログはPythonの `logging` モジュールで出力する**（`print`禁止）
  - コンソール（INFO）とファイル（DEBUG）の両方に出力する
  - ログファイルは `results/{experiment_name}/foldN/` にタイムスタンプ付きで保存する
  - フォーマット: `%(asctime)s | %(levelname)s | %(message)s`
- **全出力は `results/{experiment_name}/foldN/` に集約する**
  - best_model, checkpoint, log, training_log.json, **config.yaml** を全て同一ディレクトリに保存する
  - **config.yamlは学習開始時に自動コピーされる**（再現性のため）
  - `experiment.name` をconfigに定義し、パラメータ変更時は名前を変える
  - 同名の実験ディレクトリが存在する場合は `_001`, `_002` とナンバリングして上書きしない
- 実行は `run.sh` 経由で行う（各実験フォルダに `run.sh` を作成し、学習・推論はすべてこのスクリプト経由）
- **日報 `daily_reports/YYYYMMDD.md` がすべての記録の中心**
  - 1日1ファイル。戦略・ロードマップ・知見・実験結果をすべてここに書く
  - **知見が出たら都度追記する**（学習完了、エラー発見、スコア変化など、待たずに即記録）
  - 実験結果（CVスコア等）は数値で明記する
  - 過去分は編集せず履歴として保持
  - **セッション開始時に最新の日報を読んで状況を把握する**
  - テンプレート:
    ```markdown
    # 日報 YYYY-MM-DD

    ## コンペ情報
    - **コンペ**: コンペ名
    - **締切**: YYYY-MM-DD（残りN日）
    - **評価指標**:
    - **LB状況**:

    ## 今日やったこと
    ### 1. ...

    ## 数値まとめ
    | 実験 | データ量 | CV | LB | 状態 |
    |------|---------|-----|-----|------|

    ## 戦略・ロードマップ

    ## 判断・知見
    <!-- 今日得られた重要な判断や知見 -->

    ## データ在庫
    <!-- 利用可能なデータソースの整理 -->

    ## 次にやること
    - [ ] TODO
    ```

## Fold設計（最重要）

**安易にランダムKFoldにしない。データの性質を先に確認する。**

- 時系列 → TimeSeriesSplit
- グループ構造あり → GroupKFold
- クラス不均衡 → StratifiedKFold
- グループ+不均衡 → StratifiedGroupKFold
- fold設計の理由と各foldの分布はSESSION_NOTES.mdに記録する
- CV/LBの相関を確認し、相関が弱ければfold設計を見直す

**fold割り当ての永続化（`workspace/fold/`）:**
- fold割り当ては `workspace/fold/{version}/folds.csv` に保存し、全実験で共有する
- `generate_folds.py` で生成。バージョン管理付き
- 前処理やデータを変更した場合は新バージョンを作る。古いバージョンは削除しない
- config.yamlの `cv.folds_csv` で使用バージョンを指定する
- 設計意図・切り方の詳細は `workspace/fold/README.md` に記載

## 提出パイプライン（`submit/`）

提出用コードとモデルは `submit/` 以下で管理する。提出形式はプラットフォームによって異なるため、以下のいずれかに沿って構成する。

### 共通ルール（プラットフォーム非依存）

- **命名**: `v001_説明`, `v002_説明`, ... の連番形式
- **自己完結**: 提出用スクリプトは前処理・後処理を内部に持ち、`workspace/` の学習コードに依存しない
- **環境判別**: 実行環境（Kaggle / grand-challenge / ローカル）を自動判別してパスを切り替える
- **推論パラメータ**: ファイル冒頭の定数で管理（ハードコーディング散在を避ける）
- **出典記録**: スクリプト冒頭の docstring に `Source: workspace/expXXX/.../best_model/` とコピー元パス・CV スコアを明記する
- **モデル取り扱い**: `workspace/` から `best_model/` をコピー。`model/` は git 管理外（`.gitignore`）。サイズが大きいため
- **ローカルテスト必須**: 提出物（CSV / 予測ファイル / Docker イメージ）をローカルで生成まで確認してから提出する
- **提出物の検証**: プラットフォームの要件（ファイル数・命名規則・値の範囲・欠損）を提出前に必ずチェック
- **アップロードは手動**: Kaggle Dataset / grand-challenge Submission ページ / その他への実アップロードはユーザーが手動で行う（Claude は実行しない）
- **提出履歴**: `submit/SUBMISSIONS.md` に全提出を記録する。実験フォルダ、モデル元パス、fold 定義、学習データ、学習/推論パラメータ、前処理、CV/LB スコアを必ず記載

### Kaggle の場合（`submission.csv` 提出型）

```
submit/v001_baseline/
├── notebook.py          # Kaggle Notebook にそのまま貼れる推論スクリプト
└── model/               # 学習済みモデル一式
```

- エントリポイントは `notebook.py`
- 出力は `submission.csv`。**行数・カラム名・欠損・値の範囲**を必ず検証してから提出
- git には `notebook.py` のみコミット

### Kaggle 以外の場合（grand-challenge.org / CodaBench / 独自サイト など）

提出形式は主に次の 2 タイプ。コンペ仕様に合わせて選ぶ。

**(A) 予測ファイル提出型**（画像・マスク・JSON を zip してアップロード）

```
submit/v001_baseline/
├── predict.py           # 入力ファイル群を読んで予測を出力するスクリプト
├── run.sh               # predict.py → 出力検証 → zip 化を一発で行う
├── requirements.txt     # 再現性のため固定
├── model/               # 学習済みモデル一式（.gitignore）
└── output/              # 生成された提出物（.gitignore）
```

- 入出力ディレクトリは `predict.py` 冒頭の定数で指定（`INPUT_DIR` / `OUTPUT_DIR`）
- 出力は **ファイル名・サイズ・dtype・値域** まで厳密に公式仕様へ一致させる
- 提出前に「入力ファイル数 == 出力ファイル数」「stem 名の一致」を assert

**(B) Docker コンテナ提出型**（algorithm container として提出）

```
submit/v001_baseline/
├── Dockerfile
├── process.py           # grand-challenge の algorithm interface を実装
├── requirements.txt
├── test/                # ローカル検証用のサンプル入出力
│   ├── input/
│   └── expected_output/
├── build.sh             # docker build
├── test.sh              # ローカルでコンテナを回して出力を検証
├── export.sh            # docker save → tar.gz（提出物）
└── model/               # .gitignore
```

- プラットフォーム側の I/O 契約（入力パス / 出力パス / ファイル形式）を最優先で守る
- ビルド後に必ず `test.sh` でローカル回帰テスト。本番と同じ入出力パスで動くことを確認
- イメージサイズ・GPU 要件・推論時間制限を事前に把握し、必要ならモデル軽量化や ONNX 化を検討
- gitには `Dockerfile` / `process.py` / スクリプト類のみコミット（`model/` と `test/` の大容量データは除外）

## エラー分析の原則（スコアの前に出力を見ろ）

**スコアを上げようとする前に、まず出力を観察して「何が悪いか」を特定する。**

- 実験後は必ず prediction vs ground truth を目視確認する（最低20件）
- エラーの種類を分類し、パターンを把握する（分類はタスク依存。事前にリスト化せず、出力から帰納的に発見する）
- エラーの種類に応じた対策を打つ（前処理・後処理・推論戦略・学習データ・モデル変更）
- スコアという数字だけ見てパラメータを変える盲目的な探索をしない

順序: 出力を読む → 何が悪いか特定 → 原因に対処する → スコアで検証

## 前処理・評価

- 正規化はtrainデータの統計量で計算する（testの情報を使わない）
- コンペの評価指標を正確に再現する（既存実装のパラメータも確認）
- Augmentationはまず弱めで、過学習が確認されてから強める
- single modelのCV/LBを記録してからアンサンブルする
- 提出前の検証はプラットフォームに応じて行う:
  - Kaggle (CSV): 行数・カラム名・欠損値・値の範囲
  - 予測ファイル (画像/JSON): ファイル数・命名規則・dtype・値域・サイズ
  - Docker: ローカル回帰テスト、入出力パス、GPU/時間制限

## リファレンスコード

- `reference/` 直下には現在 **`tigersqai_challenge/`（公式評価コード）のみ**がある。2.5D セグメンテーションテンプレート等の学習コード雛形はこのリポジトリには未配置
- `reference/` は git 管理外（`.gitignore`）。別マシンでは `SETUP.md` の手順で rsync する

## 利用可能なSkills

- `/survey-papers [キーワード]` - 論文・解法調査（メインコンテキストを汚さず別コンテキストで実行）
- `retrospective-codify` - 先に知っておくべきだった知見をast-grep ルール / skill / CLAUDE.mdなどに言語化する
- `emprical-prompt-tuning` - sub agentが上手く動くようなagent 向けテキスト指示を最適化する

## Custom Agents

状況に応じて自動的にサブエージェントに委譲される。並列実行も可能。

- **kaggle-researcher** (sonnet) - 論文・類似コンペ解法・ディスカッション調査。Kaggle に限らず grand-challenge.org や CodaBench など他プラットフォームのコンペ調査にも使う
- **data-analyst** (sonnet) - EDA・可視化・特徴量分析。データの全体像把握に
- **code-reviewer** (sonnet) - ML/DLコード品質レビュー。読み取り専用で安全