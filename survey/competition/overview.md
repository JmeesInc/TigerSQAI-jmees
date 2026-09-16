# TIGER SQ-AI Challenge — コンペ把握メモ

> ⚠️ **2026-09-08 確定（公式 Docker Instructions / 最新評価コード）**: **Task1 = merged/coarse (15+bg)、Task2 = fine (30+bg)**。本リポジトリの従来表記（Task1=fine）とは**逆**。また **Task3 はファイル名の station 情報を予測に使うこと禁止**。


> 把握フェーズ 7 項目（CLAUDE.md 参照）。**確定事実**と **要確認(TODO)** を分けて記載する。
> 一次情報: データ仕様 = `data/README.md` / 評価・提出仕様 = `reference/tigersqai_challenge/`（**公式評価コード**, NCT/TSO の MICCAI2026 リポジトリ）/ 運営情報 = Synapse ページ。
> 最終更新: 2026-06-17

- **コンペ**: [Tiger SQ-AI-Challenge](https://www.synapse.org/Synapse:syn74209386/wiki/639462)（**EndoVis 2026 @ MICCAI 2026** のサブチャレンジ）
- **題材**: 低侵襲食道切除術（minimally invasive esophagectomy, 腹腔鏡/ロボット支援）の手術フレーム。リンパ節郭清に関わる解剖構造のセグメンテーションとステーション可視性判定。
- **参加登録**: Synapse アカウント＋Google フォームから join 申請 → 承認後にデータアクセス付与（要 Certified User でアップロード）。

---

## 1. プラットフォーム

- **Synapse（Sage Bionetworks）** 上で開催。Kaggle ではない。提出は **Synapse evaluation queue に Docker コンテナ**を投げる方式（§5）。
- leaderboard あり（Challenge Results ページ `639941`）。採点は最終提出のみ → 提出回数自体の上限は明記なし（最終のみ評価）。

## 2. タスク定義（3つ）

データは 1 フレーム（静止画）単位。動画ではない。

| # | タスク | 種類 | 出力 | 正解ラベル | クラス数 | 指標 |
|---|--------|------|------|-----------|---------|------|
| 1 | Fine セグメンテーション | セマンティックセグメンテーション | ピクセル単位クラス | `masks_fine/` | **31 ID（0=背景含む）**※ | Weighted Dice ＋ 正規化 HD |
| 2 | Coarse セグメンテーション | セマンティックセグメンテーション | ピクセル単位クラス | `masks_coarse/` | **16 ID（0=背景含む）** | Weighted Dice ＋ 正規化 HD |
| 3 | リンパ節ステーション可視性 | **マルチラベル分類** | 14 ステーションの各 **確率[0,1]** | `lymph_node_station_visibility.csv` | 14 ラベル | Weighted F1(@0.5) ＋ AUROC |

※ 公式評価 README は "30 fine-grained" と表記（背景を除いた数え方）。実際の color table / labelmap は ID 0〜30 の 31 個。

- Task 1↔2 は同じ解剖を粒度違いで見るもの。`labelmap.csv` の `fine_id → merged_id` で fine 予測を coarse に集約可能。
- Task 3 の 14 ステーション: `6L 6R 7L 7R 8 9 10L 10R 11L 11R 12L 12R 13L 13R`。
- **評価単位**: Task 1/2 はピクセル単位（クラス weight 付きの可能性大）、Task 3 は画像単位のマルチラベル。

## 3. データの所在

- ローカル: `data/`（symlink → `<challenge data root>`、**読み取り専用扱い**）。
- ダウンロード元: Synapse（`SYNAPSE_METADATA_MANIFEST.tsv` が同梱）。
- **全体規模（最終）**: **700 フレーム / 50 ケース / 7センター・5カ国**。1ケース=14フレーム（LNステーション各1枚）。腹腔鏡:ロボット = 50:50。
- **train/test split**: 80/20 = **train 40 ケース / test 10 ケース**。各分割でモダリティ 50:50 を維持し、**1センターを train から除外して汎化を評価**（hidden test に未知センターが入る）。
- **段階配布（2026/08/22 時点で第3バッチまで公開済み）**: 第1バッチ（5末, 10ケース140枚）→ 第2バッチ（7/17, +5ケース）→ **第3バッチ（8月中旬）で Task1/2 の訓練データは完全**（主催者アナウンス）。test 10 ケースは非公開（Docker で採点）。
- 今ローカルにあるもの: **524 枚 / 40 ケース / 6 センター**（center_1×16, center_2×8, center_3×5, center_4×3, center_6×3, center_7×5 ケース）。**center_5 が存在しない** → train から除外された held-out センターとみられる。`masks_fine/` `masks_coarse/` も各 524 枚（画像と同名）。
- **Task 3 のラベルは未完**: `lymph_node_station_visibility.csv` は **308 行のみ**（center_1: 212 / center_2: 5 / center_3: 17 / center_4: 13 / center_6: 33 / center_7: 28）。最終更新は 8月末予定で、**Task 3 の最終データ数は Task1/2 より少ない可能性あり**（主催者アナウンス）。→ Task 3 の学習は「ラベルがある行のみ」を使う設計にする。
- 解像度: **1920×1080 ×379 / 3840×2160 ×107 / 1280×720 ×38**（4K が混入。前処理のリサイズ方針・提出マスクの原寸復元に注意）。
- **ファイル名の例外**: `center_2_case_8_12L_frame_84.png` / `center_2_case_8_6L_frame_1733.png` の 2 枚だけ `{case_id}_{station}_frame_{n}.png` 形式。`{case_id}_{station}.png` 前提のパーサはこの 2 枚で壊れる（case 抽出は `center_\d+_case_\d+` の正規表現で行うこと）。
- ライセンス: **本チャレンジ内利用に限定**。参加チーム外への共有は厳禁。再配布・公開・商用利用は事前許諾なしに禁止。

### マスクの読み方（最重要）

- マスクは **RGB PNG**。ピクセル値はクラス ID ではなく色。
- `data/labelmap.csv` で `(fine_r,fine_g,fine_b) → fine_id` / `(merged_r,merged_g,merged_b) → merged_id` を引いて ID 化する。
- 前処理で **RGB→ID ルックアップを作り、一度 ID ラベル画像にしておく**（毎 epoch 変換しない）。
- `labelmap.csv` の列: `fine_id, fine_name, fine_r/g/b, weight, type, merged_id, merged_name, merged_r/g/b`。

## 4. 評価指標（**公式コードで確定** — `reference/tigersqai_challenge/metrics/`）

- **Task 1 / 2（seg）**: **Weighted Dice (DSC)** ＋ **正規化 Hausdorff Distance (HD)**。
- **Task 3（分類）**: **Weighted F1（閾値 0.5）** ＋ **AUROC（確率から閾値非依存）**。

### Dice / HD の定義（`metrics/metrics.py`, `evaluate_seg.py`）

- Dice はクラス k・画像 I 単位。**両方（pred と gt）にクラスが無ければ Dice=1.0、片方だけなら 0.0**。
- HD は正規化版: `max(h(P→G), h(G→P)) / sqrt(H²+W²)`、範囲 [0,1]、**小さいほど良い**。
- **集約は階層的**:
  ```
  画像スコア = Σ(metric(k,I) × w(k)) / Σ w(k)   # クラス重み付きマクロ平均
  case スコア = case 内画像スコアの平均
  最終スコア   = case スコアの平均             # ← case 単位で効くので CV も case 集約で評価する
  ```
- **クラス重み（確定）**:
  - **weight=3**: Lymph Node, R/L Inferior Pulmonary Ligament, R/L Subclavian Artery, Right Vagal Nerve, R/L Recurrent Laryngeal Nerve, Pulmonary Artery, Right Bronchial Artery
  - **weight=2**: Trachea, R/L Main Bronchus, Esophagus, Aorta, Azygos Vein, Superior Caval Vein, Pleura, Pericardium, Inferior Pulmonary Vein
  - **weight=1**: 残り全部（背景・Instrument 等）
  - Task 2 は構成する fine クラスの **最大 weight** を採用。
- **背景クラス (ID=0) も評価対象**（weight=1）。weight=1 は背景・Instrument・Other・Fatty_Tissue_Esophagus・Lung・Fatty_Tissue・Pool_of_Blood・Resection_Area・Gastric_Conduit・Omentum・Thoracic_Duct の11クラス。
- **Task 3**: 全14ステーションが weight=1 → 実質ただのマクロ平均。F1 は全フレーム横断でグローバルに TP/FP/FN を集計。
- case_id のパースは `rsplit("_", 1)[0]`（最後の `_` で分割した前半）。提出ファイル名の付け方に直結。
- **ランキング**: seg は `(rank(Dice↓)+rank(HD↑))/2`、cls は `(rank(F1↓)+rank(AUROC↓))/2`。最終順位はタスク順位の平均。bootstrap CI(1000)・Wilcoxon 検定あり。
- 注意: 認識外の RGB 色のピクセルは **unknown 扱いで採点から除外**。全ピクセルを定義済み色のいずれかに割り当てること。

> CV はこの公式コードをそのまま流用して測る（自前で Dice を書かない）。`from metrics.evaluate_seg import evaluate` / `evaluate_cls`。

## 5. 提出形式（**Docker コンテナ提出**型 — Synapse で確定）

> ⚠️ **重要**: 提出は予測ファイルの zip ではなく **Docker コンテナ**。`submit/` は CLAUDE.md「提出パイプライン **(B) Docker コンテナ提出型**」に従う。

- 提出物は2点: **(1) Docker コンテナ**（`syn74209386/wiki/639935` Docker Instructions）＋ **(2) Write-Up**（手法説明, `639934`）。
- **Docker 命名**: image `tigersqai26_<team_name>` / tag `v<version>`（`v1`, `v2`…バージョン毎に提出）。
- **Synapse の evaluation queue** に submit（要 Certified User アカウント）。Write-Up project は `public` に `Can Download` で共有し、Docker/Write-Up を **TigerSQ-AI Challenge Admin** にも共有 or メール通知。
- 提出後に確認メール。無効なら理由付きで通知され再提出可。**評価されるのは最終提出のみ**（＝反復は自由だが採点は最後の1つ）。
- **技術要件**: コンテナは **インターネット接続なしで動作**（オフライン）／**完全自動**（ユーザー操作なし）／結果生成コードはすべてコンテナ内に同梱。

### コンテナが出力すべき予測フォーマット（公式評価コード — `reference/.../metrics/00_README.md` §2）

コンテナは入力フレームを読んで、以下の構成で予測を出力する（これを公式 evaluator が採点）:

```
<output>/
  task1/  <case>_<station>.png ...   # RGB PNG, 入力と同名（フラット, case サブdir無し）
  task2/  <case>_<station>.png ...   # 同上（merged 16色）
  task3.csv                          # 確率値
```

- **Task1/2**: 各ピクセルを**正確な RGB 色**で塗った PNG（fine=31色 / merged=16色）。**JPEG 禁止**（非可逆圧縮で色が壊れる）。1入力=1出力、ファイル名は入力と完全一致。
- **Task3 CSV**: 列は厳密に `case_id,6L,6R,7L,7R,8,9,10L,10R,11L,11R,12L,12R,13L,13R`。値は **[0,1] の確率**（binary ではない）。`case_id = {case}_{annotated_frame}` 例 `center_1_case_6_6R`。全フレーム分の行が必要（順不同、case_id で突合）。

## 6. タイムライン（**2026/08 主催者アナウンスで改訂済み**。すべて AoE / UTC−12）

| 日程 | 内容 |
|------|------|
| 2026/04/10 – 08/31 | 登録期間 |
| 2026/05 末 | 訓練データ 第1バッチ公開（10ケース） |
| 2026/07/17 | 第2バッチ公開（+5ケース）。write-up と Docker コンテナ作成の instructions 公開 |
| 2026/08 中旬 | **第3バッチ公開 → Task1/2 の訓練データ完全**。**Docker 提出受付開始**（←現在） |
| 2026/08 末 | Task 3 ラベルの最終更新（予定） |
| **2026/09/06** | **評価フェーズ開始**（旧 9/01 から延期） |
| **2026/09/15** | **提出締切** |
| 2026/09/27 | チャレンジ発表（EndoVis 2026 セッション。**MICCAI はストラスブールに変更**） |

- 締切まで残り約3週間強（2026-08-22 時点）。**遅延提出は不可**（技術的問題のみ要相談）。
- 単一の評価フェーズ（Sep 6 開始、Sep 15 締切）。別個の validation phase は明記なし。
- **早期の Docker 提出を強く推奨**（主催者側で互換性テスト・フィードバックあり。評価スクリプトの検証にもなる）。締切間際の一発提出は避ける。
- **write-up 提出時に 3 分のビデオプレゼン必須**: Task1+2 で 1 本（統合）、Task 3 で別途 1 本。

## 7. ルール（**Synapse で確定** — `639938` Competition Guidelines）

- **外部データ**: チャレンジデータに加え **公開データセットのみ許可**。**非公開データ／非公開データで事前学習したモデルは禁止**。→ **ImageNet 等の公開事前学習はOK**、独自・社内データやそれで学習した重みはNG。使用した外部データは開示・引用必須。
- **提出**: Docker 完結（結果生成コードは全てコンテナ内）。**オフライン動作・完全自動**必須。完全な Docker 提出のみ受理。
- **評価**: 各チーム**最終提出のみ**評価。評価サーバ悪用・test ラベル/データ漏洩は失格。
- **主催者所属**の参加者は leaderboard/論文掲載可だが**受賞対象外**。
- **公表ポリシー**: 共同ジャーナル論文を作成予定。**joint paper 公開前は結果を公表不可**。最終提出時に手法の詳細記述（テンプレ準拠）を添付すると共著対象。
- チーム参加は推奨だが必須でない（Team Captain が招待/除名）。**要確認(TODO)**: チーム人数上限は明記なし。
- 商用ライセンス可否は未記載。EndoVis 本体ページの規約も併せて遵守（`EndoVis_Rules.pdf`）。

---

## CV 設計（このコンペの結論）

- ファイル名 `{case_id}_{station}.png` に **case のグループ構造**あり。
- → **GroupKFold（case でグループ化）**。同一 case が train/val に跨るとリーク。
- fold 割り当ては `workspace/fold/{version}/folds.csv` に永続化して全実験で共有（CLAUDE.md「Fold設計」）。

## 評価ツール（公式コードをそのまま使う）

- 場所: `reference/tigersqai_challenge/`（origin: `gitlab.com/nct_tso_public/challenges/miccai2026/tigersqai_challenge`）。
- 環境: Python ≥ 3.11、`uv sync`（or `pip install -r requirements.txt`）。
- 単一手法の評価例:
  ```
  python metrics/01_evaluate_challenge.py --gt gt/ --pred predictions/ --out report.md --save-json
  ```
- 自前 CV でも `metrics.evaluate_seg.evaluate` / `metrics.evaluate_cls.evaluate` を呼ぶ。Dice/HD/F1/AUROC を再実装しない。
- 合成データで挙動確認: `python metrics/demo.py --clean`。

## 未解決の TODO

- [x] ~~評価指標の厳密定義~~ → 公式コードで確定（§4）
- [x] ~~提出物のフォーマット~~ → 公式コードで確定（§5）
- [x] ~~提出手順~~ → **Docker コンテナを Synapse evaluation queue へ**（§5）
- [x] ~~タイムライン~~ → 締切 **2026/09/15**、Part 2 **2026/07**（§6）
- [x] ~~外部データ・事前学習可否~~ → **公開データ/公開事前学習のみ可、非公開はNG**（§7）
- [x] ~~Part 2 / test set 配布~~ → Part 2 は7月、test 10ケースは非公開（§3）
- [ ] チーム人数の上限（明記なし。必要なら Discussion で確認）
- [ ] 商用ライセンスモデルの可否（明記なし）
- [ ] Docker I/O 契約の詳細（入力パス/出力パス/実行時間・GPU 制限）→ **Docker instructions は 7/17 に公開済み**（`639935` / Template `639939`）。提出受付も開始済みなので早めに読んで早期テスト提出する
- [ ] Task 3 最終ラベル（8月末予定）の反映確認 → 更新されたら `lymph_node_station_visibility.csv` の行数・センター分布を再確認し、fold を再生成
- [ ] write-up テンプレと 3 分ビデオ（Task1+2 で1本 / Task3 で1本）の要件確認

> Synapse 情報は 2026-06-17 に公式 wiki（REST API）から取得・反映済み。一次情報の出典は各節の wiki ID。
