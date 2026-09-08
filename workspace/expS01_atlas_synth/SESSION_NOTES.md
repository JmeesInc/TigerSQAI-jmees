# expS01_atlas_synth — 解剖アトラス由来の合成データ

## 目的
公開解剖アトラス（Z-Anatomy / BodyParts3D）と公開CT（TotalSegmentator）から
内視鏡視点のラベルマップを生成し、(a) 形状事前分布、(b) Task3 の幾何的解法、
(c) 見た目拡張 の3経路で本タスクに効かせる。

## 位置づけ（重要）
- **締切 2026-09-15 のスコアには間に合わない前提**。write-up の差別化と次期用の資産。
- 今週の優先度は v003 提出 > expA08/09/11 完走 > 本実験。

## 方針（2026-09-08 決定）
- 忠実度は **T0（クラスIDラベルマップのみ）で十分**。T1（内視鏡光学）まではオプション。
  写実的レンダリングと synth→real 事前学習は今回やらない。
- **CAD 手作りは不採用**。既存公開アトラスの変換で幾何を得る。
- 採用する経路: 爆発②（幾何的 Task3）を本命、爆発①（実データのアトラス登録）を
  その前提として実施。堅実案（マスク条件付き画像生成）は独立並行。
- **爆発① → 爆発② は依存関係**: 登録で得たカメラ姿勢 + 実データの可視ステーションラベル
  → 空間カービングでステーション体積を逆算 → 姿勢サンプルの数だけ Task3 ラベルが無料で出る。
  この経路なら **ステーション命名法が未確定でも進められる**。

## 未確定・要確認
1. **14ステーション（6L,6R,7L,7R,8,9,10L,10R,11L,11R,12L,12R,13L,13R）の命名法が不明**。
   `data/README.md` にも記載なし。IASLC 肺癌ノードマップ / 日本食道学会 / AJCC食道 /
   TIGER study 独自マップ のいずれか。8 と 9 だけ左右が無い点が手がかり。
   → 主催者資料（Synapse wiki）で要確認。ただし空間カービング経路は命名法非依存。
2. Z-Anatomy / BodyParts3D に神経・胸管・気管支動脈の個別メッシュがあるか（要疎通確認）。
3. CC-BY-SA の share-alike が「学習利用」と「合成データ配布」でどう効くか。

## 最大の技術的リスク
アトラスをそのまま描画すると全構造が無遮蔽で写り、実データのラベル統計と別物になる。
実術野では大半が縦隔脂肪と胸膜に覆われている。
→ **脂肪シェル + 剥離進行度 t のプロシージャル生成が必須**。ここを省くと全部無駄になる。

## 成果物
- `PROMPTS_FOR_ASTRA.md` — ChatGPT Astra 向けプロンプト集（Prompt 0 + タスク A〜E）
  - **原則: Astra にはコードのみ書かせ、チャレンジデータは渡さない**（規約でチーム外共有禁止）

## 検証ゲート（未実装・後回し）
実 524 マスク vs 合成の統計比較（クラス出現率 / 面積分布 / 共起行列 / 隣接行列の JS 距離）と、
ラベルマップのみから実/合成を判別する分類器の AUC。AUC≈1.0 なら事前分布として使えない。
→ 合成データが出てきた時点で実装する。

---

## 2026-09-08 進捗: Astra タスクA/B 完了 → 較正パック作成

### Astra 成果（Mac 側 `/Users/kikus/Desktop/MICCAI2026/TigerSQAI`、未同期）
- タスクA: アトラス比較・class_mapping.yaml・blend棚卸しスクリプト。
  - BodyParts3D 現行アーカイブは **CC BY 4.0**（旧サイトの BY-SA 2.1 JP と別）
  - Z-Anatomy に迷走神経Curve・縦隔リンパ節群メッシュ・胸膜・大網あり。
    **(d) 対応不能 7 クラス**: IPL R/L, Pericardium, R bronchial artery, RLN R/L, Thoracic duct
  - ステーションは TIGER study 統合マップが有力仮説（8=AP window, 9=subcarinal なら整合）。未確定
- タスクB: Tier0 レンダラ実装・Blender 4.5.9 実証済み。pass_index直接出力、
  30度鏡の円錐幾何、脂肪/胸膜遮蔽+5段階剥離、CT+TPS経路。実測 ~5.94s/枚 (Mac CPU, 1024x576)

### こちらの検証結果（重要）
1. **fine_id 対応は完全一致で確定**: Astra 内部 1..30 = 公式 fine_id 1..30、背景=0
2. **実 528 マスクの集計統計を算出**（`assets_local/astra_calibration_pack_20260908.md` §2）:
   - 背景は実データでも平均 21.7%（胸壁未ラベル）→ 胸壁→背景0 の暫定規則は実データと整合
   - 可視クラス数中央値 15。Instrument 90% / Resection area 81% / Pool of blood 54% 出現
     → **実フレームは全て剥離進行中**。合成の pre_incision フェーズはほぼ不要（進行度priorを後期へ）
   - **(d) クラスの補充優先度が確定**: Pericardium 出現率 70.8% で最優先（心臓表面proxy）、
     IPL R/L 26/27% 次点、**R bronchial artery は出現率 0.0% → 対応不要で確定**
3. 公式パレット生成: `assets_local/palette_fine_official.yaml`（公開評価コード由来なので共有可）

### 成果物
- `assets_local/astra_calibration_pack_20260908.md` — Astra へ渡す較正情報 + 次プロンプト（B改訂+C）
- コンタクトシート目視: 遮蔽モデルは意図通り。ただし早期フェーズが胸膜一色 → 上記較正で解消予定

### TODO
- [ ] Astra 成果物一式を Mac からこのリポジトリへ同期（記録・再現性）
- [ ] タスクC 完了後: 実/合成判別器の検証ゲート実装（こちらで、実データを使って）

## 2026-09-08 追記: Astra 成果物をリポジトリへ同期 → Linux 側で独立検証

同期内容: `render/`(レンダラ一式+tests), `configs/`, `assets/`(class_mapping+evidence),
`docs/task_a_atlas_selection.md`, `scripts/audit_blend_inventory.py`, `outputs/`(release 10枚+validation 26枚, git管理外)

### 検証結果（このサーバ, Linux + Python 3.11）
1. **単体テスト 7/7 合格**（python3.11 の `.venv-render` を新設、requirements 導入のみ）。
   macOS/3.12 → Linux/3.11 の移植性 OK
2. **render.audit_outputs が release/validation2 とも合格**（PNG L/uint8、EXR Z/float32/mm、メタ整合、剛体スコープ幾何）
3. ラベル PNG 実測: mode L / uint8 / 値域 0..30 で仕様通り。meta に missing_anatomy_class_ids・
   provenance SHA・棄却理由の記録あり。visible_stations は null（命名法未確定のため正しい）
4. **合成26枚 vs 実528枚の統計差分**（較正パック§2 と同じ物差し）:
   - Pleura 面積 55.0% vs 実 16.4%、クラス数/フレーム中央値 5 vs 15、Instrument 出現 4% vs 90%
   → 剥離進行度 prior が早期寄り + 器具過少。**較正パックのB改訂指示そのものが数値で裏付けられた**
   - SCV 出現 35% vs 実 11.6%（視点が上縦隔寄り）、Esophagus 35% vs 86%
   - Pericardium/IPL/blood 0%（既知の (d)/オプション無効）
5. `.gitignore` に `.venv-render/` を追加（outputs/ は既存パターンで除外済み）

### 運用メモ
- バンク生成をこちらで回す場合: Blender 未導入。tarball 展開で導入可（root 不要）

## 2026-09-09 Astra: B改訂 + C Stage 1

- fine_id/背景0を確認済みに更新。公式配色・共有集計を保存。背景中央値はユーザー回答に従い全フレーム18.0%。
- 後期切断Beta、器具/剥離面/血液prior、可視クラス数11〜18、虚脱量を改訂。心膜11とIPL8/9は既存形状拘束のprovisional proxyとして実装、一括無効化可。22は補充対象外、RLN/胸管は未実装。
- `registration/` に複数ポート配置のバンク生成、記述子kNN、重み付きIoU、低信頼棄却、topK姿勢、同スキーマprior提案出力を実装。実マスクは読んでいない。
- Mac CPU/Blender4.5.9、256×144、24枚/2並列: 57.72秒、2.40秒/枚。可視クラス数中央値14、器具54.17%、心膜62.50%、肺66.67%、剥離面79.17%。血液/IPL/IPVは可視0%。SVC面積8.76% vs実0.31%など大きな差が残り、統計一致とは判断しない。
- 7+8テスト通過、24枚監査通過。自己参照検索24/24 IoU=1→prior出力→別seed4枚の再描画・バンク化まで通過。別seed2枚の検索はIoU0.197/0.074で両方棄却。実登録精度は未検証。
- 手順・失敗検出・バンク規模・CVリーク注意は `docs/task_bc_calibration_stage1.md`。差分は `render.calibration_report` で再現。
- 次: Linuxでtrain-caseのみのIDマスクを使い2,048枚程度のバンクで採択率/coverage確認。実フレームのposes/summary等は共有しない。統計不一致の大きい構造・器具/血液の配置を調整。

## 2026-09-09: B改訂/C Stage1 の Linux 検証 → 原因を2つに分解

### 環境構築（再現性確保）
- Z-Anatomy を GitHub raw URL から取得。**Startup.blend の SHA-256 が Astra の記録と完全一致**
  (`9f08a17e...`) → 同一アセットで再現可能。`Z-Anatomy/` は scratchpad への symlink（git 管理外）
- Blender 4.5.3 LTS を `/data4/src/shunsuke/opt/` に非 root 展開
- **15 テスト全通過**（render 7 + registration 8）。CUDA 4並列で **5.5〜7.4 秒/枚**
  → バンク 2048 枚 ≈ 3〜4h、8192 枚 ≈ 12〜16h
- 実マスク `workspace/data_proc/labels_fine_1024/` は mode L / 1024x576 / ID 0..30 で
  `registration.features.read_label` の要件を満たす（RGB 変換不要）

### 診断（A/B 3本で切り分け）
1. **心膜 proxy が IPV(12) を遮蔽【確定バグ】**: provisional 有効 → IPV 0.0% /
   `--disable-provisional-structures` → 4.2%。心膜 proxy の source が心房を含み
   margin 1.5mm の膨張和が肺静脈流入部を包む。IPV は実 58.3%/3.49% の主要クラス
2. **カメラと剥離窓が結合していない【本命・構造的】**:
   - `visible_class_count` フィルタ除去 → 数 pp しか動かず主犯ではない
   - 剥離窓 0.45 倍 → 逆振り切れ（Pleura 面積 64.75% vs 実 16.44%、可視クラス数中央値 8 vs 15、
     出現率絶対差 786pp → 850pp と悪化）
   - **実データは「Pleura 面積 16.4%（大） かつ 可視クラス数 15（多） かつ 背景 18%」**
     = 狭い露出窓を至近距離から覗く状態。窓サイズ 1 次元では到達不能
   - → 注視点を剥離窓中心から選び、撮影距離を窓サイズに結合する必要がある

### 成果物
- `assets_local/astra_round3_findings_20260909.md` — 上記診断 + ラウンド3 プロンプト + 受入条件
- 受入条件は **「出現率の絶対差 合計（31クラス）を 786pp から減らす」を主指標**とし、
  Pleura 面積 / 可視クラス数中央値 / 背景中央値 の 3 つが同時に実側へ動くことを要求。128 枚で判定

### 自己修正
前ラウンドで指示した「剥離進行度を後期に寄せる」「可視クラス数 11〜18 フィルタ」は方向は
正しかったが**単独では不十分**で、フィルタは上縦隔バイアスに寄与する副作用もあった。
窓サイズ単独の調整では実データ分布に到達できないことが実測で判明した。


## 2026-09-09 Astra Round 3 実装・Mac検証

Claudeの2acdaf2を取り込み、心膜proxyのIPV保護減算、実剥離開口への注視・距離結合、下縦隔へ到達できるポート条件を実装。合成pilotからクラス数priorの提案分布補正を追加。各変更は設定で制御しprovisional表示を維持。

最終128枚（公開アトラスのみ、256×144、Blender4.5.9 CPU2並列）は452.9秒。出現率誤差合計633.74pp、胸膜平均13.37%、可視クラス中央値15、背景中央値26.49%。基準786/8/14/6.7から目標への距離が4指標とも改善し通過。IPV出現35.16%、右/左下肺靭帯34.38/41.41%。21テストと全128枚の出力監査通過。IPVの平均面積不足・器具/血液不足・SVC等の過剰は残る。

詳細と不合格実験も docs/round3_20260909.md、全クラス集計は docs/round3_results_20260909.md。実画像・実マスクにはアクセスしていない。生成画像や個別姿勢はGitへ同期しない。2048枚バンクは未生成。Linux担当は同文書の128枚コマンドを再実行し、通過したcomparison.jsonを --calibration-report に指定してバンク生成。設定不一致・未完了バッチ・12/8/9未出現はゲートで止める。ステーション命名法は引き続き要確認、教師生成無効。
