# anatomy_graph — 解剖グラフ (隣接・排他・個数・囲み) の構築とルール応用

2026-09-11 作成。GT マスクの統計 + expS01 の 3D アトラスから「解剖学的にあり得るクラス関係」を
グラフ (行列) にし、(1) Task1/2 のルール loss (expA22)、(2) 後処理、(3) Task3 のリンパ節→station 割当てに使う。

## ファイル

| ファイル | 役割 |
|---|---|
| `build_graph.py` | GT ラベル (576×1024 キャッシュ) から統計を取る。`--task fine|coarse --fold N` で **val 画像を除外** (リーク防止)。出力 `out/{task}[_foldN]/`: `adj_img.npy` (両クラスが接した画像数), `cooc.npy` (共起画像数), `presence.npy`, `adj_pairs.npy`/`nb_dist.npy` (境界画素対・相手分布), `ncomp.npy` (画像×クラスの成分数), `enclosure.npy` (成分外周の 90% 以上が単一クラス = 囲み), `graph.json` (人が読む要約: 禁止隣接ペア / 排他ペア / 期待成分数 / 囲み) |
| `build_graph_3d.py` | expS01 アトラス (`outputs/tier0_options2/work/atlas.npz`, RAS mm) からクラス間の表面最短距離・重心・リンパ節群ごとの近傍構造を計算 → `out/atlas3d/` |
| `rules.py` | 統計 → ルール行列 `out/rules_{task}[_foldN].npz`: `W_adj` (隣接ペナルティ [0,1]), `E_excl` (排他 {0,1}), `K_max` (成分数 p90)。**覆い被さる/偏在する組織は除外** (`IGNORE_CLASSES`: fine = 背景/Instrument/Other/Fatty tissue esophagus/Pleura/Fatty tissue/Pool of blood/Resection area, coarse = 背景/Pleura/Non-anatomical Other/Fatty Tissue/Anatomical Other) |
| `violation_report.py` | GT と OOF 予測 (ens5) のルール違反率を測る診断 |
| `rule_postproc.py` | ルールによる後処理 (excl / adj / count) を OOF 確率で採点 |
| `assign_ln_stations.py` | Task3: LN 成分ごとに近傍構造から station を割り当てる (visible 集合内で argmax)。出力 `out/task3/ln_station_assignments.csv`, `station_maps/`, `review_contact_sheet.png` |
| `ln_context_cv.py` | 上記の妥当性検証 (LN 近傍から station が当たるかの GroupKFold 実験) |

## ルールの定義

- **隣接ペナルティ** `p_adj(a,b) = P(接する | 両方が写っている)` を Laplace 平滑化し、`p ≤ 0.02 → 1.0`, `p ≥ 0.2 → 0.0`, 間は対数線形。共起 10 枚未満のペアは証拠不足として 0 (罰しない)。
- **排他** = 共起ゼロ (どちらも 5 枚以上出現)。例: 右反回神経 ↔ 左反回神経、右鎖骨下動脈 ↔ 下肺靱帯/心膜/下肺静脈 (上縦隔 vs 下縦隔)。
- **3D アトラス override**: 2D では未観測でも 3D で表面距離 ≤ 5 mm かつ共起 < 30 枚のペアはペナルティを半減 (fine で 1〜3 ペア)。
- **個数** (`K_max`) は微分可能にしにくいので loss には入れず後処理のみ。

## 主な統計 (fine, 全 523 枚, `out/fine/graph.json`)

- 禁止隣接 (共起あり・接触ゼロ) の代表: 気管–下肺静脈 (共起 153)、右迷走神経–大動脈 (151)、左下肺靱帯–奇静脈 (88)、左主気管支–右下肺靱帯 (79)、下肺静脈–胸管 (63)
- 排他 22 ペア、期待成分数: IPL/IPV/右鎖骨下動脈/右反回神経は 1 個がほぼ全て (p_single ≥ 0.9)、リンパ節は中央値 2 / p90 4
- 囲み: 「Lymph node が Fatty tissue esophagus に囲まれる」33 成分が解剖的に意味のある最上位 (他は Other/器具絡み)

## 3D アトラス / 合成レンダの使えるところ・使えないところ

- **3D メッシュ距離** (`out/atlas3d/graph3d.json`): 収録 14 クラスの視点非依存な「遠いペア」(19 ペア, > 20 mm) は 2D の禁止ペアと整合する。リンパ節群 (傍気管・気管気管支・傍食道・奇静脈弓) の近傍構造も取れる。**IPL/心膜/反回神経/胸管など 16 クラスは未収録**。
- **合成レンダ 2000 枚の 2D 隣接統計** (`out/fine_synth2k`) は実 GT と **相関 0.21** しかなく (`out/synth_vs_real.txt`)、「気管–右迷走神経が合成では接しない (実 0.89)」など矛盾が多い。→ **合成 2D 統計はルールの情報源には使わない** (脂肪シェル/剥離モデルが実際の露出と違うため)。

## 結果 (2026-09-11 時点)

- ens5 OOF の違反率: fine adj_rate **0.0152 (GT 0.0020)**、排他違反 0。違反の大半は **気管–心膜 (5.2 万画素対) と気管–下肺静脈 (1.3 万)** = 気管の誤検出。coarse は GT と同水準 (0.0004) で伸びしろ無し。
- ルール後処理 (OOF, `out/postproc/`): coarse は excl/adj とも ±0.000、count は **−0.007 (有害)**。fine は `rule_postproc_fine.log` 参照。→ **後処理としての効果は無い**。効くなら学習中の正則化 (expA22) として。
- Task3 の LN 成分→station 割当て: 手書きルール表の QA 一致率 (最大成分 = ファイル名 station、multi-station フレーム 331 枚) **0.266 vs チャンス 0.244**。ロジスティック回帰 (GroupKFold) でも **0.332**、しかも情報源はフレーム全体の構造在庫 (0.344) で **成分の局所近傍だけでは 0.269 ≒ チャンス**。→ 局所近傍からの station 同定は成立しない (詳細は daily_reports/20260911.md)。
