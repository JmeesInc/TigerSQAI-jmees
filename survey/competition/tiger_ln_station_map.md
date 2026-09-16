# TIGER リンパ節ステーション分類 — 解剖学的定義まとめ

> 調査日: 2026-09-11
> 対象: Tiger SQ-AI Challenge (Synapse syn74209386, EndoVis 2026 @ MICCAI2026, 主催 NCT/TSO Dresden)
> Task3 の 14 ステーション: `6L, 6R, 7L, 7R, 8, 9, 10L, 10R, 11L, 11R, 12L, 12R, 13L, 13R`

## 0. 結論（TL;DR）

- **チャレンジの14ステーション番号 (6L/6R/7L/7R/8/9/10L/10R/11L/11R/12L/12R/13L/13R) は、TIGER study（Hagens et al., BMC Cancer 2019, "1st edition"）が定義した胸部リンパ節ステーション 6〜13 の番号体系とほぼ一致する。**出典: Hagens et al. 2019 Table 1（下記 §2 に全文引用）。
- ただし **重要な相違点**: 公表されている TIGER 1st edition（BMC Cancer 2019 / Diseases of the Esophagus 2021 の比較論文 / PMC11300550 の Dutch survey 論文のいずれも同一）では、**station 10・11・12（上部・中部・下部の傍食道リンパ節）は左右分割されていない単一ステーション**として定義されている（左右分割があるのは 6, 7, 13 のみ）。
  - 一方、本チャレンジのラベル（`data/README.md`, `reference/tigersqai_challenge/metrics/classes_stations.py`）では **10, 11, 12 も 10L/10R, 11L/11R, 12L/12R と左右分割**されている。
  - この 10L/10R 等の左右分割は、調査した文献（BMC Cancer 2019 原著, Diseases of the Esophagus 2021 の統一化提案論文, PMC11300550 の 2024 年 Dutch survey 論文）のいずれにも見当たらなかった。**TIGER-SQA サブスタディ（oaepublish.com/articles/ais.2024.47, 2024）は "19 items to rate, representing all stations of the TIGER lymph node classification" と述べるのみで、個別の解剖学的境界やL/R分割の詳細は論文中に記載されていない**（"a detailed description of anatomical boundaries for each station is essential" と今後の課題として述べているのみ）。
  - → **本チャレンジ独自（おそらく TIGER-SQA の胸腔鏡ビデオ評価ツール、または本チャレンジのアノテーション設計）による拡張とみられる**。胸腔鏡下（通常右開胸アプローチ）で食道を周囲組織ごと剥離する際、傍食道リンパ節の脂肪組織を画面上で「食道の左側」「食道の右側」に視覚的に分けてラベル付けするのは自然な設計であり、10L=食道左側の上部傍食道組織、10R=食道右側の上部傍食道組織（11, 12も同様の頭尾レベルでの左右分割）と解釈するのが最も妥当な**推測（要確認）**である。
  - station 8（aortopulmonary window）と station 9（subcarinal）は TIGER・チャレンジともに左右分割なし（解剖学的に正中〜片側限局のため妥当）。
- 従って本ドキュメントは **(a) 公表されている TIGER 1st edition の正式な解剖学的境界定義をそのまま引用**し、**(b) 10/11/12 の L/R 分割については「上/中/下の頭尾レベルを踏襲しつつ、食道を挟んで左右どちらの組織か」という推測を明示的にガイドとして付す**、という二段構成にする。

---

## 1. 情報源

1. **Hagens ER, et al. "Distribution of lymph node metastases in esophageal carcinoma [TIGER study]: study protocol of a multinational observational study." BMC Cancer. 2019;19:662.**
   PMC: https://pmc.ncbi.nlm.nih.gov/articles/PMC6610993/ / DOI: https://doi.org/10.1186/s12885-019-5761-7
   → **TIGER classification 1st edition の原論文。Table 1 に全19ステーション（頸部1-5, 胸部6-13, 腹部14-19）の解剖学的定義が明記されている。** JES（Japanese Esophageal Society）11th edition と AJCC/UICC 8th edition を統合・簡略化したもの。
2. **van der Wilk BJ, et al.（Dutch esophageal surgeons survey）"Extent and Boundaries of Lymph Node Stations During Minimally Invasive Esophagectomy: A Survey Among Dutch Esophageal Surgeons."**
   PMC: https://pmc.ncbi.nlm.nih.gov/articles/PMC11300550/
   → TIGER定義を基準に外科医間の境界認識のばらつきを調査した論文。Table 4/5 に TIGER 定義（BMC Cancer 2019 と同一文言）と外科医の実践上の境界（隣接構造の言及）が対比表になっている。**「dissecting しているときに何が隣接して見えるか」の実地データとして§3で利用**。
3. **"A proposal for uniformity in classification of lymph node stations in esophageal cancer."** Diseases of the Esophagus. 2021;34(10):doab009.
   PMC: https://www.ncbi.nlm.nih.gov/pmc/articles/PMC8503476/ / PubMed: https://pubmed.ncbi.nlm.nih.gov/33884407/
   → TIGER・JES・AJCC の対応表。10/11/12 が左右分割されていないことを確認する目的で参照。
4. **"Assessment of the extent of lymphadenectomy in esophageal cancer surgery in the observational TIGER study: [TIGER-SQA] study protocol."** Art Int Surg. 2024.
   https://www.oaepublish.com/articles/ais.2024.47
   → TIGER-SQA（Surgical Quality Assessment、ビデオ/写真ベースの郭清範囲評価ツール）の protocol。**このチャレンジの Task3（ステーション可視性ラベル）と直接連続する先行研究**とみられるが、個別ステーションの境界定義は本文に含まれず「19項目」と述べるのみ。
5. 本リポジトリのデータ仕様: `data/README.md`, `data/labelmap.csv`, `reference/tigersqai_challenge/metrics/classes_stations.py`（チャレンジ自身のラベル定義。解剖学的境界の文章記述はなし＝上記文献で補完する必要がある）。
6. 参考: Diseases of the Esophagus 2024 "History and evidence for state of the art of lymphadenectomy in esophageal cancer surgery" https://academic.oup.com/dote/article/37/4/doad065/7458231 （TIGER 2nd edition の言及なし。BMC Cancer 2019 のみ引用）。

**TIGER 2nd edition の存在は確認できなかった**（Web検索・上記文献いずれにも "2nd edition" の記載なし）。したがって 10L/10R/11L/11R/12L/12R の左右分割は公表文献に根拠がなく、§0 で述べた通り推測扱いとする。

---

## 2. 解剖学的定義（TIGER 1st edition, Hagens et al. 2019, Table 1 全文引用）

> 出典: https://pmc.ncbi.nlm.nih.gov/articles/PMC6610993/ Table 1 "Classification of lymph node stations"
> ステーション 1–5 = 頸部（JES 11th edition 準拠）、6–13 = 胸部（JES 11th + AJCC 8th 統合）、14–19 = 腹部（JES 11th + AJCC 8th 統合）。
> **3-field リンパ節郭清 = station 1–19 を切除。2-field リンパ節郭清 = station 6–19（頸部を除く）を切除。**

### 頸部（1〜5）— 参考・チャレンジ対象外

| # | 名称 | 定義（原文抜粋） |
|---|------|------------------|
| 1 | Superficial cervical lymph nodes | 外頸静脈・前頸静脈に沿う浅頸筋膜下のリンパ節。顎下腺・耳下腺周囲、副神経に沿うもの等を含む。 |
| 2 | Cervical paraesophageal lymph nodes | 頸部食道周囲のリンパ節。**反回神経に沿うもの、頸部気管傍リンパ節を含む**。外側境界は頸動脈鞘内側縁。 |
| 3 | Deep cervical lymph nodes | 内頸静脈・総頸動脈周囲。顎二腹筋下縁〜輪状軟骨下縁の範囲。 |
| 4 | Peripharyngeal lymph nodes | 頸動脈鞘内側、顎二腹筋下縁〜輪状軟骨下縁。後咽頭・側咽頭リンパ節を含む。 |
| 5 | Supraclavicular lymph nodes | 鎖骨上窩。輪状軟骨下縁〜鎖骨。内側境界は頸動脈鞘内側縁。 |

### 胸部（6〜13）— **本チャレンジ Task3 の対象**

| # | 名称 | 側 | 定義（原文引用） |
|---|------|----|----|
| **6** | Upper paratracheal lymph nodes (right/left) | R/L | **Right**: "Lymph nodes located around the upper thoracic esophagus posterior to **the right vagal nerve**. Lymph nodes located along the anterior and lateral wall of **the thoracic trachea** until the level of the right vagal nerve. Lymph nodes located along **the right recurrent laryngeal nerve** in the mediastinum. The superior boundary is drawn from the cephalic border of **the right subclavian artery** to the suprasternal notch." **Left**: "Lymph nodes located around the upper thoracic esophagus. Lymph nodes located along the anterior and lateral wall of the thoracic trachea until the upper margin of **the aortic arch**. Lymph nodes located along **the left recurrent laryngeal nerve** in the mediastinum. The superior boundary is drawn from the cephalic border of **the left subclavian artery** to the suprasternal notch." |
| **7** | Lower paratracheal lymph nodes (right/left) | R/L | **Right**: "Lymph nodes located in the tracheobronchial angle and located along the anterior and lateral wall of the thoracic trachea. The superior boundary is **the vagal nerve**, the ventral boundary **the superior caval vein**." **Left**: "...Lymph nodes located along **the azygos vein arch** and **the right bronchial artery** are included. Lymph nodes along the proximal part of **the left recurrent laryngeal nerve** along the aortic arch are also included. The superior boundary is the inferior wall of **the aortic arch**..." |
| **8** | Aortopulmonary window lymph nodes | 正中/左寄り（分割なし） | "Subaortic and para-aortic nodes lateral to the ligamentum arteriosum. Superior boundary is the lower margin of **the aortic arch**. Ventral boundary is **the pulmonary artery**, distal boundary **the left main bronchus**." |
| **9** | Subcarinal lymph nodes | 正中（分割なし） | "Lymph nodes located caudal to **the carina of the trachea**. The lateral boundaries are the extended line of both lateral margins of the trachea." |
| **10** | Upper mediastinal paraesophageal lymph nodes | 分割なし（チャレンジは 10L/10R に分割 — §0参照） | "Dissection of the lymph nodes located around the **upper thoracic esophagus**. From **the thoracic aperture** until **the trachea bifurcation**." |
| **11** | Middle mediastinal paraesophageal lymph nodes | 分割なし（チャレンジは 11L/11R — §0参照） | "Lymph nodes located around the **middle thoracic esophagus**. From **the trachea bifurcation** to **the caudal margin of the inferior pulmonary vein**." |
| **12** | Lower mediastinal paraesophageal lymph nodes | 分割なし（チャレンジは 12L/12R — §0参照） | "Lymph nodes located around the **lower thoracic esophagus**. From **the caudal margin of the inferior pulmonary vein** to **the esophagogastric junction**." |
| **13** | Pulmonary ligament lymph nodes (right/left) | R/L | **Right**: "Dissection of the lymph nodes within **the right inferior pulmonary ligament**." **Left**: "Dissection of the lymph nodes within **the left inferior pulmonary ligament**." |

### 腹部（14〜19）— 参考・チャレンジ対象外

| # | 名称 | 定義（要約） |
|---|------|------|
| 14 | Paracardial lymph nodes (right/left) | 胃食道接合部直近。右=左胃動脈上行枝第1枝に沿うもの、左=左横隔動脈食道心臓枝に沿うもの。 |
| 15 | Left gastric artery lymph nodes | 左胃動脈に沿うリンパ節。 |
| 16 | Celiac trunk lymph nodes | 腹腔動脈周囲。背側境界=大動脈、腹側境界=左胃動脈起始部。 |
| 17 | Splenic artery and splenic hilum lymph nodes | 脾動脈起始部〜膵尾部沿い、脾門部。 |
| 18 | Common hepatic artery lymph nodes | 総肝動脈起始部〜胃十二指腸動脈・固有肝動脈分岐部まで。 |
| 19 | Hepatoduodenal ligament lymph nodes | 固有肝動脈・門脈に沿う（肝管合流部〜膵上縁）。 |

---

## 3. 側性・頭尾方向の一覧（チャレンジの14ステーション）

| ステーション | 側 | 頭尾レベル（TIGER定義の範囲） | 備考 |
|---|---|---|---|
| 6L | 左 | 上縦隔（胸郭入口〜大動脈弓上縁） | 気管前面・左反回神経沿い |
| 6R | 右 | 上縦隔（胸郭入口〜右迷走神経レベル） | 気管前面・右反回神経沿い |
| 7L | 左 | 上〜中縦隔（気管分岐部付近、大動脈弓下縁が上限） | 奇静脈弓・右気管支動脈・左反回神経近位部を含む（TIGER定義上、解剖学的に「左」側だが右気管支動脈等も含む点に注意） |
| 7R | 右 | 上〜中縦隔（気管気管支角） | 上大静脈が腹側境界 |
| 8 | 正中〜左（分割なし） | 大動脈弓下縁〜肺動脈・左主気管支 | Aortopulmonary window（動脈管索外側） |
| 9 | 正中（分割なし） | 気管分岐部（隆起）の尾側 | 気管両側縁の延長線が外側境界 |
| 10L/10R | 左/右（**チャレンジ独自分割、推測**） | 上部食道周囲：胸郭入口〜気管分岐部 | TIGER原義は左右非分割の単一ステーション |
| 11L/11R | 左/右（**チャレンジ独自分割、推測**） | 中部食道周囲：気管分岐部〜下肺静脈尾側縁 | 同上 |
| 12L/12R | 左/右（**チャレンジ独自分割、推測**） | 下部食道周囲：下肺静脈尾側縁〜食道胃接合部 | 同上 |
| 13L | 左 | 下縦隔・下肺靱帯内 | 左下肺靱帯 |
| 13R | 右 | 下縦隔・下肺靱帯内 | 右下肺靱帯 |

頭尾方向の全体順序（上→下）: **6 (上縦隔) → 7 (上〜中縦隔) → 8/9 (中縦隔、大血管・気管分岐部周囲) → 10 (上部食道傍) → 11 (中部食道傍) → 12 (下部食道傍) → 13 (下肺靱帯、最尾側)**。

---

## 4. チャレンジのラベルセットとの対応（郭清時に隣接・視認されやすい構造）

以下は §2 の TIGER 定義本文中に明記された構造、および PMC11300550 の外科医調査（Table 4, 実地の境界認識）に基づく。**「TIGER定義に明記」と「一般的な胸部解剖学から見て隣接するため画面に映り込みやすい（推測）」を区別して記載する。**

| ステーション | TIGER定義に明記された構造 | 隣接して視認されやすい構造（解剖学的推測） |
|---|---|---|
| **6L** | Aorta（大動脈弓上縁が上限）, Left recurrent laryngeal nerve, Left subclavian artery, Trachea, Esophagus | Fatty tissue, Pleura |
| **6R** | Right vagal nerve, Right recurrent laryngeal nerve, Right subclavian artery, Trachea, Esophagus | Superior caval vein（尾側に隣接）, Fatty tissue |
| **7L** | Trachea, Aorta（大動脈弓下縁）, Azygos vein（奇静脈弓）, Right bronchial artery, Left recurrent laryngeal nerve, Left main bronchus（近傍） | Pulmonary artery, Esophagus |
| **7R** | Trachea, Right vagal nerve（上限）, Superior caval vein（腹側境界） | Azygos vein, Right main bronchus |
| **8** | Aorta（大動脈弓）, Pulmonary artery（腹側境界）, Left main bronchus（遠位境界） | Left recurrent laryngeal nerve, Fatty tissue |
| **9** | Trachea（隆起）, Right main bronchus, Left main bronchus（外側境界＝気管両縁延長） | Esophagus（背側）, Pericardium（腹側; PMC11300550 surgeon survey 7/11人が回答）, Pulmonary artery, Thoracic duct（後縦隔で近接） |
| **10L/10R** | Esophagus, Fatty tissue esophagus（"upper thoracic esophagus" 周囲）, Trachea（気管分岐部が下限） | Pericardium, Recurrent laryngeal nerve（R/L）, Right/Left inferior pulmonary ligament（尾側に連続）, Azygos vein（右側）, Aorta（左側）, Lung/Pleura |
| **11L/11R** | Esophagus（中部食道）, Inferior pulmonary vein（下限） | Azygos vein（右側に沿走; PMC11300550 では奇静脈が境界として言及）, Superior caval vein, Pericardium, Aorta, Lung/Pleura, Trachea（上限が気管分岐部） |
| **12L/12R** | Esophagus（下部食道）, Inferior pulmonary vein（上限） | Pericardium, Aorta, Lung/Pleura, Azygos vein, Gastric conduit（再建後の画像であれば） |
| **13L** | Left inferior pulmonary ligament（定義そのもの） | Lung, Pleura, Esophagus, Inferior pulmonary vein（上限） |
| **13R** | Right inferior pulmonary ligament（定義そのもの） | Lung, Pleura, Esophagus, Inferior pulmonary vein（上限） |

補足:
- **Right/Left subclavian artery** は 6R/6L の「上限（頭側境界）」としてのみ明記。画面上では郭清の最頭側で一瞬映る程度と推測される。
- **Thoracic duct（胸管）** は TIGER の 6-13 定義文には明記されていないが、解剖学的には大動脈と奇静脈の間（後縦隔、おおむね T4-5より下で右→左に交差）を走行するため、**station 8/9/11 付近の郭清で視認されうる**（推測）。
- **Gastric conduit（管状胃）・Omentum（大網）** は胸部リンパ節ステーション定義そのものには現れない。再建（胃管挙上）後の画像、あるいは経裂孔的操作を伴う 12L/12R・下部食道操作時の画面に映り込む可能性がある（推測、TIGER定義には根拠なし）。
- **Right bronchial artery** は 7L の定義に明記される数少ない「小血管」。TigerSQAI のクラス重み（`labelmap.csv`）でも weight=3（重要クラス）に指定されており、7L 郭清時の再現性がスコアに直結する可能性が高い。
- **Pulmonary artery** は station 8 の腹側境界として明記。station 8/9 の画像で視認されやすい。

---

## 5. 注意点・限界（要確認事項）

1. **10L/10R・11L/11R・12L/12R の左右分割の正確な境界線定義（食道正中でどちらを左/右とするか、前後方向なのか真の左右方向なのか）は、調査した公表文献のいずれにも記載がなかった。** 本チャレンジ独自のアノテーション規約が Synapse wiki 内の別ページ（wiki 639935 の Docker instructions 以外、例えばアノテーションガイドライン）に存在する可能性があるため、**Synapse wiki 全体（特に Task3 / データアノテーション方法論のページ）を未読の場合は確認を推奨**。
2. TIGER-SQA（oaepublish 2024）論文は本チャレンジのTask3設計と直接連続する先行研究と強く推測されるが、**論文内に個別ステーションの解剖学的境界表は含まれていなかった**（今後の課題として言及されるのみ）。TIGER-SQA の補足資料（Supplementary Material）に境界図が含まれる可能性があるが、本調査では未取得。
3. "TIGER 2nd edition" は検索・文献調査で存在を確認できなかった。もし存在する場合、10/11/12 のL/R分割はそちらで正式に定義されている可能性がある（未確認）。
4. 本チャレンジの `data/README.md` / `reference/tigersqai_challenge/metrics/classes_stations.py` 自体にはステーションの解剖学的境界の文章記述は一切ない（ラベル名と色のみ）。

---

## 出典一覧

- [Hagens ER, et al. "Distribution of lymph node metastases in esophageal carcinoma [TIGER study]: study protocol of a multinational observational study." BMC Cancer 19, 662 (2019).](https://pmc.ncbi.nlm.nih.gov/articles/PMC6610993/) / [DOI](https://doi.org/10.1186/s12885-019-5761-7) / [PubMed](https://pubmed.ncbi.nlm.nih.gov/31272485/)
- [van der Wilk BJ, et al. "Extent and Boundaries of Lymph Node Stations During Minimally Invasive Esophagectomy: A Survey Among Dutch Esophageal Surgeons."](https://pmc.ncbi.nlm.nih.gov/articles/PMC11300550/)
- ["A proposal for uniformity in classification of lymph node stations in esophageal cancer." Diseases of the Esophagus 34(10), doab009 (2021).](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC8503476/) / [PubMed](https://pubmed.ncbi.nlm.nih.gov/33884407/) / [Journal page](https://academic.oup.com/dote/article/34/10/doab009/6174327)
- ["Assessment of the extent of lymphadenectomy in esophageal cancer surgery in the observational TIGER study: [TIGER-SQA] study protocol." Art Int Surg (2024).](https://www.oaepublish.com/articles/ais.2024.47)
- ["History and evidence for state of the art of lymphadenectomy in esophageal cancer surgery." Diseases of the Esophagus 37(4), doad065 (2024).](https://academic.oup.com/dote/article/37/4/doad065/7458231)
- TIGER classification 1st edition 図解（未アクセス、403）: [ResearchGate figure](https://www.researchgate.net/figure/TIGER-classification-1-st-edition-station-numbers-and-naming-of-regional-lymph-nodes_fig3_351070930)
- 本チャレンジ内部資料: `data/README.md`, `data/labelmap.csv`, `reference/tigersqai_challenge/metrics/classes_stations.py`（社内パス、参加者限定データのため非公開URL）
