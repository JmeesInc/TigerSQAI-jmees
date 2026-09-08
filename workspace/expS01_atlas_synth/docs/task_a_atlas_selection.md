> 2026-09-09追記: fine_id/背景0はユーザー確認済み。心膜11・左右下肺靱帯8/9は暫定proxy実装によりclass_mappingの(c)へ変更（元の個別形状が同定された意味ではない）。class 22は学習528枚で出現0、今回補充対象外。以下の初回棚卸しは履歴として保持する。新実装は task_bc_calibration_stage1.md を参照。

タスクA：公開解剖アトラスの選定とクラス対応（2026-09-08調査）

推奨する最小構成は、**Z-Anatomyの確認済み胸部形状＋BodyParts3DのFMA対応表**です。患者変動は次段階で**TotalSegmentatorの公開CT付属マスク**から導入します。配布済み推論重みは、全学習データと初期化元の公開性を確認するまで使いません。

重要な結論は「この3候補で重み3の全構造が揃うとは、現時点で確認できない」です。反回神経を迷走神経へまとめたり、一般的な気管支動脈を右気管支動脈として扱ったりすると、最重要クラスに誤教師を作ります。

**調査範囲**：公開Web資料、BodyParts3D公式4.0配布索引、および作業フォルダに既存の公開アトラス `Z-Anatomy/Startup.blend` のみ。チャレンジ画像・マスク・CSVは読んでいません。ローカルファイルが現行GitHub配布物と同一かは未確認です。

アトラス比較表

| 候補 | 入手方法・現在の疎通 | 形式とメッシュ化 | ライセンス・商用可否・再配布条件 | 含まれる構造と30クラスへの適合 | 懸念点 |
|---|---|---|---|---|---|
| Z-Anatomy | [公式モデルリポジトリ](https://github.com/Z-Anatomy/Models-of-human-anatomy)の `Z-Anatomy.zip`。リポジトリ閲覧可。ZIP本体の今回の取得・ハッシュ照合は未実施。[公式トップ](https://www.z-anatomy.com/)は応答するがWeb抽出内容が少なく、ダウンロード導線は要疎通確認 | BlenderテンプレートZIP内の `.blend`。既存ファイルにはMESHとCURVEが混在。Curveを評価済みメッシュに変換して使用 | 全体表記はCC BY-SA 4.0。**商用可は部品ごとの条件付き**。READMEに耳・腎臓などNC素材の記載があり、全体を一律商用可とはできない。通常のBY-SA素材の再配布には出典・作者・ライセンス・変更表示とSAが必要 | 気管、左右主気管支、食道、大血管、迷走神経Curve、胸部リンパ節群。胸膜・大網も実体あり。術野ベース形状として最短 | 独立反回神経・胸管・心膜・左右肺靱帯・右気管支動脈を調査版で未同定。教育用正常解剖で、MIEの虚脱・剥離状態ではない。FMAが保持されている保証なし。Blender 4.x互換は要実機確認 |
| BodyParts3D | [LSDB公式配布ページ](https://dbarchive.biosciencedbc.jp/en/bodyparts3d/download.html)。ページとIS-A/PART-OF索引を取得できた。4.0のOBJ ZIPリンクあり。本体の取得・展開は今回未実施。Anatomographyの4.3系は4.0と区別して別途確認 | Wavefront OBJ ZIP＋TSV。FMA概念→representation→element fileを解決する。FMAとOBJは必ずしも1対1ではない | **現行LSDBアーカイブはCC BY 4.0、商用可**。2025-02-27更新。[現行許諾](https://dbarchive.biosciencedbc.jp/en/bodyparts3d/lic.html)と[README](https://dbarchive.biosciencedbc.jp/data/bodyparts3d/LATEST/README_e.html)の両方で確認。再配布時は指定帰属・ライセンス・変更表示。SA不要。一方、[旧情報サイト](https://lifesciencedb.jp/bp3d/info_en/index.html)はCC BY-SA 2.1 JPのままなので、4.3等へ新許諾を無条件に拡張しない | FMA付きの気管・食道・主気管支・大血管・肺などを確認。一般的な気管支動脈は索引にある。IDによる再現性の高い取り込みに向く | 4.0の両索引で迷走神経・反回神経・胸管・リンパ節を未同定。これは将来版や全モデルの不存在証明ではない。99%ポリゴン削減版は細構造に不向き。複合概念が不完全な場合、メッシュの和が解剖学的全体とは限らない |
| TotalSegmentator | [公式コード・モデル一覧](https://github.com/wasserth/TotalSegmentator)、[公開CTデータv2.0.1](https://zenodo.org/records/10047292)。GitHub・Zenodoメタデータへ疎通確認。CT ZIP約23.6 GBの本体ダウンロードは未実施 | CT/DICOMまたはNIfTIを処理しNIfTIマスクへ。今回は公開データ付属マスク→表面抽出→OBJ等→Blenderを推奨。推論なしで形状取得可能 | コード・公開タスクはApache-2.0、商用可、再配布時LICENSE・該当NOTICE・変更表示等。制限付きサブタスクは別条件。公開CT v2.0.1は**CC BY 4.0、商用可**、帰属等を付けて再配布可能。コード・重み・CTの許諾を別管理 | `total`は気管、食道、大動脈、SVC、左右鎖骨下動脈、左右5肺葉、肺静脈など。患者変動の供給に適合。左右主気管支や下肺静脈を分離する専用ラベルとは限らない | READMEは学習データの「大部分」が公開という表現。全配布重みがユーザー規約を満たすとは確認できない。CT仰臥位・含気肺と右胸腔MIEの差が大きい。微細神経・胸管を供給する標準モデルではない |

TotalSegmentatorのラベル名は[公式クラス定義](https://raw.githubusercontent.com/wasserth/TotalSegmentator/master/totalsegmentator/map_to_binary.py)で確認したものです。現行コードと公開CTデータv2.0.1の内容を同一とは扱わず、実際に採用するデータ・リリースで照合します。`pulmonary_vein`をそのままInferior pulmonary veinに、`heart`をPericardiumに、`lung_airways`全体を左右主気管支に割り当てることはできません。現行 `trunk_cavities` に `pericardium` はありますが、心膜の膜面と一致するか、重みの学習由来を含め要確認です。

「CTでは絶対に取れない」は一括した断定としては採用しません。例えばJESもCTによるリンパ節評価を記載しています。ただし**通常のCTと今回の標準ラベルから対象の微細構造を安定して取得できるとは限らない**ため、アトラスを使う設計上の動機は妥当です。[JES公式分類論文](https://pmc.ncbi.nlm.nih.gov/articles/PMC11199297/)

重要構造の実体確認

ローカル `.blend` の保存形式は3.05、SHA-256は `9f08a17ea0115fed80b2a73ecdf0a1bc2ab2f6956f37c593ce23d513ea35afcd`。SDNAを読み取り、7,184 Objectを棚卸ししました。証拠は `assets/evidence/z_anatomy_inventory.json`。これは名前検索だけではなく、Meshの面数、Curveの非空スプラインリストまでの確認です。評価後形状・走行・可視性の確認ではありません。

| 対象 | Z-Anatomy調査ファイルの結果 | BodyParts3D公式4.0索引の結果 | 判定・次の確認 |
|---|---|---|---|
| 右・左迷走神経 | `Vagus nerve (X).r` / `.l`、ともにスプラインを持つ独立Curve | 今回の両索引では未同定 | 既存形状あり、ただし**独立ポリゴンメッシュではない**。Curve評価・変換後に胸部走行と径を確認。右のみfine 14 |
| 左右反回神経 | 独立した該当Object名を未同定 | 未同定 | (d)。迷走神経内の枝として含まれる可能性は未排除。分枝と走行をBlenderで確認してから分離可否を判断 |
| 胸管 | 独立した該当Object名を未同定 | 未同定 | (d)。コレクションや用語辞書の項目だけでは収載と数えない。別版の非空形状を確認 |
| 右気管支動脈 | 独立した該当Object名を未同定 | `FMA68109` bronchial artery、`FMA10704` variant bronchial arteryあり | (d)。実OBJと周囲血管を表示し、右側・起始・末梢を同定するまで汎用FMAを右へ割り当てない |
| リンパ節 | `Paratracheal thoracic nodes` 960面、`Inferior tracheobronchial nodes` 960面、`Juxta-oesophageal nodes` 960面、`Node of arch of azygos vein` 96面等 | lymph nodeに該当する索引項目を今回未同定 | (a)。実メッシュはあるが、節群Objectであって各1節の独立Objectとは限らない。Task3のステーション帰属は未確認 |
| 左右下肺靱帯 | 独立形状を未同定 | 未同定 | (d)。胸膜に統合されている可能性は要確認。肺靱帯を新規に推測生成しない |
| 心膜 | 心膜そのものを未同定。心膜周囲リンパ節はある | 未同定 | (d)。リンパ節名にpericardialを含むことは心膜存在の証拠ではない |
| 胸膜・大網 | `Pleura` 3,580面、`Greater omentum` 1,212面、`Lesser omentum` 70面 | この2索引では対応未同定 | 既存解剖あり。今回は術中加工を理由に(c)とする |

`.g` / `.j` 等の補助Objectには2頂点・0面のものがあり、これを「メッシュあり」と数えると誤ります。右神経の代わりに `Posterior nucleus of vagus nerve.r` を拾うことも禁止します。

FMAによる同定方針

BodyParts3Dの[FMA→representation表](https://dbarchive.biosciencedbc.jp/data/bodyparts3d/LATEST/isa_parts_list_e.txt)、[PART-OF表](https://dbarchive.biosciencedbc.jp/data/bodyparts3d/LATEST/partof_parts_list_e.txt)、[element対応表](https://dbarchive.biosciencedbc.jp/data/bodyparts3d/LATEST/isa_element_parts.txt)を使います。FMAは概念IDで、BPは表現ID、OBJはelement file IDという区別があります。実ファイル名を `FMAxxxx.obj` と推測して組み立てません。[公式データ仕様](https://dbarchive.biosciencedbc.jp/data/bodyparts3d/LATEST/README_e.html)

Z-AnatomyはBodyParts3D派生ですが、名前・分割・表現形式の変更があるためFMAの自動継承は保証できません。`TA2.csv`の存在もFMA保持の証明にはなりません。取り込み時に「アトラスSHA→Object名/実体→確認済みFMA」のsidecarを作り、その後の実行をFMA完全一致に限定します。名称パターンはこの初回照合作業の候補抽出用です。

`assets/class_mapping.yaml` の `match.fma_ids` は**公式BodyParts3D索引で実在を確認した概念ID**であり、Z-Anatomy ObjectにそのIDが既に保存されているとの主張ではありません。ID未検証の神経・リンパ節等は空配列です。`automatic_assignment_enabled: false` にして、未確認ID・空形状から教師を自動生成しない初期設定にしました。取り込み時の自動割当てを解除するには、アセット単位のsidecarと可視形状の確認が必要です。

たとえば気管はFMA7394、食道はFMA7131、左主気管支はFMA7396。右側は収載名が `right main bronchus proper` のFMA68418なので、コンペが中間気管支を含むか確認が必要です。肺の子孫を無制限に展開すると気管支・血管まで包含するため、親概念の一括取り込みは禁止します。採用した索引行と索引ハッシュを `assets/evidence/bodyparts3d_selected_concepts.json` に保存しました。

30クラスの分類

`fine_id` は依頼文の列挙順を1〜30とした**内部ID案**です。公式マスク値との一致は未確認、backgroundも未設定です。coarse 15クラスへの統合表は未提示なので推測していません。

| 分類 | fine IDとクラス | 今回の扱い |
|---|---|---|
| (a) アトラス直接利用：9 | 4 Right main bronchus、5 Left main bronchus、12 Inferior pulmonary vein、13 Right subclavian artery、14 Right vagal nerve、16 Azygos vein、19 Lymph node、21 Left subclavian artery、23 Pulmonary artery | Z-Anatomy既存Mesh/Curveを使用。非空の実体を確認済み、解剖範囲と変換後QAは残る |
| (b) CT患者変動優先：5 | 3 Trachea、6 Esophagus、15 Aorta、17 Superior caval vein、18 Lung | 公開付属マスクを優先。最小構成では既存アトラスも利用可能 |
| (c) 術中状態の生成・加工：9 | 1 Instrument、2 Other、7 Fatty tissue esophagus、10 Pleura、20 Fatty tissue、24 Pool of blood、25 Resection area、26 Gastric conduit、29 Omentum | 方針は次表。Pleura/Omentumは解剖学的実体であり「アトラス不存在」が理由ではない |
| (d) 現時点で対応不能：7 | 8 Right inferior pulmonary ligament、9 Left inferior pulmonary ligament、11 Pericardium、22 Right bronchial artery、27 Right recurrent laryngeal nerve、28 Left recurrent laryngeal nerve、30 Thoracic duct | 根拠ある個別形状の同定が未完了。将来の版・分割で解決する可能性は残す |

(a)と(b)は能力が排他的という意味ではなく、推奨する主供給経路で分けています。鎖骨下動脈は(a)ですがCTにも対応ラベルがあります。胸管の重みは、依頼文の30クラス一覧に従い**1**を維持しました。

| (c) クラス | タスクBでの方針 |
|---|---|
| Instrument | 寸法を設定化した軸・関節・把持部から最小器具を作る。器具の形状詳細・ロボット比率を設定で切り替える。外部CADは本タスクでは未選定 |
| Other | 主催者定義に該当する物体を明示リストで生成・割当て。未知解剖や未対応クラスの一括受け皿にはしない |
| Fatty tissue esophagus | 既存食道の周囲に被覆を付与し、露出・剥離の切り欠きを作る。一般脂肪との境界は設定化 |
| Pleura | 既存胸膜を薄膜化・切開・開窓・牽引して術中状態へ加工。壁側/臓側の対象範囲は主催者定義で固定 |
| Fatty tissue | 既存胸部の表面や間隙に沿う被覆を付与し、食道周囲脂肪との重複を禁止 |
| Pool of blood | 既存表面上に液体パッチを付与。輪郭・厚さ・材質を変動させ、血管内血液とは区別 |
| Resection area | 切除・剥離操作の履歴から対象領域とマスクを作る。焼灼色だけで判定しない |
| Gastric conduit | 公開胃形状を切り出し・細径化・変形して再建胃管を作る。走行・径・再建段階を設定化 |
| Omentum | 既存大網を切り出し・牽引・変形。胃管に付着する大網など主催者の対象範囲に合わせる |

CC BY-SA：学習だけの場合と合成データを配布する場合

以下は公開ライセンス条文に基づく運用方針です。重みが翻案物に当たるか等、個別の法的判断を断定するものではありません。[CC BY-SA 4.0条文](https://creativecommons.org/licenses/by-sa/4.0/legalcode.en)、[旧CC BY-SA 2.1 JP](https://creativecommons.org/licenses/by-sa/2.1/jp/)、[CC BY 4.0条文](https://creativecommons.org/licenses/by/4.0/legalcode.en)

| 利用 | 条件・推奨運用 |
|---|---|
| チーム内で生成して学習するだけ | 非共有の複製・改変だけから、SAによる公開義務が自動で発生するわけではない。出典・許諾・変更履歴は内部にも保存。学習自体に非商用限定はないBY-SA素材と、NC素材を区別する |
| 合成画像・派生メッシュを配布 | BY-SA形状のレンダリング・加工物は派生物として扱う保守的運用。帰属、元URL、ライセンスURL、変更内容を同梱し、適用可能なSA条件で配布する。追加の利用禁止やDRMで受領者の権利を制限しない |
| マスク・アノテーションも配布 | マスク自体の著作物性・データベース権は個別判断だが、一律に権利対象外とはしない。派生画像・マスクの組に部品由来を記録し、適切なBY-SA条件で扱う設計にする |
| 学習済み重みだけ配布・コンペへ提出 | 「学習したから重みも必ずSA」「重みなら絶対SA不要」の両方を断定しない。成果物の内容、ライセンス対象権利、主催者の提出・再配布条件を照合する |
| CC BY 4.0の現行BodyParts3Dを直接使用 | 帰属等の条件はあるがSA条件はない。Z-Anatomyによる改変版までCC BYだけに変更できるわけではない |

BY-SAは、別個に作ったコード全体や関係のない実データへ当然に適用されるものではありません。混合物を配る場合の対象範囲は確認が必要です。合成物だけの配布でも、コンペ規約の外部データ許容条件は別途満たす必要があります。

各配布物には、元アセットURL、取得日、SHA-256、作者、ライセンス本文またはURL、加工内容、生成seed、使用Object/FMAの一覧を記録します。BodyParts3Dの指定帰属文は現行許諾ページから採用します。Z-AnatomyではBodyParts3DとZ-Anatomyの両帰属を保ち、追加素材の記載を部品単位で確認します。NC記載のある腎臓・内耳等は今回の胸部セットに採用しませんが、これだけで全胸部部品の由来監査が完了したとは扱いません。

ステーション命名法の候補比較：すべて要確認

| 候補・参照版 | 6Lなら | 7Rなら | 13Lなら | 14ラベルとの整合性 |
|---|---|---|---|---|
| TIGER study統合マップ（2019） | 左上部気管傍。左反回神経周囲を含む上縦隔領域 | 右下部気管傍、右気管気管支角付近 | 左下肺靱帯内 | **最も整合的という仮説**。6/7/13は左右あり、8は大動脈肺動脈窓、9は気管分岐下で左右なし。10〜12は傍食道の高さ別区分だが、コンペのL/R分割規則は別途確認が必要 |
| IASLC肺癌ノードマップ | 正式にはstation 6（傍大動脈：上行大動脈・弓部前外側）。6Lは標準ラベルそのものではない | 正式にはstation 7（気管分岐下）。7Rへの分割は独自拡張が必要 | 左区域気管支周囲の区域リンパ節 | 10/11/12/13の左右は馴染むが、6Rや7L/Rが不整合。8/9は傍食道/肺靱帯で、左右を区別する部位として扱われる。8/9だけ無側性という事実だけでは候補を決められない |
| JES食道癌分類（11/12版） | 6Lという胸部コードはない。No.6を文字通り読めば幽門下の腹部リンパ節でL拡張の定義はない | 7Rという胸部コードはない。No.7は左胃動脈幹周囲でR拡張は未定義 | 13Lという胸部コードはない。胃癌系No.13を借用するなら膵頭後面であり、左胸部の領域にはならない | 胸部は105、106recR/L、106tb、107、108、109、110、112pulR/L等。百の位・接尾辞を削ったと推定して復元してはいけない |
| AJCC食道癌8版の食道専用マップ | 6Lは掲載された標準局所リンパ節コードとして定義されない。肺癌のstation 6を輸入しない | 正式には7＝気管分岐下。7Rは標準の分割ではない | 13Lは食道専用マップの標準局所コードとして定義されない | 2R/L、4R/L、7、8U/M/Lo、9R/L等で、今回の連番6〜13とは合わない。AJCCの版と臓器を固定して照合する |

参照：TIGER原著の[Fig.1・Table 1](https://repub.eur.nl/pub/117973/Repub-117973-OA.pdf)、[IASLC公式Staging Manual](https://www.iaslc.org/file/10870/download?token=ZdsZFnAS)、[JES 11版](https://pmc.ncbi.nlm.nih.gov/articles/PMC5222932/)、[JES 12版Part II](https://pmc.ncbi.nlm.nih.gov/articles/PMC11199314/)、[JES掲載の所属リンパ節名](https://www.jsco-cpg.jp/esophageal-cancer/guideline/)、[AJCC執筆者による8版解説](https://pmc.ncbi.nlm.nih.gov/articles/PMC5591443/)。IASLC PDFは検索側で内容確認できましたが、全PDF取得はWebツール容量上限で未完了。AJCC PMC全文は一部アクセスでCAPTCHAとなるため、主催者資料との照合時に原図の再確認も必要です。

TIGER原著の図表は公開資料ですが、その番号がこのコンペでも同じ定義で使われることを証明しません。特にTIGER本文のTable 1では10/11/12の定義は傍食道の高さ区分として示され、L/R分割の境界は十分には決まりません。単純な食道中心面で二分する処理は保留します。

確認手順

1. 主催者のannotation guideline、ステーション図、データ説明、評価コードの列名一覧について、版・日付・出典をローカルに記録する。チャレンジ実データを共有する必要はない。
2. 最初に識別力の高い **8・9・13L** を見る。「8＝大動脈肺動脈窓、9＝分岐下、13L＝左肺靱帯」であればTIGER統合マップ仮説と整合。「8＝傍食道、9＝肺靱帯、13L＝区域気管支周囲」であればIASLC系が近い。
3. 6Lに左反回神経周囲、7Rに右下部気管傍が含まれるか確認する。大動脈弓・鎖骨下動脈・迷走神経・SVC等の上下前後境界を図と文章で突き合わせる。画像上の左右ではなく患者解剖学的左右を使う。
4. 10L/R・11L/R・12L/Rの左右分割面と高さ境界を確認する。TIGER仮説では上/中/下傍食道だが、L/Rの意味や切除側の意味は別確認。8/9が無側性である理由も確認する。
5. visibilityが「節そのものが見える」「郭清領域が露出」「境界ランドマークが見える」のどれか確認する。Task1のLymph nodeピクセルの有無からTask3ラベルを作らない。1症例14枚でも各画像をone-hotにはしない。
6. 曖昧な点は、主催者へ命名法名・版・14ラベルの定義・Task3陽性基準を文面で照会する。照会の送信はユーザーが行う。
7. 確認後に `station_nomenclature` と版、定義出典、境界ランドマーク、遮蔽時の判定を別設定へ固定する。確認前はTask3の意味ラベル生成を無効のままにする。

タスクBに進む前提・未確定事項

| 項目 | 状態 | 確認方法・Bへの影響 |
|---|---|---|
| 右胸腔、腹臥位/半腹臥位、30度鏡等 | ユーザー指定を採用 | pose・camera・port・鏡角度を設定化。左側構造が存在することと、右胸腔から可視であることを分ける |
| 独立反回神経、胸管、肺靱帯、心膜、右気管支動脈 | 未確定 | 調査版で未同定の7クラスは保留。既存アトラスの分割で解決できるか監修付きで確認。欠落を背景やOtherにしない |
| Z-Anatomy変換互換 | 未確定 | 手元のBlender 5.2.1はファイル読み込み前後にクラッシュし、評価済み形状は検証できなかった。Linux Blender 4.xでautoexec無効・アドオンなし読み込み、Curve→Mesh、法線・面数・単位・変換行列を確認 |
| FMA→Z-Anatomy Object | 概念IDのみ一部確定 | カスタムプロパティ内のFMAは今回未検査。SHAを固定して部品ごとにsidecarを確認。IDとメッシュ名を推測で結ばない |
| 公開アセットのライセンス | 配布元の表記は確認 | Z-Anatomy胸部Objectの追加素材由来を確認。BodyParts3Dは採用版の許諾を保存。配布ZIP内表記との矛盾はDBCLSへ照会 |
| TotalSegmentator重み | 使用無効 | 全学習データ・追加学習データ・初期化元の公開性を重み単位で確認できるまで使わない。まず公開付属マスクでBを進められる |
| 公開CT付属マスクの由来 | 公開・CC BY 4.0を確認 | 注釈生成にAI補助がある場合の許容範囲がコンペ規約に含まれるか確認。配布マスクの利用と非公開データ学習済み重みの利用を区別 |
| 外部アトラス原型の来歴 | 公開成果物の許諾は確認 | 元の制作参考資料・CTまで公開が要求される規約なら別途来歴確認が必要。公開アトラスであることだけから規約適合を断定しない |
| 公式fine ID、背景値、ignore値、coarse統合 | 未確定 | 主催者仕様とローカルで照合。ignore値を255等に決め打ちしない |
| Pleura、Other、Resection area、脂肪境界、肺動脈範囲 | 未確定 | annotation guidelineで対象範囲と重なりの優先規則を確定 |
| 14ステーション定義とvisibility | 要確認 | 上記の手順。候補比較をそのまま教師生成へ流用しない |
| 形状・材質・カメラの分布 | 未確定 | 分布・寸法・seed・解像度を設定/CLIに出す。ユーザーがローカルで適合を確認 |

再実行できる確認コード

`scripts/audit_blend_inventory.py` はPython 3.11以降の標準ライブラリだけで動きます。現在の非圧縮Blender 3.05形式に対象を限定し、異なる形式では明示的に失敗します。BlenderのPythonや同梱スクリプトを実行しません。

```bash
python3 scripts/audit_blend_inventory.py Z-Anatomy/Startup.blend \
  --output assets/evidence/z_anatomy_inventory.json
```

この棚卸しは実行済みです。7,184 Object、対象Curveの非空スプライン、対象Meshの面数を確認しました。**Blender 4.xでのメッシュ変換・レンダリングはタスクBで検証すべき残作業**です。
