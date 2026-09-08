> 2026-09-09 Round 3追記: バンク生成は128枚の4指標合格と12/8/9の可視確認まで停止。新しい必須引数 `--calibration-report` と最新の条件は [round3_20260909.md](round3_20260909.md) を参照。下記Stage 1初版の時間・結果は履歴。

# B改訂・C Stage 1 — ローカル較正とアトラス登録

2026-09-09。実マスクにはアクセスせず、ユーザー提供の528枚の集計値と公開アトラスで実装・検証した。実フレームの登録精度、実患者のカメラ姿勢、学習スコアは未検証。

## 確定事項と較正案の区別

- fine_id 1..30、背景0はユーザー確認済み。`assets/class_mapping.yaml` に反映した。形状の意味・FMA binding・部品ライセンスの未確認事項まで解消したわけではない。
- `assets/palette_fine_official.yaml` は提供された31色。`schema_version / palette` 形式と従来のID→RGB形式の両方を可視化CLIが読む。出典はユーザー提供の公式評価コード由来。元リポジトリの具体的URL・ライセンス条項は未提示なので捏造しない。
- class 22は提供された学習528枚で出現0、今回の補充対象外。未知の評価集合で必ず0とは主張しない。
- 背景面積の**全フレーム中央値18.0%**をユーザーが追認。表の出現時33.96%は争点を残した原値として保存し、条件付き中央値の差分には使わない。
- 可視クラス数11〜18は**背景除外、1画素以上**として設定した。この数え方は集計担当側で一致を確認すること。レポートはこの定義を明記する。
- ステーション命名法は「要確認」、`visible_stations: null` のまま。Cはステーション名・番号を検索特徴として使わない。

## Bの変更

`configs/dissection.yaml` はBeta(4,1.5)を[0.5,1]に切断。pre_incisionの確率は0。器具数重みは[0.098,0.602,0.300]、剥離面生成確率0.809、出血生成確率0.536。**これらは集計値に基づく初期提案で、最終的な可視率の推定値ではない。** 衝突・遮蔽・画素数による選択で分布は変わる。

器具は別ポートから、カメラ先端と注視点の間の作業空間へ伸ばす。`target_depth_fraction` と `tip_standoff_mm` を設定化した。組織を横断する姿勢の棄却は維持する。器具ありの候補が棄却されやすいため、0本priorを9.8%としても採用画像の器具可視率90.2%は保証されない。

右肺の虚脱変形を[0.35,0.65,0.85]から[0.50,0.75,0.90]へ弱めた。肺の可視面積を増やすための仮設定であり、虚脱の実測ではない。可視クラス数11〜18を主要な受入条件に追加し、95%単一クラス棄却を維持した。**このフィルタは欠けたクラスを生成するものではなく、上縦隔など多くのクラスが同時に見える視点へ偏る可能性がある。**

### 一括無効化できる暫定構造

`configs/atlas.yaml: provisional.enabled` または `--disable-provisional-structures` で次をすべて無効化できる。既存アトラスの未確認identityを許容する `--allow-provisional` とは役割が別。

| ID | proxy | 拘束・限界 |
|---|---|---|
|11|既存 Left/Right atrium、Left/Right ventricle の充填膨張和の外表面|1.5mm margin、2mm voxel。心膜反転部・洞・大血管根部の形状を再現しない。心臓を心膜と同一視する確定ラベルではない|
|8/9|既存下肺静脈より尾側で、下葉の内側表面と既存縦隔表面を結ぶ二重の薄い帯|左右別の下葉/IPV名、近接表面・頭尾位置で拘束。実肺靱帯の同定ではなく、長さ・厚さは設定値。切離シミュレーションは行わない|

両者ともアトラス形状の変換・ランドマーク拘束による**provisional_anatomy_proxy**であり、メタデータにID・元部品名・方法・ライセンスを記録する。元アトラスと同じ出典・派生条件を引き継ぐ。CT経路は四心腔をTPS変形してから心膜proxyを作る。CT肺置換で左右下葉名が失われる場合、肺靱帯proxyはエラーにするため、そのproxyを設定で無効化するか、明示的な下葉対応を追加する。

肺靱帯が下葉を縦隔に繋ぐ二重胸膜である根拠は[CT/cadaver対応研究](https://pubmed.ncbi.nlm.nih.gov/6603110/)、心膜が心臓・大血管根部を囲む関係は[心膜を保った解剖研究](https://pubmed.ncbi.nlm.nih.gov/34147441/)を参照した。ここからproxyの個別寸法・輪郭の正しさは導かれない。

反回神経27/28、胸管30は今回追加していない。器具・血液以外の自由なプリミティブでこれらを捏造しない。

## 実行手順

以下は実験ディレクトリ `workspace/expS01_atlas_synth/` をカレントとして実行する。依存は既存 `render/requirements.txt` のみ。Linux/Python 3.11は以前の7テストがLinux担当者により確認済み、新規C実装の当方検証環境はmacOS/Python 3.12。

```bash
python3.11 -m venv .venv-render
.venv-render/bin/python -m pip install -r render/requirements.txt

# まず小バッチ。Blenderへのパスは各環境で指定。
.venv-render/bin/python -m render.generate \
  --frames 64 --output outputs/calibration_001 --resolution 256 144 \
  --blender /path/to/blender --device CUDA --workers 2 --allow-provisional

.venv-render/bin/python -m render.calibration_report outputs/calibration_001 \
  --reference assets/real_mask_statistics.yaml --output outputs/calibration_001/report
.venv-render/bin/python -m render.visualize outputs/calibration_001 \
  --output outputs/calibration_001/preview --palette assets/palette_fine_official.yaml
```

Linuxは[Blender公式ダウンロード](https://www.blender.org/download/lts/)からLinux x64の4.x LTSアーカイブを取得し、ユーザー権限のディレクトリへ展開した実行ファイルを指定できる。root不要。配布アーカイブ名・チェックサムは取得するリリースの公式一覧で確認する。BlenderはGPL、商用利用可、ソフト再配布はGPL条件。アトラス再配布の条件とは別。CUDAランタイム・ドライバの対応はその環境で要確認。

## C-1: 参照バンク

```bash
.venv-render/bin/python -m registration.generate_bank \
  --frames 2048 --batches 8 --output outputs/bank_001 \
  --blender /path/to/blender --device CUDA --workers 2 --allow-provisional
```

各バッチでseedとポート配置を変え、同じ患者座標系・アトラスで別の物理的カメラ候補を生成する。自由空間の一様サンプリングは行わない。全バッチ成功時だけ `outputs/bank_001/bank/` を索引化する。既存のバッチも利用できる。

```bash
.venv-render/bin/python -m registration.bank outputs/calibration_001 \
  --output outputs/bank_small
```

`coverage.json` は各クラス出現率、対応可能だが未出現のID、target別件数、ロール/FOV/距離の範囲を出す。**対応可能でも一度も写らないクラスは検索で解決できない。** そのIDを都合よく自動除外せず、バンクの視点・被覆を改善するか、設定で理由付き除外する。

|用途|解像度|バンク規模|限界|
|---|---|---|---|
|配線・失敗検出の確認|256×144|64〜256|網羅性・登録精度の判断には足りない|
|最初のStage 1実験|256×144|2,048、8以上のポート配置|粗い検索。細線・小血管の消失がありうる|
|探索範囲拡張|256×144|8,192、32以上のポート配置|受入率・候補端への集中を見て増やす|
|小構造を含む再実験|512×288|2,048〜8,192|メモリ・I/Oが増える。変更後はバンクを再作成|

256×144ならPNGの非圧縮相当は2,048枚で約72MiB。深度・元EXR・形状中間物が別途ある。全体の生成時間は画素数だけに比例しない。今回24枚は57.72秒（Mac CPU、2並列、準備込み2.40秒/枚）。同じ条件の粗い外挿で2,048枚約1.4時間、8,192枚約5.5時間だが、別ポート配置や厳しい棄却で増える。CUDAは未実測。

## C-2/3: 記述子検索→IoU→信頼度

```bash
.venv-render/bin/python -m registration.register \
  --bank outputs/bank_001/bank \
  --masks /LOCAL_ONLY/class_id_masks/train_fold0 \
  --output outputs/real_registration_fold0
```

入力は**クラスIDのLまたはPモードPNG**。RGBはエラーにする。実マスクがRGBで保存されている場合、Linux側の公式labelmap変換でID化してから渡す。レンダラのObject Index出力は一切RGB変換を経由しない。

記述子は出現binary、全画素に対する面積比、正規化重心、中心2次モーメント(xx,yy,xy)、4近傍で境界を共有する隣接行列。存在maskを別に持つ。距離はpresenceのJaccard型差、面積L1の正規化、共通可視クラスの重心/モーメント差、隣接edgeのJaccard差を設定重みで合算する。全バンクに対する厳密距離の上位64をkNNとし、選択候補だけを画素IoUで再スコアする。

比較IDは**実側の許可IDとバンクの対応可能IDの交差**。既定除外は0背景、1/2/24/25/26/29の術中状態、22/27/28/30の未対応構造。8/9/11はproxyが有効かつ設定で許可した場合のみ使う。両方の画像に現在写っているIDの共通部分だけに限定する実装ではない。片側で欠けた対応可能クラスはpresenceとIoUで誤差になる。重心・モーメントだけは両側に値があるとき比較する。

IoUはfineクラスの評価重みを用いるが、**公式コンペ評価指標ではなく検索用スコア**。両側で不出現のクラスは分母に加えない。器具など除外前景に覆われた画素は両側とも未知として比較から外す。背景はIoUクラスに含めないが、臓器対背景の誤差は残す。比較可能な臓器画素が少なければ棄却する。

### ロール・スケール

画像平面の後付け回転、拡大縮小、左右反転でIoUを最大化しない。バンクで実際にレンダリングしたロール、FOV、撮影距離の候補から選ぶ。入力解像度だけ最近傍で正規化し、アスペクト比が設定許容差を超える場合は棄却する。元解像度のKはpixel-center規約で戻し、正規化歪み係数は維持する。これにより2D整列だけ良くして3D姿勢を不整合にすることを避ける。バンクにないロール/FOVはStage 1で回復できない。

### 信頼度と失敗検出

confidenceは `weighted_IoU × exp(-descriptor_distance) × class_support × spatial_agreement`。**確率較正済みの正解確率ではない。** 閾値は `configs/registration.yaml` で変更する。

- weighted IoU不足、記述子が分布外、共通可視クラス不足、比較可能画素不足を棄却。
- ほぼ同じIoU（既定差0.025以内）の候補が25mmまたは25度以上離れる場合、幾何的に曖昧として棄却。
- クラスごとのIoU、上位候補差、近接スコア候補の姿勢分散、失敗理由を保存。
- バンク候補が10未満なら十分な代替仮説を比較できないとして棄却。
- 小構造が低解像度で消える、心膜proxyが血管を隠す、肺虚脱が合わない、ROI不足などは低IoU/coverage不足で検出し、形状側の改善が必要。信頼度が高くても単一マスクからの一意性は証明できない。

`poses/{frame_id}.json` に採択フラグ、推定camera、t、confidence、top_k、対応ID、暫定構造ID、失敗理由を出す。棄却例にも最良候補は診断用に残るので、後段は**必ず `accepted == true` を確認**する。姿勢の並進はアトラスRAS mmであり、実患者のCT座標ではない。Dのcarvingで患者座標の真値と混同しない。

## C-4: 事前分布の出力

```bash
.venv-render/bin/python -m registration.fit_priors \
  --registration outputs/real_registration_fold0 \
  --output outputs/fitted_fold0

.venv-render/bin/python -m render.generate \
  --frames 128 --output outputs/calibration_002 \
  --camera-prior outputs/fitted_fold0/camera_prior.yaml \
  --dissection outputs/fitted_fold0/dissection.yaml \
  --blender /path/to/blender --device CUDA --allow-provisional

.venv-render/bin/python -m render.calibration_report outputs/calibration_002 \
  --output outputs/calibration_002/report
```

既定で採択20枚以上、異なる参照姿勢8以上、単一候補への集中25%以下を要求し、満たさなければYAMLを出さない。平均・分散は元priorへ縮約して、元の物理範囲を保持する。ロール、FOV、距離、斜視角、シャフト方位、注視対象、歪み、内部パラメータの分布を更新する。ポート位置・患者形状の再推定は行わない。進行度は切断区間の正規化を含むBeta尤度＋元priorへの罰則で更新する。

器具の有無は元解像度の実マスクのclass 1出現から推定するが、連結成分数を器具数と同一視しない。1本/2本の比は維持し、0本確率だけ更新する。剥離面/血液も採択集合の可視率からの提案確率である。これらの生成確率をそのまま可視率だと解釈せず、**再描画後の差分表**で判断する。

出力は元と同じYAMLスキーマで、fit_reportに採択数・参照姿勢の集中・有効候補数を記録する。検索は既存priorで作ったバンクに条件付けられるため、真のパラメータへの収束を保証しない。

CVでは**case単位で分け、train側マスクだけでfit**する。全528枚を使ったpriorを同じ528枚のCV改善として報告しない。ユーザー提供の全体集計を使う本初期priorも厳密なfold内推定とは異なる。最終全train学習用とfold別検証用を分ける。

## 検証と残る差

```bash
.venv-render/bin/python -m unittest discover -s render/tests -v
.venv-render/bin/python -m unittest discover -s registration/tests -v
.venv-render/bin/python -m render.verify_blender --blender /path/to/blender
```

既存7テスト＋Cの8テストを通過。改訂版24枚はObject Index/深度/画素数/カメラ幾何監査を通過。自己参照検索は24/24で対応する元フレームがIoU=1となり、そこから同スキーマのcamera/dissection YAMLを出力した。**これは配線・座標整合のテストで、実データの登録精度ではない。**

|量|参照実集計|改訂版24枚|
|---|---:|---:|
|可視クラス数 中央値|15|14|
|Instrument出現率|90.2%|54.17%|
|Pericardium出現率|70.8%|62.50%|
|Lung出現率|71.0%|66.67%|
|Resection area出現率|80.9%|79.17%|
|Pleura平均面積|16.44%|8.00%|
|SVC平均面積|0.31%|8.76%|
|Pool of blood出現率|53.6%|0%|
|IPL R/L出現率|25.9/27.1%|0/0%|

血液パッチ・靱帯proxyは形状として構築されるが、このバッチで可視画素は得られなかった。下肺静脈も0%。この差を隠して生成成功を統計一致と報告しない。主な次の較正対象は器具の太さ/到達空間、下縦隔を含むポートと被覆、心膜proxy/IPVの遮蔽、血液の位置。全31クラスの差分は `render.calibration_report` が出す。画像単位ではなく大きい独立バッチで改善を確認する。

## 同期とローカル情報

同期対象はコード、設定、共有済み集計、ドキュメントのみ。`outputs/`、実マスク、実フレームごとの推定姿勢・confidence・topK・ファイル名を含む派生物は同期しない。実行時は `outputs/` 配下に保存する。公開アトラス本体も大容量・派生条件のためコミットせず、各環境で入手してSHAを照合する。

追加の結合検証: 出力した較正版YAMLを再投入し、別seedの2バッチ・計4枚を描画・バンク化できた。元24枚バンクで別seedの2枚を検索するとIoU 0.197/0.074で両方低信頼棄却。小バンクの不足を検出した結果であり、失敗を隠して姿勢採択しない。
