探索用の元材質バックエンド（①）と外部素材の調査（②）は [比較記録](../docs/tier1_native_and_external_exploration_20260911.md) を参照。①は試写済みですが、術野の見た目の改善策としては採用していません。

# Appearance v2

Current defaults use shared chroma plus organ-specific procedural patterns. See `docs/tier1_appearance_v2_20260911.md`. Geometry and class IDs remain unchanged; EEVEE replays were checked for exact label/depth equality.

# Tier 1 RGB

`--emit-rgb --materials configs/materials.yaml` enables same-render EEVEE RGB / scalar-ID / depth. See `docs/tier1_20260911.md` for replay of existing synthetic batches and the Cycles-to-EEVEE exact-match limitation.

# Round 4 update

The active reference is `assets/reference_anatomy_only.yaml`: IDs 3–30, per-frame anatomy normalization. Instruments are disabled; no circular FOV mask is generated. Round 3 reports cannot unlock the bank. See `docs/round4_20260909.md`.

# Tier 0：内視鏡クラスIDレンダラ

Round 3: [窓とカメラの結合・IPV保護・128枚受入条件](../docs/round3_20260909.md)。バンク生成は受入ゲート通過まで停止。

**2026-09-09改訂**：fine_idと背景0はユーザー確認済み。既定priorを後期剥離に変更し、可視クラス数11〜18、心膜・肺靱帯proxy、公式配色、集計差分、Stage 1登録を追加した。新しい実行・検証・限界は [B改訂/Cの説明](../docs/task_bc_calibration_stage1.md) を優先する。以下の初版ベンチマークは変更前の参考値。

既存の公開Z-Anatomy形状を読み、**Blender CyclesのObject Index (`Object.pass_index`) とDepthパス**を直接取り出します。クラスIDはRGBや色管理を経由しません。生成対象はクラスID・深度・メタデータで、写実画像は生成しません。

## 最小実行

プロジェクトルートで実行します。Linux、Python 3.11、Blender 4.xを想定しています。Blenderに追加Pythonパッケージをインストールする必要はありません。Blender側は同梱NumPyとbpyだけ、前後処理は下記の仮想環境で動きます。

```bash
python3.11 -m venv .venv-render
.venv-render/bin/python -m pip install -r render/requirements.txt

.venv-render/bin/python -m render.generate \
  --frames 1000 --output outputs/tier0_1000 \
  --blender blender --device CUDA --workers 2 \
  --allow-provisional
```

`--allow-provisional` は、タスクAで残った部品の意味・由来の未確認事項をメタデータへ残して、**検証用の暫定マッピングで生成する明示的な選択**です。公開アトラスを用いる今回の最小実行用です。Task Aのマッピングファイルを黙って確定済みに書き換えません。監修済みのアトラスでは `configs/atlas.yaml` の部品・出典・SHA・statusを更新すれば、このフラグは不要です。

既定のアトラスは既存 `Z-Anatomy/Startup.blend`。SHAが一致しない別版はエラーにします。別版は `--atlas-config` と `--atlas` で指定し、同版のObject名または確認済みFMA sidecarから作った明示的な部品リストを使ってください。名前の曖昧一致は行いません。

`--workers` は常駐Blenderプロセス数です。CPUならコア数とメモリに合わせて、単一GPUならまず1〜2を使ってください。1プロセスがN枚を順次処理し、毎フレームBlenderを再起動しません。CUDA/OPTIX指定時にデバイスが見つからなければエラーにし、CPUへ黙って切り替えません。複数GPUの割当てはプロセス起動時の `CUDA_VISIBLE_DEVICES` 等でユーザー側で管理します。

出力先は新規または空ディレクトリに限ります。既存生成物を上書きしません。棄却上限に達したフレームは理由を `meta/*.rejected.json` に残して失敗を返し、指定N枚に足りない状態を成功と報告しません。現版は途中再開機能を持ちません。大量生成は複数の独立したseed・出力ディレクトリのバッチに分けられます。

## ファイル構成

| ファイル | 役割 |
|---|---|
| `render/generate.py` | CLI、準備、並列ワーカー、棄却再試行、出力・由来記録 |
| `render/export_atlas.py` | Blenderで既存Mesh/Curveを評価し、ワールド座標の三角形へ変換。右肋骨3〜10も取得 |
| `render/camera.py` | 肋間ポート、非一様事前分布、硬性鏡の逆幾何、内部パラメータ、逆歪みマップ |
| `render/volume.py` | 既存形状の膨張和、食道周囲脂肪、胸膜、累積剥離窓、剥離面、血液パッチ |
| `render/blender_worker.py` | 常駐bpy、シャフト衝突検査、Object Index/Depth EXR出力 |
| `render/output.py` | Indexの整数検査、最近傍歪み、uint8 PNG・float32 EXR保存、画素統計フィルタ |
| `render/registration.py` | 公開CTマスクのメッシュ化と3D TPSによるアトラス変形 |
| `render/visualize.py` | 出力済みラベルのQA用着色。教師生成経路から独立 |
| `render/audit_outputs.py` | 保存されたPNG/EXR/JSONとカメラ幾何の整合確認 |
| `render/verify_blender.py` | 既知の平面・遮蔽・非中央主点・fx≠fyを実際のBlenderで検証 |

## 出力

```text
label/00000000.png       # uint8、単チャンネルL、整数0..30
depth/00000000.exr       # float32、単一Zチャンネル、線形カメラ軸方向深度mm
meta/00000000.json
provenance.json         # アトラス・CT・設定・ポート・未対応クラスの由来
resolved_configs.json   # 当該実行で実際に使った全設定
metrics.json            # 初期化、フレーム、実測処理時間
logs/                   # Blenderログ
work/                   # 評価メッシュ・脂肪ボリューム・ワーカー中間物
raw/                    # --keep-rawを指定した場合の検証用多層EXR
```

`fine_id` は公式IDと一致するクラス順1〜30、0は背景です（ユーザーが2026-09-08に確認）。`meta`には以下を保存します。

- 歪み適用後画像に対応するK、歪み係数、overscanレンダー用Kと解像度。
- Blender座標系とCV座標系それぞれのcamera-to-world、CVのworld-to-camera。並進の単位はmm。
- ポート位置・肋間番号、先端位置、シャフト軸、視軸、斜視角、シャフト回転、センサーロール。
- ポート→対象距離、挿入量、先端→対象距離。これらは別の量です。
- 患者ID、seed・試行番号、剥離進行度・フェーズ・残存被覆ボクセル数。
- 可視クラスID、背景を含む31クラスの画素数、器具数、手続き的物体数、棄却理由。
- `visible_stations: null`、`station_nomenclature: 要確認`。空配列で「全ステーション陰性」と誤解させません。

深度0は遠景・未定義画素です。背景ID 0でも、対象外の肋骨などが遮蔽物として映れば深度は有効です。Depth.Zは**Blender 4.5.9のCycles透視カメラで実測した軸方向深度**を保存します。光線距離への変換は行いません。別のレンダーエンジンへ変更する場合は実平面テストを再実行してください。

肋骨の0割当ては `configs/atlas.yaml` の `context_class_id` に明示した暫定規則です。主催者が肋骨をOtherに含めると定義している場合は2へ変更してください。公式注釈仕様の代わりに解剖名から決めた規則ではありません。

Object Indexは非アンチエイリアスパスです。`filter_size=0.01` とCyclesのフィルタ幅0.01、1 sample、DOFなし、モーションブラーなし、デノイズなしです。生のfloat Indexで整数からの偏差が1e-5を超えた場合は失敗させます。PNGにする最後の段階のみuint8へ変換します。色をクラスIDへ戻す処理はありません。[Blenderパス仕様](https://docs.blender.org/manual/en/4.4/render/layers/passes.html)

## カメラモデル

`configs/camera_prior.yaml` を差し替えるだけで、タスクCの較正結果を反映できます。

```bash
.venv-render/bin/python -m render.generate \
  --frames 100 --output outputs/calibrated \
  --camera-prior configs/camera_prior_calibrated.yaml \
  --device CUDA --allow-provisional
```

採用した分布は、肋間・ポート数・斜視角の重み付きカテゴリカル、角度・FOV・ロールの切断正規、撮影距離の切断対数正規、シャフト方位のvon Misesです。一様な自由空間位置・SO(3)姿勢は使いません。ポートは既存の右肋骨nとn+1の形状から、設定した腋窩〜後側方の角度帯にある肋間点を求めます。全注視候補へ到達できるポート集合に条件付けます。

座標は患者RAS（+X右、+Y前、+Z頭側）、mm。左右・前後変換はアトラス設定に明示しています。ポートは患者ごとに3〜4個固定し、フレームごとに撮影ポートを選びます。中腋窩〜肩甲骨下角線を正確な筋膜ランドマークとして同定したわけではなく、**既存肋骨形状と設定した後側方角度帯による近似**です。これらの境界は較正・監修対象です。

硬性鏡は、P=ポート、S=シャフト単位軸、i=挿入量、C=先端、V=視軸、d=撮影距離、T=注視点として、

```text
C = P + i S
T = C + d V
S·V = cos(theta)
V(phi) = cos(theta) S + sin(theta) [cos(phi) U + sin(phi) W]
```

を満たします。U,WはSに直交します。固定シャフトのphi回転でVが円錐を描くことと、逆問題で求めた先端・シャフトがPとTに一致することをテストしています。30度はカメラの適当なEuler角への加算ではありません。センサーロールはこの光学配置に対するカメラヘッドの回転として独立に扱います。

ご指定の「ポートから対象まで20〜120mm」は `ports.port_to_target_mm` の制約として保持しています。一方、硬性鏡で可変なのは挿入量と先端からの撮影距離なので、`scope.working_distance_mm` にも20〜120mmの近接優先分布を用意しています。幾何学的に両者を満たせない候補は棄却します。実際の胸郭でポート→対象が120mmを超える場合、**この設定をユーザーの較正結果で変更する必要**があります。

視軸とシャフトの内側成分、先端の離隔、シャフトの表面交差を確認します。シャフトの半径も、軸に沿った距離サンプルに半ステップの安全距離を加えた保守的検査で扱います。右肋骨も衝突対象です。これは形状ベースの検査で、肋間軟部組織や組織変形を伴う挿入シミュレーションではありません。器具もポートから伸ばし、組織を横断する候補を棄却します。器具とスコープ同士の完全な剛体接触・関節運動学は未実装です。

内部パラメータは水平FOVまたは `intrinsics.focal_x_px`、fy/fx、主点の正規化オフセットを設定できます。後処理の樽型歪みはBrown型のk1/k2による逆写像をNewton法で解き、**ラベルと深度に同一の最近傍座標**を適用します。歪みで画角端を失いにくいようoverscanを行い、そのKも記録します。単調性のない歪みや収束しない逆写像は棄却します。高次・魚眼・接線歪みは現版の範囲外です。

## 脂肪・胸膜・剥離

アトラス由来の縦隔ターゲットをボクセル化し、内部充填した形状の膨張和を作ります。脂肪はその和から元の構造領域を除いたボリュームです。食道からの距離でclass 7と20を分離し、その外側にclass 10の胸膜層を付けます。肺と肋骨は文脈・遮蔽物として扱い、縦隔脂肪で肺全体を包みません。包むクラス集合は `envelope_class_ids` で明示しています。

既定ボクセル幅2.5mm、脂肪6mm、食道周囲8mm、胸膜外形の膨張幅2.5mm、薄膜メッシュ厚0.3mmは**Tier 0の近似パラメータで、実測値ではありません**。胸膜は膨張和の外表面から作り、窓内の面を除去して薄膜の表裏と縁を構成します。独立した中空ボクセルラベルの内面と脂肪の面を重ねないため、境界の三角形状の漏れを避けられます。小さいボクセル幅へ変更すると境界が細かくなりますが、メモリ量は概ね幅の逆3乗で増えます。元の神経Mesh/Curve自体をこの解像度へ置き換える処理ではありません。

`configs/dissection.yaml` は次の累積窓を定義します。

| t | フェーズ |
|---|---|
| 0〜0.15 | 胸膜切開前、除去なし |
| 0.15〜0.35 | 下方の胸膜切開・下肺靱帯領域 |
| 0.35〜0.50 | 奇静脈弓周囲の露出 |
| 0.50〜0.78 | 食道周囲・気管支周囲の郭清窓 |
| 0.78〜0.92 | 上縦隔の露出 |
| 0.92〜1.0 | 大動脈側の追加露出 |

窓の原形は楕円体ですが、固定された3Dノイズ場で境界を不規則にしています。tが増えると既に除去した脂肪・胸膜は復活しません。フェーズ、アンカー、半径、時間区間、ノイズをすべて設定化しています。これは**術式順序の例示事前分布**であり、症例ごとの正しい郭清手順を保証するものではありません。

実装するのは被覆組織の除去です。奇静脈の切離・クリップ処理や、未収載の肺靱帯・反回神経・胸管を新しく作る処理はありません。タスクAで未同定だった構造は引き続き未対応として記録します。その構造を既存アトラスから同定できれば、明示的な部品リストへ追加して同じ窓で露出できます。

`Resection area` は除去窓に隣接する残存脂肪の小領域へ付与し、臓器表面を無差別に25へ塗り替えません。`Pool of blood` は剥離窓に面する残存組織のうち、重力に対して支持面となる低い位置へ小さい扁平パッチを置く近似です。流体シミュレーションではありません。両者はオプションで、出血は剥離面ラベルの有効/無効と独立です。

器具はシリンダ＋2本の顎で0〜2本です。出現数の事前分布と、衝突・画素統計フィルタを通った最終分布は一致するとは限りません。各フレームの生成数と可視画素数を較正に使ってください。

切開前に「露出臓器が必ず1つある」と要求すると、そのフェーズをすべて捨ててしまいます。このため既定フィルタはt≤0.20で胸膜・脂肪も主要可視クラスとして認め、その後は設定された臓器クラスを要求します。95%以上を1クラスが占めるフレーム、背景過多、光学領域不足も棄却します。これは明示的なフェーズ依存設定です。

## 任意のCT患者経路

`configs/patient.example.yaml` は入力仕様のテンプレートです。nullは実データなしでは決められない座標等であり、架空の患者値ではありません。公開CTの付属セグメンテーションをローカルで用意して埋めます。TotalSegmentator推論や重みダウンロードはこのコードから一切行いません。

```bash
.venv-render/bin/python -m render.generate \
  --frames 100 --output outputs/public_patient_001 \
  --patient configs/patient_001.yaml \
  --device CUDA --allow-provisional
```

1. 骨格、気管、大動脈の対応ランドマークを、アトラスRAS mmとCTのNIfTI affineで定義されるRAS mmで入力します。最低5点かつ非共面、実用上は上下・左右・前後に分布した8〜12点以上を推奨します。
2. 3次元polyharmonic TPS（U(r)=−r、affine項と正則化付き）を解き、アトラス細構造・肋骨ポート基準・術野アンカーを一貫して変形します。
3. (b)の気管3、食道6、大動脈15、SVC17、肺18を、公開CTマスクからmarching cubesで得たメッシュに置換できます。NIfTIの回転・反転・spacing・originをaffineとして適用します。単純にspacingだけを掛ける実装ではありません。
4. 変形のJacobianを複数点で調べ、折り返しや過大圧縮を棄却します。サンプル点で正でも全領域の単射性は証明されないため、位置合わせは目視確認が必要です。
5. CTのROI、胸郭中心、肺虚脱の支点は患者別に設定します。脂肪・胸膜は変形・置換後の形状から作り直します。

この経路は、人工的なテスト用マスクによる**affineとTPS統合テスト**を実行済みです。実際の公開患者CTをダウンロードしての品質検証は未実施です。公開CT付属マスクはCC BY 4.0ですが、注釈生成由来の許容範囲はタスクAの未確認事項を引き継ぎます。

## 確認と可視化

```bash
.venv-render/bin/python -m unittest discover -s render/tests -v
.venv-render/bin/python -m render.verify_blender --blender blender
.venv-render/bin/python -m render.audit_outputs outputs/tier0_1000
.venv-render/bin/python -m render.visualize outputs/tier0_1000 \
  --output outputs/tier0_1000/preview --limit 12
```

`--palette assets/palette_fine_official.yaml` でユーザー提供の公式配色を使えます。省略時だけ識別用の仮パレットです。直接の `fine_id: [R,G,B]` と、`palette`キーで包んだ形式の両方を受け付けます。着色は教師PNGからの一方向処理です。

固定した剥離段階を確認するには `--progress` を使います。新しい既定のクラス数11〜18フィルタは切開前には適しません。早期フェーズのQAだけは別カメラ設定で `quality.visible_class_count.enabled: false` にしてください。患者ポートとseedは同じにできますが、棄却後のカメラが同一になるとは限りません。

## ライセンスと出典

| 使用物 | 出典 | ライセンス / 商用 / 再配布 |
|---|---|---|
| Z-Anatomyの明示的な胸部部品 | [公式リポジトリ](https://github.com/Z-Anatomy/Models-of-human-anatomy) | CC BY-SA 4.0表記、帰属・変更表示・SA。部品ごとのNC等の追加素材由来はTask Aの確認事項を継続。全体を無条件に商用可とはしない |
| 任意の公開CTマスク | [TotalSegmentator v2.0.1](https://zenodo.org/records/10047292) | CC BY 4.0、商用可、帰属・許諾・変更表示。推論重みとは別 |
| Blender | [公式](https://www.blender.org/about/license/) | GPL、商用利用可、ソフト再配布はGPL条件。生成データがBlenderを使っただけでGPLになるわけではなく、アトラス由来の条件を保持 |
| NumPy / SciPy / scikit-image | [NumPy](https://github.com/numpy/numpy)、[SciPy](https://github.com/scipy/scipy)、[scikit-image](https://github.com/scikit-image/scikit-image) | BSD-3-Clause、商用可、再配布時に著作権・許諾・免責表示 |
| trimesh / PyYAML / NiBabel | [trimesh](https://github.com/mikedh/trimesh)、[PyYAML](https://github.com/yaml/pyyaml)、[NiBabel](https://github.com/nipy/nibabel) | MIT、商用可、再配布時に著作権・許諾表示 |
| Pillow | [公式](https://github.com/python-pillow/Pillow) | MIT-CMU、商用可、再配布時に許諾表示 |
| OpenEXR | [公式](https://github.com/AcademySoftwareFoundation/openexr) | BSD-3-Clause、商用可、再配布時に著作権・許諾・免責表示 |

新しい神経・胸管等の解剖形状を推測生成していません。新規プリミティブは器具と血液パッチ、手続き的形状は既存アトラスを元にした被覆・除去です。ソフトウェアの配布条件と、アトラス由来の生成画像・メッシュの配布条件は別です。詳しい素材条件は `docs/task_a_atlas_selection.md` を参照してください。

## 検証環境と時間

実機検証はmacOS arm64、Python 3.12、**公式Blender 4.5.9 LTS / Cycles CPU**です。依存バージョンは `render/validation_environment.txt`。Linux CUDAでの実測は未実施です。Pythonコードは3.11対応の構文・依存範囲で記述しています。

実測値は生成先の `metrics.json` を正としてください。初期化・ボクセル化、採用フレームの描画時間、棄却込みの処理時間を分けて保存します。Tier 0はBVH・メッシュ処理・入出力・棄却の割合が大きいため、GPUで全体が描画速度に比例して速くなるとは見積もりません。最初の10〜20枚で実環境の処理時間と棄却理由を測り、N枚の予算を求められます。

胸膜修正後の `outputs/tier0_release/metrics.json` の実測（1024×576、2並列、10枚）は以下です。

| 指標 | 実測 |
|---|---|
| 初期化・アトラス評価・被覆準備 | 5.75秒 |
| 採用フレームのBlender処理中央値（形状処理も含む） | 0.70秒 |
| 各フレーム処理中央値（棄却等を含む） | 3.33秒 |
| 全体時間÷出力枚数（初期化・並列化・棄却を含む） | **5.94秒/枚** |
| 10枚合計 | 59.39秒 |

同じ実機・事前分布なら、まず**数秒〜20秒/枚程度の個別処理、バッチ全体で約6秒/枚**を目安にします。1000枚で約1.7時間はこの短い測定からの外挿であり保証値ではありません。切開前・厳しい距離/衝突条件・器具を強制する設定は棄却数が増えます。Linux NVIDIA/CUDAの予算は未実測のため、このCPU測定を暫定値にし、同じCLIで再計測してください。

この10枚は5フェーズすべてを含み、保存後の全画素統計・型・深度・カメラ幾何を検査済みです。別の器具強制テストではclass 1を含む3フレームを確認しました。7つの数値/出力テストと実Blenderの投影・深度テストも通過しています。出血パッチの構築は単体テストで確認しましたが、この10枚でclass 24の可視画素はありません。配置可能性と可視性を混同しません。
