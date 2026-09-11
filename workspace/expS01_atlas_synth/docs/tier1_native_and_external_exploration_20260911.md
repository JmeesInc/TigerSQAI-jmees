# Tier1: 元材質と公開テクスチャの探索（2026-09-11）

追記: neshalladsの公開ZIPをユーザーから受領し、実体を確認しました。以降の転写実装・検証は [肺テクスチャ転写記録](tier1_lung_texture_transfer_20260911.md) を参照。以下の未取得との記述は受領前の探索記録です。

現時点の判断は **見た目の改善を狙うなら②、①は比較用**。①を先に実装・試写した結果、対象の Z-Anatomy 材質は主に図譜配色で、期待した臓器の表面テクスチャを含まなかった。③の色度誤差が小さいことも、良い種画像である証拠にはならなかった。

## ① 実際に行った比較

同じ Startup.blend（SHA-256 `9f08a17ea0115fed80b2a73ecdf0a1bc2ab2f6956f37c593ce23d513ea35afcd`）から、manifest の29解剖オブジェクトに対応する21材質を抽出した。元の面ごとの材質スロットを維持し、ROI切り出し後の三角形を頂点インデックスで厳密に対応付けた。UVがある場合は三角形コーナーごとに保存する。頂点・面・カメラを変更しない。

材質は「元の不透明な表面」を復元する。図譜UIの断面切断、透過、comic/key-color表示は外し、元の Principled BSDF とその色・粗さ等を残した。これは Startup.blend の全表示設定を丸ごと再現したものではない。臓器材質の根拠がない手続き的な脂肪・胸膜・剥離面等には既存材質が残る。

重要な結果:

- グループまで展開して調べた21材質の表面ネットワークには画像テクスチャ、Noise/Voronoi/Wave、bumpがない。残るのは定数色、色の混合、Gamma等が主体。
- 血管が青紫、リンパ節が緑になる。視認性は高いが術野用の種画像として有利とは判断しない。
- 元の肺材質には `BLENDED` が残り、最初の試験で Depth は同一でも AOV に背後のクラスが出た。不透明な Principled/Alpha=1 と `DITHERED` に正規化して修正した。失敗出力は `outputs/tier1_native_compare`、採用した比較出力は `outputs/tier1_native_final`。前者を使用しない。
- 最終4枚（1024×576、Mac Blender 4.5.9 EEVEE）は **label PNG / depth EXR / geometry NPZ 全12ファイルのSHAが③の元出力と同一**。ラベル差分0画素、深度差0mm。
- 境界診断4/4通過。これは画素対応の確認であり、写実性の評価ではない。
- 描画0.89〜1.28秒/枚、起動・出力込み4枚8.74秒。小バッチ・キャッシュ状態に依存し、Linuxや2000枚の速度保証ではない。
- ユニットテスト render27 + registration8 =35通過。

ローカル比較画像: `outputs/native_vs_procedural.png`（左③、右①、同じ2姿勢）。この画像と比較用バイナリは自動Git同期対象外。

## 再現コマンド

expS01ルートで実行。生成される materials.yaml はそのマシンの絶対パスとライブラリSHAを含むため、各環境で作り直す。

```bash
python -m render.prepare_native_materials \
  --atlas Z-Anatomy/Startup.blend --blender /path/to/blender \
  --output outputs/native_library \
  --output-config outputs/native_library/materials.yaml

python -m render.replay_rgb outputs/pretrain_2k \
  --output outputs/pretrain_native_probe --limit 4 \
  --materials outputs/native_library/materials.yaml \
  --blender /path/to/blender --workers 1 --rerender-triplet

python -m render.alignment outputs/pretrain_native_probe --require-edge-evidence
```

既存入力がEEVEEなら `--rerender-triplet` を省略し、元のラベル・深度と厳密比較できる。既存Cycles入力からのEEVEE化では、承認済みの新tripletを使う。①を2000枚の既定に変更してはいない。

これは現行アトラス専用の探索用バックエンド。別バージョンで未対応の空間テクスチャノードや複数Principledが見つかった場合は停止し、無言で情報を捨てない。元blendは保存・上書きしない。

## ② 調査結果とライセンス

下表の配布ページと条件を2026-09-11に確認。Sketchfabは検索取得が403だったが、ブラウザの実ページでは閲覧できた。公開プレビューの存在だけを公開ダウンロードの許可とみなしていない。

| 候補 / 出典URL | ライセンス / 商用 | 再配布条件 | 今回の判断・確認状態 |
|---|---|---|---|
| [Z-Anatomy](https://github.com/Z-Anatomy/Models-of-human-anatomy) | 全体表記 CC BY-SA 4.0 / 可。上流に別ライセンス構成物の記載もある | 帰属、変更明記、翻案共有は同一/互換ライセンス。上流のコンポーネント別帰属も維持 | 現行胸部manifestのみ使用。全アトラスの全構成物に商用可と一括保証しない |
| [Realistic Human Lungs — neshallads](https://sketchfab.com/3d-models/realistic-human-lungs-ce09f4099a68467880f46e61eb9a3531) | **CC BY 4.0 / 可**。実ページのライセンスリンクまで確認 | 作者、出典、ライセンス、変更を明示。CC BY単体ではSA不要。ただしZ-Anatomyとの合成側条件は別途維持 | **②の第一候補**。配布者は詳細なテクスチャを説明。64k三角形/32k頂点。無料Downloadボタンあり、押すとログイン要求。未取得なのでPBRマップの実体・解像度・UV・見た目の改善は未検証 |
| [Meshy Organ gallery](https://www.meshy.ai/tags/organ) | ギャラリーの既成品についてCC0と案内 / 可 | CC0では帰属不要・再配布可。コンペwrite-upの開示は別途必要 | AI生成。個別素材ライセンスと生成元の適格性は別問題。今回の公開データ制約を満たすことを確認できず未採用 |
| [Human Internal Organs — unlim3d](https://sketchfab.com/3d-models/human-internal-organs-fe69d7b1ed6f46a3bd0b6933b796092e) | このページで配布ライセンスを確認できない / 要確認 | 要確認 | 4Kテクスチャの説明あり。ただしページにDownloadボタン/CCライセンスなし。今回未採用。別ストアの同名品と同一と断定しない |
| [Human Anatomy Organs Pack — ShowBeat Studio](https://sketchfab.com/3d-models/human-anatomy-organs-pack-7fd440196fe3480587de330967737848) | All rights reserved / 商用一般条件は別ストアで要確認 | 要確認 | **NoAIを実ページで確認。生成AIへの入力も禁止する表記のため今回除外**。2K/PBR/UV/GLB等の説明はあるが採用しない |
| [Stylized Accurate Human Lungs — danes_dysfunction](https://www.blendkit.com/asset-gallery-detail/c6c2ecde-5c4a-4e4e-ae30-ea0e2ae86150/) | **Royalty free、CC0ではない / 商用可** | 独立素材としての再配布不可。プロジェクト内組込みにも条件あり | 無料/3.2MiB/48,754 polygons。AI・合成データ配布への適用は要確認。CC BYの上記候補を優先。未取得 |
| [Human Lungs — Scribe / Daniel Stephens](https://opengameart.org/content/human-lungs) | **CC0 / 可** | 帰属不要、再配布可。作者は任意クレジットを希望 | **取得して調査済み、不採用**。478頂点、UVなし。旧式Cloudsテクスチャのみ。画像参照2つは未同梱・サイズ0で肺のPBR画像として利用できない |

条件の一次資料: [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/)、[CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/)、[CC0](https://creativecommons.org/publicdomain/zero/1.0/)、[Blendkit licenses](https://www.blendkit.com/docs/licenses/)、[Blendkit terms §5](https://www.blendkit.com/terms-and-conditions-2026/)。

取得したCC0ファイルは `https://opengameart.org/sites/default/files/lung.blend`。調査出力は `outputs/external_asset_audit/oga_audit.json`。配布者が付けたライセンスの確認と、素材内の第三者由来情報の完全な来歴検証は区別する。

## 次の具体的な比較

neshalladsの公開配布ファイルを取得できたら、まず肺だけで②を比較する。元のUV画像をそのままZ-Anatomyへ貼ると臓器配置が一致しないため、単なる画像差し替えでは完了しない。

1. GLB/glTFまたは元形式を監査し、画像・材質・UV・部位・出典を記録。別の未同梱画像や不明な権利素材は使用しない。
2. ドナーの左右肺とZ-Anatomyの左右肺を対応付け、**見た目の転写のためだけの**表面対応を作る。ドナー形状でアトラス形状を置き換えない。初期のbbox位置合わせだけでは局所対応は保証できず、要視認確認。
3. Base Color/roughnessを対応点のドナーUVからベイクする。法線マップはドナーの接線空間をそのまま流用しない。初回は省略するか、対応先接線空間へ再ベイクする。
4. 同じ4〜16姿勢で③/①/②を比較。特に肺の面積が大きい姿勢を含める。label/depth/geometryの不変を検証し、目視が改善した場合のみimg2img忠実度をLinux側で比較する。

現在の残る障害はSketchfabログインが必要な点。公開ファイルのダウンロード後のローカルパスがあれば転写検証を進められる。**②で改善したという結果はまだない**。今回の探索で①の限界と、利用可能性が確認できた②の候補を分けて記録した。
