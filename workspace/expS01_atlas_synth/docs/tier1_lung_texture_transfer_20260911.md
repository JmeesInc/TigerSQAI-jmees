# neshallads 肺テクスチャ確認・転写試験

ユーザー提供 `realistic-human-lungs.zip` を確認。アーカイブSHA-256:
`6ac2e8f0e0838d525ca0f4cdacd861a53b2fef160c8883e0d174c9a7b79c1d78`。

出典: [Realistic Human Lungs — neshallads](https://sketchfab.com/3d-models/realistic-human-lungs-ce09f4099a68467880f46e61eb9a3531)。ライセンスは配布ページで確認済みの [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/)。商用・改変・再配布可、作者・出典・ライセンス・変更を明示する。Z-Anatomy由来の形状を含む合成物はその帰属・ShareAlike条件も別途維持。ZIP自体にはライセンス文書が含まれないため、配布ページの確認記録とこのアーカイブSHAを紐付ける。

## 実ファイルの確認結果

- `source/Lungs With Texture.fbx`: 30,812,812 bytes。
- `textures/`: 13画像。part01/part02にBase Color、Normal、Height、AO、Roughness、Metalnessが各1枚。各2048×2048。残る1枚はground AO（512×512）で使用しない。
- Roughnessは両方とも77/255の定数画像。Metalnessは両方ゼロ。画像が存在することと細かな材質変化を含むことは異なる。
- Base Color/Height/Normalには非一様の模様がある。今回はBase ColorとHeightのみ利用。
- Blender4.5.9でFBXを読み、part01: 41,735 vertices/46,236 triangles、part02: 13,334 vertices/17,738 triangles、合計63,974 trianglesを確認。
- part02の材質 `thairoid01:lungh_part02` が転写元の肺表面。part01は気道等を含む別組。利用部位は材質名で明示し、部位全体の細かい解剖学的正確さは保証しない。
- FBXが参照するBase Color名は `.jpg`、ZIPの対応ファイルは `.jpeg`。材質をインポート任せにせず、設定のファイル名で明示的に結び直す。

## 今回の実装範囲

**既存5肺葉の形状を一切変更せず、肺クラス18の見た目だけを変更**する。その他のクラスは③のまま。剥離窓・胸膜/脂肪・カメラ・クラスID・深度には手を加えない。

1. アトラスの未虚脱メッシュから左右それぞれの肺全体bboxを取得する。
2. ドナーを設定された軸方向に変換し、左右のbboxをそれぞれ正規化する。
3. アトラス表面の各サンプルに最も近いドナー三角形を探し、その点の重心座標でドナーUVを得る。
4. Base Color/HeightをドナーUVでサンプルし、アトラス各三角形に独立した小さなテクスチャ領域としてベイクする。UVシームをまたぐ画像補間を避けるため周囲をパディングする。
5. 元の頂点インデックス・三角形が一致する場合だけベイクUVを付与する。描画時に肺が虚脱していても面の対応は維持される。

**これは暫定的な外観対応で、医学的な非剛体位置合わせではない。** 左右軸・bbox・最近傍対応の誤差や模様の歪みは残りうる。ドナーNormalは接線空間が異なるため使用せず、Heightを法線だけに作用するbumpへ使う。displacementは使わない。

`pixels_per_face: 8` は探索用の小さなベイク解像度であり、元の2K画像の情報をすべて保持しない。増やすと細部は増えるがベイク時間・メモリ・画像サイズが増える。初回に一度ベイクすれば、フレームごとの転写計算は不要。

## 再現手順

元バッチの `work/atlas.npz` と同名 `atlas.json` が必要（`base.npz` は虚脱・ROI等の加工済みなので代用しない）。典型例:

```bash
python -m render.prepare_lung_texture \
  --archive realistic-human-lungs.zip \
  --atlas-mesh outputs/pretrain_2k/work/atlas.npz \
  --output outputs/lung_transfer \
  --blender /path/to/blender

python -m render.replay_rgb outputs/pretrain_2k \
  --output outputs/lung_texture_probe --limit 4 \
  --materials outputs/lung_transfer/materials.yaml \
  --blender /path/to/blender --workers 1 --rerender-triplet

python -m render.alignment outputs/lung_texture_probe --require-edge-evidence
python -m render.audit_outputs outputs/lung_texture_probe
```

EEVEE入力との厳密比較では `--rerender-triplet` を省く。Cycles入力をEEVEEへ変換する場合は既定の承認済みtriplet再出力方針に従う。再ベイク先・再出力先は空のディレクトリを指定する。

`configs/lung_texture_transfer.yaml` で転写対象材質・ファイル・軸・ベイク解像度・bumpを変更可能。生成されたmanifestには対応距離のp50/p95/max（bbox正規化空間での距離）、画像SHA、元アトラスSHA、出典を保存。CT由来の別トポロジーには無言で転写せず停止する。

ライブラリと画像はローカルに生成し、自動Git同期ではバイナリアセットを含めない。比較が良好でも、拡散後の忠実度向上はLinux側のimg2img試験で別途確認する。

## 実測結果（Mac / Blender4.5.9 / EEVEE）

- 最終ベイク: `outputs/lung_transfer_final`。5肺葉、各688〜776pxの画像。肺葉ごとのベイクは約5.8〜7.2秒（一度きり）。中断した旧 `outputs/lung_transfer` は使用しない。
- 同一4姿勢の出力: `outputs/tier1_lung_texture`。比較元は `outputs/tier1_v2_final`。
- **label / depth / geometry の12ファイルすべてSHA完全一致**。ラベル差分0、深度差0mm。
- **肺以外の画素ではRGBも全4枚で完全一致**。最初の1枚には肺が写らず、そのRGB全体も同一。残る3枚の肺画素だけが変わった。
- RGB境界診断4/4通過、出力監査通過、render30 + registration8 = **38テスト通過**。
- 再描画4枚で起動・出力込み **7.24秒**。描画は初回1.92秒、以後0.74〜0.83秒/枚。小バッチの実測でありLinux/2000枚の保証ではない。
- 比較画像: `outputs/lung_texture_comparison.png`（左③、右②、フレーム1/3）。

見た目の変化は主に肺の色むら。局所的な模様の転写は動くが、全術野の質感改善を達成したとは判断しない。肺以外は従来の模様が残り、転写元自体も術中の虚脱肺を測定したアセットではない。高い空間周波数は今回の低解像度ベイクや対応誤差で失われうる。

対応距離p95は肺葉によって正規化bbox空間で約0.18〜0.24と小さくない。これを解剖学的な対応精度の成功とは扱わない。次に比較する場合は、目視で対応を確認してからベイク解像度・部位対応を改善する。拡散後weighted Diceは未検証。既定材質を勝手に置換したり2000枚を再生成したりはしていない。
