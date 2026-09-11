# Tier 1 appearance v2: 方式③を採用

共有コミット `383c5da` を取り込み、実測色度＋臓器別プロシージャル材質を採用した。EEVEE、既存のクラスID・頂点・面・カメラ・剥離窓・脂肪シェル・登録は維持する。図譜の色を移植したり、外部臓器の形状へ置換したりしない。

## 選択理由

|方式|判断|理由|
|---|---|---|
|① アトラス材質移植|今回は不採用|確認した対象臓器にUVがなく、材質はノードグループ・属性への依存がある。材質名の存在は術野の見た目の適合を保証しない。移植の工数より共有色度を直接利用する効果を優先。|
|② 外部PBRテクスチャ|不採用|追加のライセンス確認・対応付け・投影調整を締切前に増やさない。候補のCC0等の記載を未検証のまま採用しない。|
|③ 実測色度＋臓器別材質|採用|既存形状と同じ画素対応を保ったまま、灰緑の色域・等方ノイズだけの材質を改善できる。|

アトラス監査はSHA `9f08a17e…35afcd`、Blender4.5.9、embedded scripts無効。ロード後のbpy.dataは188 Material/3 Image/0 Texture。提供された428/133/169とは異なるので**集計対象・方法は要確認**。右肺下葉・食道・気管・奇静脈・胸大動脈・右下肺静脈でUV層なし、材質直下のImage Textureノードなし（ノードグループ内部の画像依存まで否定する結果ではない）。再現用 `scripts/audit_atlas_materials.py` を追加。

## 材質

- `assets/real_class_chroma.yaml` は共有JSONからの集計コピー。実画像・マスクは読まない。
- `configs/materials.yaml` の `base_chroma_rgb` に実測r/g/bを入れる。丸め誤差だけ合計1へ正規化する。RGB比を色として指定し、共通の明度初期値からsRGB→linear変換してBSDFに入力する。
- `target_saturation` は独立した比較対象として保存する。RGB色度を固定すると彩度も決まるため、中央値同士が整合しない場合に彩度だけを強制して色度を壊さない。
- 欠測の13/21/22/27は共通の暫定色度で、クラス別実測値を捏造しない。`chroma_status` に区別を残す。
- 赤い血管柄はseed固定の分岐木を手続き的に作る。太い幹から細い枝へ分岐し、周期コピーしたgrayscale textureを3方向からbox投影する。**血管走行の解剖モデルではなく表面模様**。外部の画像テクスチャは使わない。
- 血管柄の色は基底色度よりrを0.08増やし、G/Bの比を保って減らす。柄の混合強度は共有vessel_patternの単調な初期変換。実画像の高周波r指標がそのままBSDFの係数になるとはみなさない。
- 器具・肺は血管柄0。血液の高周波rは血管網とは解釈せず、液面のノイズ/色むらとして扱う。血液の高specular・低roughnessは後続の材質v2指示による見た目の暫定overrideであり、前回測定のハイライト率から同定された物性ではない。
- 脂肪は1〜3mm程度のVoronoi小葉、リンパ節は粒状、肺は血管網を伴わない弱い小葉境界、胸膜は弱い低周波変動。
- 気管支の輪はmeshの主軸に直交する周期柄、血管の筋は主軸方向へ伸びる柄。主軸は既存頂点からPCAで計算し、スカラー属性として補間する。曲がった管全体の中心線/厳密な円周UVではない局所的近似。
- 模様はalbedoとbumpの両方へ反映。smooth shadingも使用するが頂点/面/displacementは変更しない。見た目の法線のみを変える。

色度は理想的な一様乗算では明るさに不変だが、実際のtone mapping、sRGB変換、白い鏡面反射、クリッピングでは変わる。画像上の実測統計からalbedoを一意に逆算したとは主張しない。`render.appearance_report` で**描画後**の色度・彩度を比較する。

## 利用

```bash
# 共有JSONから色度/柄強度を再反映する場合
python -m render.configure_chroma \
  --input assets_local/real_class_chroma.json \
  --materials configs/materials.yaml --output configs/materials.yaml

# 新規生成
python -m render.generate --frames 128 --emit-rgb --allow-provisional \
  --materials configs/materials.yaml --output outputs/tier1_v2 \
  --blender /path/to/blender --workers 4

# ユーザー採用済み: 旧2000枚のcameraからEEVEEの3点を揃える
python -m render.replay_rgb outputs/pretrain_2k \
  --output outputs/pretrain_2k_tier1_v2 --rerender-triplet \
  --materials configs/materials.yaml --blender /path/to/blender --workers 4

python -m render.alignment outputs/pretrain_2k_tier1_v2 --require-edge-evidence
python -m render.appearance_report outputs/pretrain_2k_tier1_v2
```

旧Cyclesのlabelと新EEVEEのRGBを混在させない。`383c5da`内の詳細指示でユーザーが `--rerender-triplet` を採用済みと確認した。EEVEE製の既存バッチを材質だけ変える場合はこのフラグを外せば元label/depthとの厳密比較が有効になる。元2000枚自体はMacに存在しないため、このバッチの全件処理はLinux側で実行する。

## 来歴

追加の外部アセット・モデル重みは使わない。分岐画像はこのリポジトリのアルゴリズムが生成するマスクで、第三者PBRマップではない。既存アトラスの来歴は維持: Z-Anatomy / BodyParts3D、CC-BY-SA-4.0、出典 https://github.com/Z-Anatomy/Models-of-human-anatomy 。商用利用可、改変アセットを再配布する場合は帰属・変更表示・同一ライセンス等が必要。外部候補のライセンスを確認済みとは主張しない。


## 最終検証（Mac、Blender4.5.9、EEVEE）

同じ1024×576の4カメラを材質だけ変えて再出力。label4枚・depth4枚・保存geometry4個すべてのSHA-256が元のEEVEEバッチと一致。境界診断4/4通過、境界/内部RGBコントラスト比12.59〜17.53。33単体テスト（render25＋registration8）と出力監査通過。

描画された16クラスの「クラス別絶対差の単純平均」で比較した（クラスサイズによる重み付けなし）。実側は共有集計、合成側はフレームごとの画素中央値をさらに中央値で集計。実側の集計方式と完全同一かは未確認。

|指標|材質v1|材質v2|
|---|---:|---:|
|r色度の平均絶対差|0.17035|0.02145|
|HSV彩度の平均絶対差|0.42408|0.03808|

4枚の少数診断であり、色度の完全一致や拡散後weighted Diceの改善を証明しない。新しい色・材質の効果はLinux側のimg2img/fidelity評価で確認する。クラスによっては鏡面や暗部の影響で色度が目標より外れる。

再出力の全体時間16.69秒（1 worker、4.17秒/枚、起動・shader compile・出力保存込み）。最後2枚のrender処理は0.871秒/0.986秒。初回の材質compileが大きいので、この4枚から2000枚の時間を直接外挿しない。Linuxでの材質v2性能は未測定。

使用した試験バッチは `outputs/tier1_v2_final`、比較元は `outputs/tier1_final1024`。生成RGBや個別姿勢はGitへ含めず、コード・設定・共有集計・検証記録のみを同期する。
