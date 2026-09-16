# 学習から除外したファイル

`workspace/fold/v3/folds.csv` から外したファイルの実体をここに退避している。
`data/` は共有ストレージへの symlink で読み取り専用扱いのため, そちらは変更していない。
戻す場合はこのディレクトリから元の場所へ mv すればよい。

| ファイル | 理由 |
|---|---|
| `center_7_case_2_13R.png` | **マスクが破損**。背景 93.7% / 非背景クラスは器具 1 個のみ (全体は中央値 18.0%, 次点 43.7%)。同一画像の `center_7_case_2_9.png` に 15 クラスの完全な注釈があるのでそちらを残す |
| `center_2_case_8_12L_frame_84.png` | `center_2_case_8_12L.png` と画像・マスクとも完全一致 |
| `center_2_case_8_6L_frame_1733.png` | `center_2_case_8_6L.png` と完全一致 |
| `center_3_case_3_na_station_1.png` | `center_3_case_3_12R.png` と完全一致 |
| `center_7_case_3_na_station_1.png` | `center_7_case_3_10L.png` と完全一致 |

## 残したもの

**画像は同一だがマスクが異なる 6 組は両方残している**（注釈のばらつきとして学習させる方針）。
公式指標で測った注釈間一致は Task1 Dice 0.8302 / Task2 Dice 0.8386。

- `center_1_case_10_6R` + `_7R`   (注釈間 macro Dice 0.651)
- `center_1_case_11_6L` + `_7L`   (0.900)
- `center_1_case_12_6R` + `_7R`   (0.547)
- `center_1_case_15_6R` + `_7R`   (0.613)
- `center_1_case_3_11L` + `_13R`  (0.722)
- `center_7_case_3_12L` + `_12R`  (0.751)

なお `na_station_1` は公式アナウンスで「選定フレームが当該 station を写していなかったため extra frame として公開」と説明されており、
テスト入力は `center_<n>_case_<m>_<station>.png` の形式のみと明記されている。
