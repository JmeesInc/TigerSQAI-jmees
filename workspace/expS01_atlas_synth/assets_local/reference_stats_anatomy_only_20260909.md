# 【最終版】実データ基準統計 — 解剖のみ正規化（2026-09-09）

## 方針（ユーザー決定・確定）
1. **内視鏡の円形視野マスクは生成しない** → 学習時 augmentation で付与
2. **Instrument(1) と Other(2) は生成しない** → こちらで器具貼り付け（expA02 の
   SAR-RARP50 / SurgToolLoc cutout 資産を流用）

→ 比較の分母は **背景(0)・器具(1)・Other(2) を除いた解剖画素**とする。本表が最終基準。
`astra_calibration_pack_20260908.md` §2、`reference_stats_foreground_normalized_20260909.md`、
round3 の `acceptance.yaml` はすべてこれで置き換える。

## なぜ分母を変えるか
実データでは器具が前景の 7.7% を占め、その分だけ解剖の見かけの比率が下がっている。
合成で器具を描かないなら、**実側からも器具・Other を分母から外さないと解剖の比率が一致しない**。

## 新基準表（実 528 枚、クラス 3–30 のみ）

|ID|Class|出現%|面積%（解剖のみ正規化）|
|---:|---|---:|---:|
|3|Trachea|58.0|4.08|
|4|Right main bronchus|54.5|2.88|
|5|Left main bronchus|60.4|2.91|
|6|Esophagus|85.6|10.79|
|7|Fatty tissue esophagus|55.9|4.71|
|8|R inferior pulmonary ligament|25.9|0.80|
|9|L inferior pulmonary ligament|27.1|0.33|
|10|**Pleura**|93.2|**22.40**|
|11|Pericardium|70.8|6.08|
|12|Inferior pulmonary vein|58.3|4.47|
|13|R subclavian artery|2.5|0.10|
|14|R vagal nerve|40.7|1.09|
|15|Aorta|72.3|6.22|
|16|Azygos vein|69.5|2.37|
|17|Superior caval vein|11.6|0.50|
|18|**Lung**|71.0|**12.44**|
|19|Lymph node|65.3|3.44|
|20|Fatty tissue|84.1|4.07|
|21|L subclavian artery|0.2|0.01|
|22|R bronchial artery|0.0|0.00|
|23|Pulmonary artery|7.0|0.15|
|24|Pool of blood|53.6|1.49|
|25|Resection area|80.9|4.96|
|26|Gastric conduit|9.5|1.50|
|27|R recurrent laryngeal nerve|4.0|0.03|
|28|L recurrent laryngeal nerve|15.3|0.27|
|29|Omentum|11.7|1.57|
|30|Thoracic duct|17.6|0.17|

## 受入条件（最終版）

|指標|目標（実データ）|round3 実測|
|---|---:|---:|
|出現率の絶対差 合計（3–30）|0|**502.8 pp** ← 新 baseline|
|面積の絶対差 合計（3–30, 解剖正規化）|0|**47.2 pp** ← 新 baseline|
|Pleura 面積|**22.40%**|19.29%|
|Lung 面積|**12.44%**|14.78%|
|解剖クラス数 中央値|**12**（p10 8 / p90 15）|14（12/17）|
|背景(空白) 中央値|**0〜2%**|26.75%|

- 器具・Other を外したことで出現率誤差は 632.8 → **502.8 pp**（130.0pp が対象外化）
- **解剖クラス数の目標は 12**（背景・器具・Other を除く数え方）。合成は 14 でまだ多い
- Pleura / Lung / Esophagus / Resection area は既にかなり近い。
  残る大物は **IPV 面積 4.47 → 0.26**、**Fatty tissue esophagus 4.71 → 0.80**、
  **Pericardium 6.08 → 10.01（過剰）**、**SVC 0.50 → 6.23（過剰）**

## レンダラ側の簡略化（Astra への依頼）

器具と Other の生成を**削除してよい**。副次的な利点:
- 「instrument crosses anatomy」による棄却が消え、**生成が速くなる**
- 器具の到達空間による**カメラ分布のバイアスが1つ減る**
- 「Instrument 35.9% vs 90.2%」という解決困難だった差分が消える
- Pool of blood(24) と Resection area(25) は**引き続きレンダラ側で生成する**
  （貼り付けでは扱いにくいため）。Resection area は既に 4.90 vs 4.96 とほぼ一致している

較正レポートは **クラス 3–30 のみ**を集計対象とし、1/2 は「対象外」と明示すること。
