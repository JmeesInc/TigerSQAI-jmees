# Round 4 anatomy-only calibration

N=128; IDs 0/1/2: 対象外。面積はフレームごとにID 3–30画素で割り、全フレームで平均。背景率のみ別途全画面で測定。

|ID|Class|出現 実%|合成%|差 pp|解剖面積 実%|合成%|差 pp|
|---:|---|---:|---:|---:|---:|---:|---:|
|3|Trachea|58.00|42.19|-15.81|4.08|3.01|-1.07|
|4|Right main bronchus|54.50|46.09|-8.41|2.88|2.63|-0.25|
|5|Left main bronchus|60.40|30.47|-29.93|2.91|0.05|-2.86|
|6|Esophagus|85.60|91.41|+5.81|10.79|7.01|-3.78|
|7|Fatty tissue esophagus|55.90|50.00|-5.90|4.71|1.10|-3.61|
|8|Right inferior pulmonary ligament|25.90|17.97|-7.93|0.80|0.84|+0.04|
|9|Left inferior pulmonary ligament|27.10|20.31|-6.79|0.33|0.14|-0.19|
|10|Pleura|93.20|100.00|+6.80|22.40|38.39|+15.99|
|11|Pericardium|70.80|89.06|+18.26|6.08|10.28|+4.20|
|12|Inferior pulmonary vein|58.30|17.19|-41.11|4.47|0.26|-4.21|
|13|Right subclavian artery|2.50|0.00|-2.50|0.10|0.00|-0.10|
|14|Right vagal nerve|40.70|85.94|+45.24|1.09|0.48|-0.61|
|15|Aorta|72.30|99.22|+26.92|6.22|6.91|+0.69|
|16|Azygos vein|69.50|96.88|+27.38|2.37|8.11|+5.74|
|17|Superior caval vein|11.60|45.31|+33.71|0.50|4.07|+3.57|
|18|Lung|71.00|69.53|-1.47|12.44|4.24|-8.20|
|19|Lymph node|65.30|79.69|+14.39|3.44|0.69|-2.75|
|20|Fatty tissue|84.10|97.66|+13.56|4.07|3.14|-0.93|
|21|Left subclavian artery|0.20|0.00|-0.20|0.01|0.00|-0.01|
|22|Right bronchial artery|0.00|0.00|+0.00|0.00|0.00|+0.00|
|23|Pulmonary artery|7.00|42.97|+35.97|0.15|4.08|+3.93|
|24|Pool of blood|53.60|1.56|-52.04|1.49|0.01|-1.48|
|25|Resection area|80.90|77.34|-3.56|4.96|4.54|-0.42|
|26|Gastric conduit|9.50|0.00|-9.50|1.50|0.00|-1.50|
|27|Right recurrent laryngeal nerve|4.00|0.00|-4.00|0.03|0.00|-0.03|
|28|Left recurrent laryngeal nerve|15.30|0.00|-15.30|0.27|0.00|-0.27|
|29|Omentum|11.70|0.00|-11.70|1.57|0.00|-1.57|
|30|Thoracic duct|17.60|0.00|-17.60|0.17|0.00|-0.17|

解剖クラス数: {'p10': 9.0, 'median': 12.0, 'p90': 15.0}; 目標: {'p10': 8, 'median': 12, 'p90': 15}
空の解剖フレーム: 0

Status: **failed**; bank allowed: **False**

|Metric|Baseline|Target|Synthetic|Pass|
|---|---:|---|---:|---|
|presence_absolute_error_sum_pp|502.80|0.0|461.77|True|
|area_absolute_error_sum_pp|47.20|0.0|68.16|False|
|pleura_mean_area_pct|19.29|22.4|38.39|False|
|lung_mean_area_pct|14.78|12.44|4.24|False|
|visible_class_count_median|14.00|12|12.00|True|
|background_median_pct|26.75|[0, 2]|0.00|True|

Required IDs absent: []
