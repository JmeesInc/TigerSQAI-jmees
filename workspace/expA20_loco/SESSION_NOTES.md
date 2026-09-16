# expA20_loco — Leave-One-Center-Out で「未知センター」の汎化ギャップを測る

## 動機

テストセットは **center_5 が train から除外された未知センター**を含むとみられる
（ローカルは center_1,2,3,4,6,7 の 6 センターのみ）。
一方、これまでの CV（`fold/v2`）は **center を stratify** しているので全 fold に全センターが
混ざっており、「見たことのないセンターでどれだけ落ちるか」を測れていない。
CV は良いのに LB が悪い、という典型的な失敗をこの実験で事前に検知する。

## 設計

- レシピは expA17（ConvNeXt-base in22k + Unet++ / 20ep）と**完全同一**。fold 定義だけ差し替え
- `fold/loco/folds.csv`: fold0=center_1 … fold5=center_7（1 センター丸ごとを val）
- 1 case は 1 center にしか属さないので、center 分割は自動的に case 分割でもある（リークなし）

## 読み方

**同一レシピ・同一データで、v2（center 混在 5-fold）と LOCO（center 除外 6-fold）の
スコア差 = 未知センターでの劣化量**。expA17 の v2 5fold 平均 = 0.6593 が比較基準。

センターごとの val 枚数が 38〜220 と偏るので、単純平均ではなく
**枚数（あるいは case 数）で重み付けした値**も併記して判断すること。
