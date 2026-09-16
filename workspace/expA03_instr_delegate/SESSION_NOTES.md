# expA03_instr_delegate

**学習なし**の後処理 ablation: STIR の器具セグメンテーションモデルで Instrument クラスを上書きする。

## 構成

- モデル: `../STIR/reference/stitch_track/weights/convnext-unet-best.pth`
  - `smp.Unet(encoder=tu-convnext_base.dinov3_lvd1689m, classes=1, sigmoid)`（binary tool mask）
  - ⚠️ encoder の事前学習は **DINOv3 (LVD-1689M = Meta の非公開データ)** 由来。NS 重みと同種のルールリスク。また **この ckpt 自体の学習データの出所も要確認**（公開データのみか）→ 採用するなら write-up 前に確認必須
- 前処理は STIR `tracker.py` と同一: /255 → 512×512 → imagenet norm → sigmoid → 元解像度へ bilinear → thr 0.5
- expA00 の OOF 予測 PNG の tool 画素を Instrument 色 (184,61,245) に置換（task1/task2 とも同色）→ 公式コードで採点

## Ablation プロトコル

- fold0 (105枚) で override 前後を比較
- 比較対象: expA00 fold0 = **task1 Dice 0.5973 / HD 0.3358, task2 Dice 0.5731 / HD 0.3515**
- 効けば: thr の掃引 / 「器具画素をマスクして学習からも除外し器具は完全委譲」の学習実験へ発展

## 結果（fold0, thr=0.5, 2026-08-22）

| task | Dice (base→override) | HD (base→override) |
|------|---------------------|--------------------|
| task1 | 0.5973 → **0.5986** (+0.0013) | 0.3358 → **0.3347** (−0.0011) |
| task2 | 0.5731 → **0.5747** (+0.0016) | 0.3515 → **0.3504** (−0.0011) |

**判断: 一貫してプラスだが効果は微小（+0.001〜0.002）**。ベースラインが既に器具を良く取れている（エラー分析どおり）ため上書きの伸び代が小さい。DINOv3 の
ルールリスクを取ってまで採用する価値は現状なし。**不採用**。ただし発展形（tool mask を
入力チャネルに足す / tool 上の Lymph_Node FP 抑制に使う）は後日検討の余地あり。
