# expA09_lymph_aux

**expA06 + リンパ節に絞った Task3 補助タスク**（ユーザー TODO「lymph node だけ残す Task3 同時学習」）。

## 動機

- **Lymph_Node は最大のボトルネック**: weight=3 かつ Dice 0.32（expA06 時点）。
  加重指標への寄与が最も大きいのに最も低い
- Task3 のラベル（14 ステーションの可視性）は「どこにリンパ節が見えるか」の画像レベル情報 =
  Lymph_Node セグメンテーションの弱教師になりうる
- expT02（14 クラス分類ヘッドを encoder 特徴から生やす全面 MTL）は seg −0.032 / cls −0.02 で失敗。
  原因は **BCE 勾配が backbone 全体を乱すこと** → **経路を Lymph_Node チャネルだけに絞る**

## 構成

- ベースは expA06 と完全同一（MaxViT + dual Unet++ + 強aug + f2c loss）
- 追加（学習時のみ）:
  - fine logits の **Lymph_Node チャネル (id=19) の softmax 確率マップ**を空間集約 →
    `[全体平均, 最大値, 左半分平均, 右半分平均]` の 4 特徴（L/R station を区別できるよう左右分割）
  - + station one-hot(15) → MLP(64) → 14 sigmoid、BCE（Task3 ラベル無しの 8 枚はマスク）
  - `loss += 0.15 × L_lymph`
- **encoder には直接勾配が流れず、Lymph_Node チャネル経由でのみ伝播する**のが expT02 との決定的差
- `lymph_head` は checkpoint から除外 → 保存 ckpt は expA06 と同形式（提出コード流用可）
- fold v2（526 枚。Task3 GT は 518 枚）

## プロトコル

- 5-fold → OOF 公式評価。比較対象 expA06: T1 0.6663/0.2646, T2 0.6530/0.2574
- 注目点は全体スコアだけでなく **per-class の Lymph_Node Dice（0.3173 → ?）**

## 結果

（学習後に記入）
