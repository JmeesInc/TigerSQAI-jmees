# expG02_synthpretrain — 合成データによる事前学習の効果検証

## 動機（2026-09-10、ユーザー指摘）
expG01 の忠実度指標（合成条件 0.157）を根拠に「学習データとして使えない」と判断したのは早計。
**fine-tune と違い事前学習はラベルノイズ耐性が高く、量で効く**という指摘を受けて設計。

さらに、expG01 の指標には**交絡がある**:
合成条件で Dice が低い原因は (a) 生成器が条件に従っていない、
(b) 審判の expA06 が実データ学習のため合成由来の画像を読めない、の 2 つがあり得るが切り分けていない。
**事前学習の A/B を直接回せば (a)/(b) の切り分けは不要で答えが出る。**

## 設計
```
合成ラベル (expS01 Blender) → ControlNet v2 で画像化 → (image, fine, coarse) の合成データセット
  → expA06 と同一アーキ・同一レシピで事前学習 (15 epoch)
  → 実データ fold0 で fine-tune（重みのみ引き継ぎ、optimizer/epoch は引き継がない）
  → 対照: 事前学習なし・同一 config・同一 seed で fold0 学習
```
- コードは expA06 のコピー + `--init-from`（model 重みのみロード、strict=False で
  `model.` 接頭辞の欠損を assert）
- coarse ラベルは公式 labelmap の `fine_id → merged_id` で生成
- 合成側の case_id は 14 枚ごとの疑似 case（実 case 構造が無いため）

## 実行
```bash
workspace/expG02_synthpretrain/run.sh prepare <生成画像dir> <合成ラベルdir>
workspace/expG02_synthpretrain/run.sh pretrain
workspace/expG02_synthpretrain/run.sh finetune
workspace/expG02_synthpretrain/run.sh baseline
```

## 判定
fold0 の val score で **finetune vs baseline** を比較。
公式指標での比較は `predict_oof.py` 相当を fold0 に対して回す。

## 状態
- 合成ラベルマップ 2000 枚を生成中（1024×576, CPU 24 並列）
- 画像化は GPU の空き待ち（現在 4 枚とも他スレッドが使用中）
