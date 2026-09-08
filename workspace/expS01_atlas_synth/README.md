# TigerSQAI expS01 — 解剖アトラス由来 合成データパイプライン

MICCAI2026 EndoVis TIGER SQ-AI Challenge 用の合成ラベルマップ生成（Tier 0）。
本体リポジトリ（private）の `workspace/expS01_atlas_synth/` を切り出した進捗共有用リポジトリ。

- 設計・経緯: `SESSION_NOTES.md` / 実行方法: `render/README.md`
- Astra への依頼プロンプト: `PROMPTS_FOR_ASTRA.md`、較正情報: `assets_local/`

## このリポジトリにコミットしてはいけないもの（厳守）

チャレンジデータはチーム外共有禁止のため:
- 実データ（画像・マスク・CSV）とその**フレーム単位の派生物すべて**
  （タスクCで生成する実フレームの推定姿勢 `poses/*.json` を含む → .gitignore 済み）
- 共有可能なのは**集計統計のみ**（`assets_local/astra_calibration_pack_20260908.md` §2 の類）

大容量物も対象外: `outputs/`（生成物はローカル保持）、Z-Anatomy の .blend 本体（各自配置、SHA 照合）。
