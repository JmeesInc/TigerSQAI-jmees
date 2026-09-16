# expA23_sweep — アーキテクチャ / loss / 擬似ラベル の横断スイープ

**目的**: 最終提出はアンサンブルになるので、ConvNeXt+Unet++ 以外の「勝てる別系統」を探す。
併せて loss を振り、公式スコアの半分を占める **正規化 Hausdorff** に効く項を見つける。

## ゲートの取り決め（これを崩すと過去の比較が切れる）

- **fold v2 / fold0 / 20 epoch / 1024×576 / bs2×acc4 / AdamW 2e-4 / dice + f2c 0.25**
- 基準 = `expA19 l_384`（convnext_large.fb_in22k_ft_in1k_384 + Unet++）の **fold0 0.6821**
- **1 本の config で 1 軸だけ**動かす。組合せは fold0 を通ったもの同士でのみ後から作る
- `z_seed43` で **fold0 のノイズ幅**を測る。これ無しに ±0.005 を読まない
- val は Dice に加えて **正規化 HD の proxy**（終盤 3ep・1/2 解像度）も出す

## 構成

| ファイル | 役割 |
|---|---|
| `model.py` | 任意の smp アーキ × timm encoder を **encoder 共有 + decoder 2 本**に。3 経路（自前 Unet++ / `create_model` / `smp-hub`） |
| `losses.py` | dice / dicece / dicefocal / **dicedet** / sizeweighted / focaltversky / **diceboundary** / dicermi / dicehd |
| `train.py` | expS03(擬似ラベル) + expA22(解剖ルール) を合流。`summary.json` を吐く。`best_fp16.pt` を書き出す |
| `make_configs.py` | 全 arm の config を生成（**BASE dict が正本**） |
| `sweep.py` | config キューを GPU ワーカに配る。`results/sweep_results.csv` に集約 |
| `smoke_arch.py` | 576×1024/bs2 で構築→fwd→bwd できるかの事前選別（VRAM ピークと s/step も測る） |
| `run_dl2.sh` | dl2 側の起動（`CUDA_DEVICE_ORDER=PCI_BUS_ID` と `WANDB_MODE=disabled` を固定） |

## 実装上の落とし穴（今回踏んだもの）

- **smp の TimmUniversalEncoder は ConvNeXt 等で 1/2 解像度段を C=0 のダミーで埋める**。
  Unet++ と Linknet はそこで落ちる（`Conv2d(out_channels=0)`）→ 自前経路で C=0 段を除外
- **DPT だけ encoder が `(features, prefix_tokens)` を返す**。decoder も 2 引数を要求する
- **smp-hub の head は arch ごとに形が違う**（Sequential / DPTSegmentationHead）。
  クラス数を決めているのは常に最後の Conv2d なので、そこだけ差し替える
- **swin は img_size を渡さないと 576×1024 で window 分割に失敗**。逆に **DPT/ViT に img_size を
  渡すと位置埋め込みの形が変わって事前学習重みの strict load に失敗する**
- **MONAI HausdorffDTLoss は CPU 距離変換で 8 s/step**（1/4 解像度でも）→ 20ep に 12 時間。
  代わりに GT 距離マップを dataset 側で 1/4 解像度に前計算する boundary loss を実装
- **CUDA の既定デバイス順は nvidia-smi と違う**（FASTEST_FIRST）。dl2 で 24GB のつもりが
  8GB に載って OOM した。`CUDA_DEVICE_ORDER=PCI_BUS_ID` を必ず設定する

## 実現性スモーク結果（576×1024, bs2, RTX8000）

| arm | params | peak VRAM | s/step |
|---|---|---|---|
| d_base (unetpp) | 228M | 11.6GB | 0.36 |
| d_unetpp_scse | 231M | 12.1GB | 0.62 |
| d_upernet / d_fpn / d_manet / d_deeplabv3p | 201–339M | 8.3–10.7GB | 0.31–0.39 |
| h_upernet_swin_l | 269M | 14.5GB | 0.77 |
| h_upernet_convnext_l | 270M | 11.4GB | ~0.4 |
| h_segformer_b5 | 88M | 12.9GB | ~0.4 |
| h_dpt_large | 384M | 12.2GB | 1.06 |
| e_caformer_m36 / e_convnextv2_base / e_seresnext101 / e_mitb5 | 65–143M | 8.5–11.2GB | 0.3–0.8 |
| **NG** e_beitv2_large | — | — | TimmUniversalEncoder が等方 ViT 非対応 |
| **NG** e_swinv2_base | — | — | 576×1024 で window 分割が割り切れない（hub 版 swin で代替） |

## 結果

`results/sweep_results.csv` が正本（`python3 sweep.py --collect` で更新）。
スコアは `claudeSummary.md` にも転記する。

## 提出（オフライン）に向けた確認済み事項

- **smp-hub モデルはオフラインで再構築できる**。`m.config` に `_model_class` + encoder 名 +
  decoder 設定が入っており、`smp.create_model(arch=..., encoder_weights=None, **config)` で
  **state_dict のキーが完全一致**するモデルが作れる（407 テンソルで確認）。
  提出コンテナには config.json と自前の fp16 重みだけ入れればよく、ADE20k 重みは不要
- swin は `img_size=(576,1024)` を渡して構築すること（これが無いと window 分割で落ちる）

## ノイズの前提（読み方の取り決め）

expE01 の記録によると、**config 完全同一の 2 本（expA06 vs expA11）が 5fold OOF で +0.0120
(p=0.0005) 離れた**。つまり run 間分散は無視できない。aug も loss も expA19 と数値一致を
確認した `d_base` が A19 l_384 (0.6821) を再現しない場合、それは実装差ではなく **GPU 非決定性
込みの run 間分散**とみなす（`z_seed43` で実測する）。

したがって:
- fold0 の差 **±0.01 未満は単独では信用しない**
- 上位は必ず **fold1（+fold3）で確認**してから 5fold に上げる
- 最終アンサンブルのメンバー選びでは、スコア順だけでなく **系統の多様性**を優先する
  （アンサンブルはノイズに強い）

## 途中知見（fold0 ゲート）

- **UPerNet というアーキが強いのではなく、ADE20k で decoder ごと事前学習された重みが強い**。
  同じ UPerNet でも乱数 decoder（`d_upernet`, convnext_large encoder）は **0.6069** と
  Unet++ (0.6741) に大きく負ける。一方 smp-hub の upernet-swin-large は **0.6921**。
  差は「decoder の事前学習」と「swin encoder」のどちらか → `h_upernet_convnext_l` で切り分く
- **swin は Dice 最高だが HD proxy は最悪**（hd_fine 0.297 vs d_base 0.250）。
  公式スコアは Dice と HD の両方なので、この系統に一本化はしない

## fold0 のノイズ実測（2026-09-12 02:50）

**完全同一レシピ（convnext_large.384 + Unet++ / 20ep / dice+f2c）の 3 サンプル**:

| run | fold0 best |
|---|---|
| expA19 l_384 (seed42, 別実装・同一設定) | 0.6821 |
| expA23 d_base (seed42) | 0.6741 |
| expA23 z_seed43 (seed43) | 0.6691 |

→ **レンジ 0.013 / σ ≈ 0.007**。seed が同じでも GPU 非決定性だけで 0.008 動く。

**読み方の結論**:
- fold0 単独で意味があるのは **±0.015 以上の差**（sizeweighted −0.030 / d_upernet −0.067 は本物）
- swin (+0.010 vs base 平均 0.675)・dicedet (+0.010)・boundary (+0.004) は **fold0 では判定不能** →
  fold1 以降で確認する
- 最終メンバー選びはスコア順だけで決めず、**系統の多様性**を優先する

## a_nohflip（水平反転 aug を切る）

fold0 **0.6851**（base 3 サンプル平均 0.675 に対し +0.010）かつ **HD proxy が全 arm で最良
(fine 0.2459 / coarse 0.2352)**。差はノイズ幅と同程度だが、仮説は原理的:
クラスには L/R の別（L_RLN vs R_RLN, R_Vagal_Nerve, L/R_Inf_Pul_Lig …）があり、
反転画像に元のラベルを付けて学習すると「左右は見た目から決まらない」と教えることになる。
**expA00 以来すべての実験が hflip を使っていた**ので、効くなら全レシピに波及する。
→ fold1 で確認する（GPU2 第 2 陣）。

## 運用上の落とし穴: `pgrep -f` によるジョブ待ちは壊れる

「前のジョブが終わるまで待って次を投入する」チェーンを
`while pgrep -f "sweep.py --gpus 2 --fold 0"; do sleep 60; done` と書いたら、**永久に待ち続けた**。
原因は、そのスクリプトを書き出した **Bash ツール側のラッパープロセスのコマンドラインに
同じ文字列が含まれており、pgrep がそれ自身にマッチしていた**こと。

→ 待ち方は次のどちらかにする:
1. **PID の生存確認**: `while kill -0 "$pid" 2>/dev/null; do sleep 60; done`（最も確実）
2. **GPU メモリ**: `nvidia-smi -i N --query-gpu=memory.used` が 1GB 未満になるまで待つ
   （ただしジョブ間の隙間で誤発火しうるので、同一 GPU に複数チェーンを張らないこと）

実際 dl2 でも GPU メモリ待ちの取りこぼしで二重起動 → OOM を起こしている。

## 落とし穴: venv を activate せずに `python3` を叩くと死ぬ

`CUDA_VISIBLE_DEVICES=1 nohup python3 train.py ...` と直接起動したら
`ImportError: huggingface-hub>=0.34.0,<1.0 is required ... found huggingface-hub==1.30.0`。
CLAUDE.md にあるとおり **user site の transformers が非互換**で、リポジトリ直下の `.venv`
（transformers>=5 を上書き済み）を使う必要がある。Bash ツールの各呼び出しは独立したシェルなので
**前の呼び出しの `source .venv/bin/activate` は効かない**。
→ 単発起動は `../../.venv/bin/python3 train.py ...` とフルパスで叩く
（`sweep.py` 経由なら `sys.executable` を継承するので問題ない）。

## デコーダ軸は決着（fold0）

| decoder（encoder = convnext_large.384 固定） | fold0 |
|---|---|
| Unet++ + scse | 0.6783 |
| **Unet++**（基準） | 0.6741 |
| UPerNet（乱数 decoder） | 0.6069 |
| FPN | 0.5896 |
| MAnet | 0.5173 |

→ **乱数初期化の decoder では Unet++ が最良**。UPerNet/FPN/MAnet は大差で負ける
（この規模のデータでは、skip 接続を密に持つ decoder が有利）。
唯一 Unet++ を超えたのは **ADE20k 事前学習 UPerNet + swin-large (0.6921)**。
そこで「勝っている encoder × 勝っている decoder」= swin-large + Unet++ を追加投入した。

## loss の効果はアーキに転移しない

`diceboundary` は convnext_large+Unet++ では +0.004（0.6785 vs 0.6741）だったが、
**swin+UPerNet(hub) に同じ loss を入れると 0.6380 と −0.054 の大崩れ**（同 lr 1e-4、loss のみ差分）。
→ 「loss 軸の勝者」を別アーキにそのまま持ち込まない。組合せは必ず測る。

## encoder 軸（Unet++ 固定, fold0）

| encoder | fold0 |
|---|---|
| convnext_large.384（基準） | 0.6741 |
| caformer_m36.384 | 0.6446 |
| （expA19 既知）convnextv2_large / v2huge / xlarge @224 | 0.6715 / 0.6703 / 0.6662 |

CAFormer は expA04（SurgeNetXL = CAFormer-S18）と同じく ConvNeXt に届かない。

## swin の勝因は「encoder 単体」ではなく「ADE20k で揃えて学習された encoder+decoder のセット」

| 構成 | fold0 |
|---|---|
| **smp-hub upernet-swin-large（ADE20k 一式）** | **0.6921** |
| swin-large(ImageNet) + Unet++ (`c_swin_unetpp`) | 0.6709 |
| convnext_large + Unet++（基準） | 0.6741 |
| smp-hub upernet-convnext-large（ADE20k 一式） | 0.6320 |
| convnext_large + UPerNet（乱数 decoder） | 0.6069 |

swin を ImageNet 重みで Unet++ に載せても基準と同等（0.6709）。
ADE20k の UPerNet+swin だけが +0.018 抜けている。
→ **セマンティックセグメンテーションで事前学習された「組」をそのまま使うのが効いている**。
（ADE20k 事前学習でも convnext 版は 0.6320 なので、swin との組合せ特有）

## α 後処理は「fine にだけ」効く（2026-09-12）

別スレッドが ens5 で見つけたクラス別スケーリング係数 α を、**dicedet で学習したモデル**に
掛けたときの cross-fit 結果:

| task | fold0 | fold1 | fold2 | fold3 | fold4 | 平均 |
|---|---|---|---|---|---|---|
| fine | +0.0252 | +0.0135 | +0.0203 | +0.0461 | +0.0118 | **+0.0234** |
| coarse | −0.0004 | — | — | — | — | **≈ 0** |

ens5（dicedet なし）では fine +0.0117 / coarse +0.0089 だったので、

- **coarse の後処理ゲインは dicedet が学習中に吸収してしまう**（15 クラスで希少クラスが少なく、
  存在判定を hinge で直しきれる）
- **fine は 30 クラスで希少クラスが多く、後処理の余地が残る**（むしろ ens5 のときより大きい）

→ **提出では α を fine (公式 Task2) にだけ掛ける**。coarse は係数なしの argmax。

半解像度での比較: ens5+α fine 0.6712 に対し **dicedet+α ≈ 0.693**。

## dicedet × nohflip は不採用（4 fold で確定）

| fold | 0 | 1 | 2 | 4 | 平均 |
|---|---|---|---|---|---|
| 差 (+nohflip − dicedet) | +0.0035 | −0.0107 | −0.0089 | −0.0255 | **−0.0104** |

hflip は L/R クラスのラベルを実質ノイズ化し「どちらか決めきれず両方出す」誤りを生む。
dicedet はその誤り（GT に無いクラスの出力）を直接罰するので、**両者は同じ失敗モードを
別経路で直しており加算されない**。dicedet がある状態で hflip を切ると aug の多様性を失う分だけ損。

## アンサンブルの性質（2026-09-13、原寸 Dice で測定）

| 候補 | 構成 | fine | coarse | 平均 |
|---|---|---|---|---|
| B | 7 recipe 35 モデル（弱いメンバー含む） | 0.7007 | 0.6900 | **0.6953** |
| G | 7 recipe 35 モデル（**すべて強い**、最強 +0.0248 含む） | 0.6989 | 0.6897 | 0.6943 |
| E | 9 recipe 45 モデル | 0.7010 | 0.6890 | 0.6950 |
| A | 4 recipe 20 モデル | 0.6925 | 0.6819 | 0.6872 |
| F | **最強 2 recipe** 10 モデル | 0.6840 | 0.6863 | 0.6852 |
| D | 2 recipe 10 モデル | 0.6810 | 0.6680 | 0.6745 |

**この規模ではアンサンブル性能は recipe 数だけで決まり、メンバー個々の強さはほぼ効かない。**
- 2 → 4 → 7 で単調に伸び、7 で飽和（9 にしても変わらず）
- 弱いメンバー 3 つを最強レシピに置き換えても **−0.001**（B vs G）
- 単体最強レシピ 2 つだけで組むと 7 recipe に **−0.010** 負ける（F vs B）

→ 「良いモデルを集める」より「**違う間違い方をするモデルを集める**」が支配的。

## 学習データ量の傾き（全データ学習の是非）

`k_dicedet_rules` を 314 枚(60%) で学習し、通常の 420 枚(80%) と比較（val は同一）:

| fold | 420 枚 | 314 枚 | 差 |
|---|---|---|---|
| 1 | 0.6902 | 0.6774 | **−0.0128** |

+106 枚(+34%) で +0.0128 → **420 → 526 枚(+25%) では +0.008〜0.010 と外挿**。

**結論**: 全データ学習の利得（+0.01 前後）は recipe を 1 つ増やす効果と同程度。
- **置き換えは損**（多様性を失う。候補 F で実証済み）
- **追加は得**（多様性を保ったまま低バイアスのメンバーが増える）
→ 最終構成は **7 recipe × 5fold (35) + 7 recipe × 全データ (7)**。
   全データ側は検証不可だが、35 本が支配するので壊れていても安全side に倒れる。

## 提出コンテナの設計（メモリ安全化）

35 モデルを同時に GPU へ載せると **fp16 でも 20.6GB** になり、24GB 級の評価環境では
4K 画像の活性化と合わせて OOM する（実際にホストテストで再現）。
→ **メンバー外側・画像内側のループ**に組み替え、1 モデルずつ載せ替える。
   確率は **モデル解像度のまま CPU に累積**（原寸で貯めると 4K 1 枚 31ch fp16 で 512MB）。
   ピーク GPU メモリ **20.6GB → 約 6GB**。
   副次的に、解像度の違うメンバーを混ぜられるようにもなった。
