# TASK: v13 = v12 + 全データ seed 違いを 6 枠に 1 本ずつ追加（別スレッド用の手順書）

作成 2026-09-16 16:25 JST。**締切 2026-09-16 20:59 JST（9/15 23:59 AoE）**。
ゴール: v12 の Task 1/2 アンサンブル（7 レシピ枠 × 25 checkpoint）のうち、シードが 1 本しかない 6 枠に
全データ学習のシード違いを 1 本ずつ足した **v13** を build → test → push する。
**間に合わなければ v12 のまま**。v12 を壊す作業は一切しない。

## 0. 前提・現状（触る前に読む）

- 最終候補 **v12** は dl2 上でビルド済み（`tigersqai26_shunsuke:v12` = `docker.synapse.org/syn77311180/tigersqai26_shunsuke:v12`、30.2 GB、回帰テスト合格）。
  **push は未実施（ユーザーが手動）**。この作業と無関係に、まず v12 を push してもらう。
- v12 の中身: seg = `submit/v006_a23/model_v11/`（7 枠 25 本、`config.json` の `group` が枠、`tta` が反転 TTA）、
  Task3 = `model_t3_v12/` + `model_t3enc_v12/`。`Dockerfile` は `COPY model_v11/ /app/model/`。
- 枠と本数（`model_v11/priority.json` 順）: `q_endovis18_dlv3` 17 本、`s_kdr_seed43`（= k_dicedet_rules）3 本、
  `r_xl` / `r_ft_fine` / `q_both_dlv3` / `l_dicedet` / `r_nohflip` 各 1 本。
- 推論側 (`process.py`) は **group 内で 1/k 平均 → group 間で等重み**。seed を足しても枠の重みは変わらない。
  予算制御は `priority.json` の後ろから削るので、新規 seed は **priority の末尾に置く**（削られるのは新規 seed から）。
- 学習は dl1（RTX 8000 ×4、全部空き）。docker の build/test は **dl2** のみ（dl1 は GPU docker 不可）。
  dl2 からはリポジトリが `<repo, as mounted on the build host>` (NFS) に見える。
- Python は必ずリポジトリ直下の `.venv`（`.venv/bin/python3`）。`workspace/expA23_sweep/worker_run.sh` は自動で選ぶ。
- **/data4 は残り 21 GB（100%）**。学習 1 本は途中で best.ckpt + latest.ckpt ≈ 11 GB を持つ。**先に §1 で空ける**。

## 1. ディスクを空ける（必須、5 分）

```bash
cd <repo>/submit/v006_a23
# 提出は docker push なので docker save の tar.gz は一切不要。v10 の save 29 GB と v11 の save（hardlink pair、実体 29 GB）を削除
rm -f tigersqai_shunsuke_task123_v10.tar.gz tigersqai_Jmees_task123.tar.gz tigersqai_shunsuke_task123.tar.gz
# 書き出し済み run の生 ckpt（best.ckpt / latest*.ckpt、計 138 GB）。fp16 の best_fp16.pt / last_fp16.pt が同じ fold dir にあるものだけ消す
cd ../../workspace/expA23_sweep
for c in $(find results -name "*.ckpt" -size +1G); do
  d=$(dirname "$c"); [ -f "$d/best_fp16.pt" ] || continue
  case "$d" in *full_*) [ -f "$d/last_fp16.pt" ] || continue;; esac   # 全データ run は最終 epoch (last_fp16.pt) が無いと export が best に化けるので残す
  rm -v "$c"
done
df -h /data4
```
目標: **60 GB 以上空ける**（6 本同時 × 11 GB）。tar.gz 2 種で 58 GB、生 ckpt で最大 138 GB 戻る。
以後 `export_image.sh`（docker save）は実行しない。

## 2. 学習 config を作る（5 分）

`workspace/expA23_sweep/configs/` にコピーして `experiment.name` と `experiment.seed` だけ変える（`use_all_data: true` はそのまま）。
既に存在するものはそのまま使う。

| 枠 (group) | 元 config | 新 config | 備考 |
|---|---|---|---|
| r_xl | `expA23_full_r_xl.yaml` | `expA23_full_r_xl_s43.yaml` (seed 43) | ConvNeXt-XL、他より 1.5 倍遅い。**GPU 単独で**回す |
| q_both_dlv3 | `expA23_full_q_both_dlv3.yaml` | `expA23_full_q_both_dlv3_s43.yaml` (seed 43) | |
| l_dicedet | — | **`expA23_full_l_dicedet_s43.yaml` は既存** | そのまま投入 |
| r_nohflip | `expA23_full_r_nohflip.yaml` | `expA23_full_r_nohflip_s43.yaml` (seed 43) | |
| s_kdr_seed43 | — | **`expA23_full_k_dicedet_rules_s44.yaml` は既存** | 枠は `s_kdr_seed43` |
| r_ft_fine | `expA23_full_r_ft_fine.yaml` | `expA23_full_r_ft_fine_s43.yaml` (seed 43) | `model.init_from: expA23_full_q_cholec_dlv3_s43` に変える（seed 43 の基底から 8 epoch 追学習。基底は `results/expA23_full_q_cholec_dlv3_s43/` に存在） |

```bash
cd <repo>/workspace/expA23_sweep/configs
for r in r_xl q_both_dlv3 r_nohflip r_ft_fine; do
  sed -e "s/^  name: expA23_full_${r}$/  name: expA23_full_${r}_s43/" -e "s/^  seed: 42$/  seed: 43/" expA23_full_${r}.yaml > expA23_full_${r}_s43.yaml
done
sed -i "s/init_from: expA23_full_q_cholec_dlv3 /init_from: expA23_full_q_cholec_dlv3_s43 /" expA23_full_r_ft_fine_s43.yaml
grep -n "name:\|seed:\|init_from" expA23_full_{r_xl,q_both_dlv3,r_nohflip,r_ft_fine}_s43.yaml expA23_full_l_dicedet_s43.yaml expA23_full_k_dicedet_rules_s44.yaml
```
`results/expA23_full_<name>/` が既に存在する config は投入しない（既に学習済み = そのまま §4 へ）。

## 3. 投入（daemon キュー）

キュー行は `<fold>:<config名>`。全データ学習は `0:` で投入する（`use_all_data: true` なので fold は無視される）。
daemon は GPU ごとに `queue_backlog.txt` から 1 行ずつ取る。`MAX_USED_MB=30000` なので 1 GPU に 2 本まで同居する。

```bash
cd <repo>/workspace/expA23_sweep
rm -f daemon.stop
# r_xl は重いので先頭（先に取られて単独時間が長い）。6 行
cat >> queue_backlog.txt <<'EOF'
0:expA23_full_r_xl_s43
0:expA23_full_q_both_dlv3_s43
0:expA23_full_l_dicedet_s43
0:expA23_full_r_nohflip_s43
0:expA23_full_k_dicedet_rules_s44
0:expA23_full_r_ft_fine_s43
EOF
./refill.sh 0        # daemon 4 基が死んでいれば再起動。状態表示
```
- 進捗: `tail -2 daemon_gpu*.log`、`nvidia-smi`、`logs/expA23_full_<name>_fold0.log`。
- 完了の目印: `results/expA23_full_<name>/fold0/last_fp16.pt` と `summary.json` ができる。
- 所要: 2 本同居で **約 85 分/本**、`r_xl` 単独で **90〜120 分**、`r_ft_fine` は 8 epoch なので **30 分**。
  16:40 投入 → **18:15〜18:45 に全完了**の見込み。**19:00 までに終わらないものは捨てる**（あるものだけで v13 を作る）。
- 二重起動注意: `pgrep -f "train.py --fold"` で本数確認。OOM したら `refill.sh` が 1 回だけ再投入する。

## 4. 書き出し（fp16、最終 epoch）

```bash
cd <repo>
.venv/bin/python3 submit/v006_a23/export.py --prefer-latest --folds 0 --out submit/v006_a23/model_v13_new \
  --members A23:expA23_full_r_xl_s43 A23:expA23_full_q_both_dlv3_s43 A23:expA23_full_l_dicedet_s43 \
           A23:expA23_full_r_nohflip_s43 A23:expA23_full_k_dicedet_rules_s44 A23:expA23_full_r_ft_fine_s43
```
出力 `model_v13_new/<name>/{config.json, fold0.pt}`。ログに `last_fp16.pt (epoch 19)`（r_ft_fine は epoch 7）と出ることを確認。
完走していないメンバーは `--members` から外す。

## 5. model_v13 を組む（v11 の hardlink + 新規 6 本、group / tta / priority を付ける）

```bash
cd <repo>/submit/v006_a23
PY=../../.venv/bin/python3
$PY - <<'EOF'
import json, os, shutil
src, new, dst = "model_v11", "model_v13_new", "model_v13"
# 新規メンバー -> (group, tta)。tta は枠の既存メンバーと同じにする（r_xl / r_nohflip は False）
SPEC = {
  "full_r_xl_s43": ("r_xl", False), "full_q_both_dlv3_s43": ("q_both_dlv3", True),
  "full_l_dicedet_s43": ("l_dicedet", True), "full_r_nohflip_s43": ("r_nohflip", False),
  "full_k_dicedet_rules_s44": ("s_kdr_seed43", True), "full_r_ft_fine_s43": ("r_ft_fine", True),
}
if os.path.isdir(dst): shutil.rmtree(dst)
shutil.copytree(src, dst, copy_function=os.link)          # 既存 25 本は hardlink（容量ゼロ）
order = json.load(open(f"{dst}/priority.json"))
for name, (group, tta) in SPEC.items():
    if not os.path.isdir(f"{new}/{name}"): print("skip (not exported):", name); continue
    os.makedirs(f"{dst}/{name}"); os.link(f"{new}/{name}/fold0.pt", f"{dst}/{name}/fold0.pt")
    c = json.load(open(f"{new}/{name}/config.json")); c["group"] = group; c["tta"] = tta
    json.dump(c, open(f"{dst}/{name}/config.json", "w"), indent=2)
    order.append(name)                                     # 末尾 = 予算で最初に削られる
json.dump(order, open(f"{dst}/priority.json", "w"), indent=1)
groups = {}
for m in order:
    g = json.load(open(f"{dst}/{m}/config.json"))["group"]; groups[g] = groups.get(g, 0) + 1
print(len(order), "checkpoints;", groups)
EOF
```
期待: `31 checkpoints; {'q_endovis18_dlv3': 17, 'r_xl': 2, 'r_ft_fine': 2, 'q_both_dlv3': 2, 's_kdr_seed43': 4, 'l_dicedet': 2, 'r_nohflip': 2}`（完走分だけ +1）。
**group 名が既存の枠名と 1 字でも違うと別枠になって重みが増える**ので、上の出力で枠数が 7 のままであることを必ず確認。

## 6. Dockerfile / .dockerignore を v13 に

```bash
cd <repo>/submit/v006_a23
cp Dockerfile Dockerfile.v12.bak
sed -i 's#^COPY model_v11/ /app/model/#COPY model_v13/ /app/model/#' Dockerfile
sed -i 's#^\# v011:.*#\# v013: v12 の seg 7 枠 25 本 + 6 枠に全データ seed 違いを 1 本ずつ（枠内平均、priority 末尾）#' Dockerfile
grep -q "^model_v11/" .dockerignore || echo "model_v11/" >> .dockerignore
grep -q "^model_v13_new/" .dockerignore || echo "model_v13_new/" >> .dockerignore
cat Dockerfile; cat .dockerignore
```
Task3 の行（`model_t3_v12/`, `model_t3enc_v12/`）と `process.py` は **変更しない**。

## 7. ビルドと回帰テスト（dl2、約 30 分）

```bash
ssh dl2
cd <repo, as mounted on the build host>/submit/v006_a23
VER=v13 ./build.sh 2>&1 | tail -5          # コンテキスト転送 ~4 分 + build ~20 分。「built tigersqai26_shunsuke:v13」
VER=v13 ./test_v12.sh 6 2>&1 | tail -25    # RTX 4090、--network none、4K 1 枚含む 6 枚
```
合格条件（test_v12.sh が assert）: ファイル数・命名・解像度・RGB 色・task1/2 取り違え・task3.csv 列順/行/値域。
ログに `31 models across ... / 7 groups (recipes)` と `seg 31/31 本` が出ること（CPU 実行になっていないこと: `device=cuda`）。
`docker images | grep v13` でサイズ確認（v12 30.2 GB + 6 本 ≈ 33 GB 前後）。

## 8. push（ユーザー手動、dl2）

```bash
docker push docker.synapse.org/syn77311180/tigersqai26_shunsuke:v13
```
30 GB 超なので **30〜60 分**。**20:00 に push が始められていなければ v13 は諦めて v12 のまま**（v12 が既に push 済みであること）。
push 後は Synapse の提出フォームで v13 を task1/2/3 担当として登録し、主催者にメール（`submit/writeup/email_to_organizers.md`）。

## 9. 記録（push の成否に関わらず）

- `submit/SUBMISSIONS.md` に `## v013` を追記（構成・枠数・本数・テスト結果・push 時刻）。
- `daily_reports/20260916.md` に経緯を追記。
- write-up `submit/writeup/writeup_draft.md`: v13 になった場合のみ
  - 冒頭注記と `Docker image` 行の `v12` → `v13`、「25 full-data segmentation checkpoints」→「31」、§2.4 の「25 checkpoints」、§2.9「Seed replicas」段落の枠内本数（`q_endovis18_dlv3` × 17, `s_kdr_seed43` × 4, 他 × 2）
  - Appendix A 再生成: `.venv/bin/python3 submit/writeup/make_member_table.py submit/v006_a23/model_v13` の出力で表を差し替え（`Total: 31 members`）
  - §2.6 のイメージサイズ
- ビデオ `submit/writeup/video/TigerSQAI_task12_jmees_v2.pptx` の Final members 表は 16 本版のまま古い（別途対応、write-up 優先）。

## 10. 中止基準

- 19:00 時点で完走が 3 本未満 → 中止（v12 のまま）。
- テスト不合格 or `groups` が 7 以外 → 中止して原因を記録。v12 には触らない。
- /data4 が再び 100% になったら学習を止める（`touch workspace/expA23_sweep/daemon.stop`）。

## ★ 11. 【17:10 追加・必須】後処理を v13 で無効化する（別スレッドの CV 結果）

最終 7 レシピ（fold OOF、TTA 5 枠）に **提出コンテナと同じ後処理**を掛けて公式 Dice/HD を測った
（`workspace/expA23_sweep/score_final_postproc.py` → `eval_official_hd.py`、`official_hd_final*.json`）:

| 後処理 | T1 Dice / HD | T2 Dice / HD | Dice−HD (T1 / T2) |
|---|---|---|---|
| なし（v10/v11 の CV 値） | 0.7134 / 0.2021 | 0.7371 / 0.1876 | 0.511 / 0.550 |
| α のみ | 0.7138 / 0.2018 | 0.7354 / 0.1933 | 0.512 / 0.542 |
| 島除去 0.4% のみ | 0.7188 / 0.2242 | 0.7465 / 0.1974 | 0.495 / 0.549 |
| α + 島除去（**v12/v13 の既定**） | 0.7188 / 0.2242 | 0.7383 / 0.2101 | 0.495 / 0.528 |

- **島除去は HD を大きく悪化させる**（T1 +0.022, T2 +0.010）。公式 HD は「片方だけ空 = 1.0」なので、GT にある小構造の唯一の成分を消すと重み 3 のクラスが最悪値になる。Dice の +0.005/+0.009 では割に合わない。過去の +0.032 は Dice だけで測った旧アンサンブル（DiceDet 無し）の値
- **α も現アンサンブルでは T2 を悪化**させる（旧アンサンブルの OOF で fit した係数のまま）
- → **v13 では α を OFF、島除去はしきい値を下げる**（下記）。`model_v13/alpha.json` を削除（`load_alpha` は無ければ None）。
  ```bash
  cd <repo>/submit/v006_a23
  rm -f model_v13/alpha.json
  grep -q "ISLAND_PPM" Dockerfile || sed -i 's#^ENV HF_HUB_OFFLINE=1#ENV ISLAND_PPM=0 HF_HUB_OFFLINE=1#' Dockerfile
  grep -n "ISLAND_PPM\|model_v13" Dockerfile; ls model_v13/alpha.json 2>&1
  ```
  回帰テストのログに `alpha` 読込の行が出ないこと、`ISLAND_PPM=0` で島除去がスキップされることを確認（`process.py` は `ppm <= 0` で素通し）。
- **島除去しきい値のスイープ（α なし、公式 Dice/HD、最終 7 レシピ TTA の fold OOF）**:

  | 方式 | T1 Dice / HD | T2 Dice / HD | Dice−HD 合計 |
  |---|---|---|---|
  | なし | 0.7134 / 0.2021 | 0.7371 / 0.1876 | 1.0608 |
  | 一律 0.05 % | 0.7177 / 0.2043 | 0.7429 / 0.1869 | 1.0694 |
  | 一律 0.1 % | 0.7194 / 0.2067 | 0.7457 / 0.1875 | 1.0709 |
  | 一律 0.4 %（v12 既定） | 0.7188 / 0.2242 | 0.7465 / 0.1974 | 1.0437 |
  | クラス別: GT 平均面積の 1 % | 0.7179 / 0.2029 | 0.7427 / 0.1851 | 1.0726 |
  | **クラス別: GT 平均面積の 2 %（採用）** | **0.7193 / 0.2053** | **0.7453 / 0.1848** | **1.0745** |
  | クラス別: 4 段階 (100/250/500/1000) | 0.7178 / 0.2029 | 0.7421 / 0.1850 | 1.0720 |
  | クラス別: GT 平均面積の 3 % | 0.7204 / 0.2060 | 0.7464 / 0.1862 | 1.0746（2 % と同等、HD がやや悪いので 2 % のまま） |

  クラス別しきい値 = そのクラスが写っている GT 画像での平均面積（`workspace/expA23_sweep/gt_class_area_ppm.json`）の 2 %。
  例: 右反回神経 90 ppm、リンパ節 711 ppm、食道 1 775 ppm、胸膜 3 528 ppm。一律 0.05 % より T1/T2 とも Dice 上・HD 同等以下。

- **v13 の設定（確定、17:58 に適用済み）**:
  - `process.py` に `patch_island_per_class.py` を適用済み（`model/island_ppm.json` があればクラス別、無ければ `ISLAND_PPM` 一律。既存 v12 と同じ入力なら出力同一を関数テストで確認）。バックアップ `process_v12_backup.py`
  - `submit/v006_a23/island_ppm.json`（= 2 % 版）を **`model_v13/island_ppm.json` に配置済み**。`model_v13/alpha.json` は **まだ残っている（削除は build 担当スレッドで実施）**。
    §5 の組み立てスクリプトを再実行すると model_v13 が作り直されるので、**build 直前に必ず**:
    ```bash
    cd <repo>/submit/v006_a23
    cp island_ppm.json model_v13/island_ppm.json && rm -f model_v13/alpha.json
    grep -q "ISLAND_PPM" Dockerfile || sed -i 's#^ENV HF_HUB_OFFLINE=1#ENV ISLAND_PPM=500 HF_HUB_OFFLINE=1#' Dockerfile   # json が無い場合の保険
    ls model_v13/island_ppm.json; ls model_v13/alpha.json 2>&1 | tail -1   # alpha は No such file が正
    ```
  - 回帰テストのログで `alpha loaded` が出ないこと、`ISLAND_PPM_MAP` 由来の島除去が動くこと（`process.py` 起動ログ）を確認。

