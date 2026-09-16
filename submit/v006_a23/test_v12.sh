#!/usr/bin/env bash
# ローカル回帰テスト: 本番同等条件 (--network none, /input RO) でコンテナを回し出力を検証。
#   ./test.sh [n_images]   # data/images から先頭 n 枚 (default 6, 4K を必ず1枚含める)
set -euo pipefail
TEAM="${TEAM:-shunsuke}"; VER="${VER:-v6}"; N="${1:-6}"
DIR="$(cd "$(dirname "$0")" && pwd)"
REPO="$(cd "${DIR}/../.." && pwd)"
IN="${DIR}/test_v12/input"; OUT="${DIR}/test_v12/output"
rm -rf "${IN}" "${OUT}"; mkdir -p "${IN}" "${OUT}"

# 入力サンプル: 先頭 N-1 枚 + 4K を 1 枚 (ls|head の SIGPIPE を避けるため配列化)
mapfile -t FILES < <(ls "${REPO}/data/images")
for f in "${FILES[@]:0:$((N-1))}"; do cp "${REPO}/data/images/${f}" "${IN}/"; done
python3 - "$REPO" "$IN" <<'EOF'
import cv2, os, shutil, sys
repo, indir = sys.argv[1], sys.argv[2]
for f in sorted(os.listdir(f"{repo}/data/images")):
    if not f.lower().endswith(".png"):
        continue
    im = cv2.imread(f"{repo}/data/images/{f}")
    if im is not None and im.shape[1] >= 3840:
        shutil.copy(f"{repo}/data/images/{f}", indir); print("4K sample:", f); break
EOF

# 既定は GPU (nvidia runtime, GPU_DEV で番号指定)。CPU で回すなら RUNTIME=runc GPU_ARGS=""
docker run --rm --runtime "${RUNTIME:-nvidia}" -e NVIDIA_VISIBLE_DEVICES="${GPU_DEV:-0}" --network none --shm-size 8g \
  -v "${IN}:/input:ro" -v "${OUT}:/output" -e FOLDS="${FOLDS:-0,1,2,3,4}" \
  "tigersqai26_${TEAM}:${VER}"

echo "--- 検証 ---"
python3 - "$IN" "$OUT" "$REPO" <<'EOF'
import cv2, os, sys
import numpy as np
IN, OUT, REPO = sys.argv[1:4]
import csv
fine_colors, coarse_colors = set(), set()
with open(f"{REPO}/data/labelmap.csv") as fh:
    for r in csv.DictReader(fh):
        fine_colors.add((int(r["fine_r"]), int(r["fine_g"]), int(r["fine_b"])))
        coarse_colors.add((int(r["merged_r"]), int(r["merged_g"]), int(r["merged_b"])))
names = sorted(n for n in os.listdir(IN) if n.lower().endswith(".png"))
seen = {"task1": set(), "task2": set()}
for sub, colors in [("task1", coarse_colors), ("task2", fine_colors)]:  # 公式: task1=merged, task2=fine
    outs = sorted(os.listdir(f"{OUT}/{sub}"))
    assert outs == names, f"{sub}: 名前不一致 {outs} vs {names}"
    for n in names:
        src = cv2.imread(f"{IN}/{n}"); pred = cv2.imread(f"{OUT}/{sub}/{n}")
        assert pred is not None and pred.shape == src.shape, f"{sub}/{n}: 解像度不一致"
        rgb = cv2.cvtColor(pred, cv2.COLOR_BGR2RGB).reshape(-1,3)
        uniq = set(map(tuple, np.unique(rgb, axis=0)))
        bad = uniq - colors
        assert not bad, f"{sub}/{n}: 未定義色 {list(bad)[:3]}"
        seen[sub] |= uniq
# merged 16 色は fine 31 色の部分集合なので「色が正しい表に載っている」だけでは
# task2 に coarse を書いても検出できない。fine 固有色が出ることを追加で要求する。
fine_only = fine_colors - coarse_colors
assert seen["task2"] & fine_only, "task2 に fine 固有色が 1 つも無い → coarse を誤って出力している疑い"
assert not (seen["task1"] & fine_only), "task1 に fine 固有色 → fine を誤って出力している"
import pandas as pd
t3 = pd.read_csv(f"{OUT}/task3.csv")
STATIONS = ["6L","6R","7L","7R","8","9","10L","10R","11L","11R","12L","12R","13L","13R"]
assert list(t3.columns) == ["case_id"] + STATIONS, f"task3 列不一致: {list(t3.columns)}"
assert sorted(t3.case_id) == sorted(n.rsplit(".",1)[0] for n in names), "task3 行の対応不一致"
assert ((t3[STATIONS] >= 0) & (t3[STATIONS] <= 1)).all().all(), "確率範囲外"
print("OK: task1=merged / task2=fine (ファイル数・命名・解像度・RGB色・表の取り違え検出) + task3.csv (列順・行対応・値域) すべて仕様準拠")
EOF