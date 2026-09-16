"""expA23: config のキューを空いている GPU に流し込むスイープ実行機.

1 GPU = 1 ワーカで直列、GPU 間は並列。各ジョブは `train.py --fold F --config C`。
結果は各 run の summary.json から `results/sweep_results.csv` に集約する。

Usage:
    # fold0 ゲート: configs/ 全部を GPU 0,1,2,3 に配る
    python3 sweep.py --gpus 0 1 2 3 --fold 0 --configs configs/expA23_*.yaml

    # 特定の arm だけ / 別 fold
    python3 sweep.py --gpus 2 --fold 1 --configs configs/expA23_d_upernet.yaml

    python3 sweep.py --collect          # 走らずに結果表だけ更新・表示

キューは実行開始時に固定される。途中で追加したい場合は別プロセスで起動してよい
(GPU が衝突しないように --gpus を分ける)。
"""

from __future__ import annotations

import argparse
import json
import os
import queue
import subprocess
import sys
import threading
import time
from datetime import datetime
from pathlib import Path

import pandas as pd
import yaml

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
LOG_DIR = HERE / "logs"
RESULTS = HERE / "results"


def run_one(cfg_path: Path, fold: int, gpu: int, extra: list[str]) -> dict:
    name = yaml.safe_load(cfg_path.read_text())["experiment"]["name"]
    LOG_DIR.mkdir(exist_ok=True)
    log_path = LOG_DIR / f"{name}_fold{fold}.log"
    # dl2 など python が別の環境でも動くよう、起動に使った実行系をそのまま使う
    cmd = [sys.executable, str(HERE / "train.py"), "--fold", str(fold),
           "--config", str(cfg_path), *extra]
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu))
    t0 = time.time()
    with open(log_path, "a") as f:
        f.write(f"\n===== {datetime.now():%Y-%m-%d %H:%M:%S} gpu={gpu} {' '.join(cmd)}\n")
        f.flush()
        rc = subprocess.call(cmd, stdout=f, stderr=subprocess.STDOUT, env=env, cwd=str(HERE))
    return {"name": name, "fold": fold, "gpu": gpu, "rc": rc,
            "minutes": round((time.time() - t0) / 60, 1), "log": str(log_path)}


def collect() -> pd.DataFrame:
    rows = []
    for s in sorted(RESULTS.glob("*/fold*/summary.json")):
        if "_smoke" in str(s):          # 疎通確認の 2ep run は表に出さない
            continue
        try:
            rows.append(json.loads(s.read_text()))
        except json.JSONDecodeError:
            continue
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows).sort_values(["fold", "best_score"], ascending=[True, False])
    RESULTS.mkdir(exist_ok=True)
    df.to_csv(RESULTS / "sweep_results.csv", index=False)
    return df


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpus", nargs="*", type=int, default=[0])
    ap.add_argument("--fold", type=int, default=0)
    ap.add_argument("--configs", nargs="*", default=[])
    ap.add_argument("--collect", action="store_true")
    ap.add_argument("--extra", nargs="*", default=[], help="train.py に素通しする引数")
    args = ap.parse_args()

    if args.collect or not args.configs:
        df = collect()
        if len(df):
            cols = [c for c in ["name", "fold", "best_score", "tail5_mean", "best_epoch",
                                "hd_fine", "hd_coarse", "arch", "loss"] if c in df.columns]
            print(df[cols].to_string(index=False))
        else:
            print("まだ結果がありません")
        return

    q: queue.Queue = queue.Queue()
    for c in args.configs:
        q.put(Path(c))
    done: list[dict] = []
    lock = threading.Lock()

    def worker(gpu: int) -> None:
        while True:
            try:
                cfg = q.get_nowait()
            except queue.Empty:
                return
            print(f"[gpu{gpu}] start {cfg.stem} fold{args.fold}", flush=True)
            r = run_one(cfg, args.fold, gpu, args.extra)
            with lock:
                done.append(r)
                collect()
            print(f"[gpu{gpu}] done  {cfg.stem} rc={r['rc']} {r['minutes']}min", flush=True)

    threads = [threading.Thread(target=worker, args=(g,), daemon=False) for g in args.gpus]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    print("\n=== 完了 ===")
    for r in done:
        print(f"{r['name']:30s} fold{r['fold']} rc={r['rc']} {r['minutes']:6.1f}min")
    df = collect()
    if len(df):
        print(df.to_string(index=False))


if __name__ == "__main__":
    main()
