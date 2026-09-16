"""expA23: 枝番ディレクトリ (fold0_004 など) に入った成功 run を fold0 に正規化する.

dl2 で失敗リトライが重なると train.py が `fold{N}_001`, `_002` ... を作り、
`fold{N}` 本体は空のまま残る。下流 (save_probs / predict_oof / export) は
`fold{N}` しか見ないので、重みがあるのに「ckpt が無い」と誤認する。

安全のため **best_fp16.pt を持つディレクトリだけ**を正とみなし、
本体が空なら入れ替える (中身のある本体は触らない)。
"""
from __future__ import annotations
import re, shutil
from pathlib import Path

HERE = Path(__file__).resolve().parent
n_fix = 0
for exp in sorted((HERE / "results").iterdir()):
    if not exp.is_dir():
        continue
    for fold in range(5):
        main = exp / f"fold{fold}"
        if (main / "best_fp16.pt").exists():
            continue
        cands = sorted([d for d in exp.glob(f"fold{fold}_[0-9][0-9][0-9]")
                        if (d / "best_fp16.pt").exists()])
        if not cands:
            continue
        src = cands[-1]           # 最後に成功したもの
        if main.exists():
            shutil.rmtree(main)
        src.rename(main)
        print(f"{exp.name}: {src.name} -> fold{fold}")
        n_fix += 1
print(f"{n_fix} 件を正規化")
