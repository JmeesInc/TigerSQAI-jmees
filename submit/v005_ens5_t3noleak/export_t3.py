"""expT03 (station 入力なし) の Task3 凍結ヘッド 5 fold を model_t3/ へ書き出す.

expT01 (v002〜v004) のヘッドは入力に station one-hot(15) を連結していたが、
公式 Docker Instructions で「ファイル名の station 情報を予測に使ってはならない」と
明示されたため、画像特徴のみ (GAP 768) を入力とする expT03 の重みに差し替える。

ヘッドは expA06 の fold 対応 encoder 特徴に対して学習されているので、
model/fold{N}.pt (expA06) と 1:1 で対応させて使うこと。
Source: workspace/expT03_task3_noleak/results/fold{0..4}_head.pt
        CV (公式 evaluate_cls, 5fold OOF 518行): F1@0.5 0.7070 / AUROC 0.8867
"""
from __future__ import annotations

from pathlib import Path

import torch

REPO = Path(__file__).resolve().parents[2]
SRC = REPO / "workspace/expT03_task3_noleak/results"
OUT = Path(__file__).parent / "model_t3"


def main() -> None:
    OUT.mkdir(exist_ok=True)
    for fold in range(5):
        p = SRC / f"fold{fold}_head.pt"
        assert p.exists(), f"missing {p}"
        sd = torch.load(p, map_location="cpu")
        sd = {k.removeprefix("head."): v for k, v in sd.items()}
        assert sd["0.weight"].shape == (256, 768), sd["0.weight"].shape  # one-hot が無いこと
        assert sd["3.weight"].shape == (14, 256), sd["3.weight"].shape
        q = OUT / f"frozen_head_fold{fold}.pt"
        torch.save({k: v.half() for k, v in sd.items()}, q)
        print(f"{q.name}: {q.stat().st_size / 1e3:.0f} KB")


if __name__ == "__main__":
    main()
