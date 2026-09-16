"""expA23: 保存済み OOF 確率から **アンサンブルの組合せを貪欲探索**する（再推論なし）.

`save_probs.py` が書いた `results/<name>/probs/{fine,coarse}/*.npy`（288x512 fp16）を読み、
1/2 に縮めて（144x256）GPU に載せ、メンバーの確率平均 → argmax → 公式 weighted Dice
（画像→case→全体の階層平均、両方空=1.0 / 片方空=0.0 の規約）を評価する。

解像度を落としても **メンバー選択の順位**は変わらない（α 後処理の探索と同じ考え方）。
最終的なスコアは選ばれた組合せで原寸 OOF を出し直して確定する。

Usage:
    python3 ens_search.py --members l_dicedet d_base l_rules d_deeplabv3p h_upernet_swin_l
    python3 ens_search.py --members ... --alpha   # 選ばれた組合せに α 後処理も掛ける
"""

from __future__ import annotations

import argparse
import itertools
import json
import logging
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch

REPO = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO / "reference" / "tigersqai_challenge"))

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
log = logging.getLogger("ens")
SUB = 2   # 288x512 -> 144x256


def load_task(task: str):
    if task == "fine":
        from metrics.classes import CLASSES as C, rgb_mask_to_label_mask as dec
        return C, dec, "masks_fine"
    from metrics.classes_merged import CLASSES_MERGED as C, rgb_mask_to_label_mask as dec
    return C, dec, "masks_coarse"


def load_gt(task: str, names: list[str], hw: tuple[int, int], device: str):
    """GT を予測と同じ解像度へ nearest で縮めて読む。"""
    C, dec, sub = load_task(task)
    gts = []
    for n in names:
        rgb = cv2.cvtColor(cv2.imread(str(REPO / "data" / sub / f"{n}.png")), cv2.COLOR_BGR2RGB)
        lab = dec(rgb)
        gts.append(cv2.resize(lab.astype(np.uint8), (hw[1], hw[0]), interpolation=cv2.INTER_NEAREST))
    ids = torch.tensor([c.label_id for c in C], device=device)
    w = torch.tensor([float(c.weight) for c in C], device=device)
    n_lab = int(max(c.label_id for c in C)) + 1
    return torch.from_numpy(np.stack(gts)).to(device).long(), ids, w, n_lab


def weighted_dice(pred: torch.Tensor, gt: torch.Tensor, case_idx: torch.Tensor,
                  ids: torch.Tensor, w: torch.Tensor, n_lab: int) -> float:
    """公式規約の weighted Dice を 画像 -> case -> 全体 で集計する。"""
    b = pred.shape[0]
    flat = (gt * n_lab + pred).view(b, -1)
    cm = torch.zeros(b, n_lab * n_lab, device=pred.device)
    cm.scatter_add_(1, flat, torch.ones_like(flat, dtype=torch.float32))
    cm = cm.view(b, n_lab, n_lab)
    tp = torch.diagonal(cm, dim1=1, dim2=2)
    n_gt, n_pred = cm.sum(2), cm.sum(1)
    both0 = (n_gt == 0) & (n_pred == 0)
    one0 = ((n_gt == 0) | (n_pred == 0)) & ~both0
    d = 2 * tp / torch.clamp(n_gt + n_pred, min=1)
    d = torch.where(both0, torch.ones_like(d), torch.where(one0, torch.zeros_like(d), d))
    per_img = (d[:, ids] * w).sum(1) / w.sum()
    n_case = int(case_idx.max()) + 1
    cs = torch.zeros(n_case, device=pred.device)
    cn = torch.zeros(n_case, device=pred.device)
    cs.scatter_add_(0, case_idx, per_img)
    cn.scatter_add_(0, case_idx, torch.ones_like(per_img))
    return float((cs / torch.clamp(cn, min=1)).mean())


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--members", nargs="+", required=True, help="results/expA23_<name> の <name>")
    ap.add_argument("--extra", nargs="*", default=[],
                    help="外部の確率ディレクトリを追加 (例 ens5=workspace/analysis/oof_probs)")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--max-size", type=int, default=8, help="貪欲選択の最大メンバー数")
    args = ap.parse_args()

    dirs = {m: HERE / "results" / f"expA23_{m}" / "probs" for m in args.members}
    for spec in args.extra:          # 旧 ens5 などを 1 メンバーとして混ぜられる
        name, path = spec.split("=", 1)
        dirs[name] = REPO / path
        args.members.append(name)
    for m, d in dirs.items():
        assert (d / "fine").is_dir(), f"probs が無い: {d}"
    names = sorted(p.stem for p in (dirs[args.members[0]] / "fine").glob("*.npy"))
    common = set(names)
    for m in args.members[1:]:
        common &= {p.stem for p in (dirs[m] / "fine").glob("*.npy")}
    names = sorted(common)
    log.info("%d メンバー / 共通 %d 枚", len(args.members), len(names))

    case_ids = [n.rsplit("_", 1)[0] for n in names]
    import re
    case_ids = [re.match(r"center_\d+_case_\d+", n).group(0) for n in names]
    uniq = {c: i for i, c in enumerate(sorted(set(case_ids)))}
    case_idx = torch.tensor([uniq[c] for c in case_ids], device=args.device)

    scores = {}
    probs = {}          # task -> member -> (N,C,h,w) on GPU
    gts = {}
    for task in ("fine", "coarse"):
        probs[task] = {}
        for m in args.members:
            arr = []
            for n in names:
                p = np.load(dirs[m] / task / f"{n}.npy")            # (C,288,512) fp16
                arr.append(p[:, ::SUB, ::SUB])
            probs[task][m] = torch.from_numpy(np.stack(arr)).to(args.device)
            log.info("%s/%s 読込 %s", task, m, tuple(probs[task][m].shape))
        hw = probs[task][args.members[0]].shape[-2:]
        gts[task] = load_gt(task, names, hw, args.device)

    def evaluate(members: list[str]) -> tuple[float, float, float]:
        out = []
        for task in ("fine", "coarse"):
            gt, ids, w, n_lab = gts[task]
            acc = None
            for m in members:
                p = probs[task][m].float()
                acc = p if acc is None else acc + p
            pred = acc.argmax(1)
            out.append(weighted_dice(pred, gt, case_idx, ids, w, n_lab))
            del acc, pred
            torch.cuda.empty_cache()
        return out[0], out[1], (out[0] + out[1]) / 2

    log.info("--- 単体 ---")
    singles = {}
    for m in args.members:
        f, c, mean = evaluate([m])
        singles[m] = mean
        log.info("%-24s fine=%.4f coarse=%.4f mean=%.4f", m, f, c, mean)

    # 貪欲前進選択
    chosen = [max(singles, key=singles.get)]
    best = singles[chosen[0]]
    log.info("--- 貪欲選択 (起点 %s = %.4f) ---", chosen[0], best)
    history = [(list(chosen), best)]
    while len(chosen) < min(args.max_size, len(args.members)):
        cand, cand_score = None, best
        for m in args.members:
            if m in chosen:
                continue
            _, _, s = evaluate(chosen + [m])
            if s > cand_score:
                cand, cand_score = m, s
        if cand is None:
            log.info("これ以上追加しても上がらない")
            break
        chosen.append(cand)
        best = cand_score
        history.append((list(chosen), best))
        log.info("+ %-24s -> %.4f", cand, best)

    f, c, mean = evaluate(chosen)
    log.info("=== 選択 %s ===", chosen)
    log.info("fine=%.4f coarse=%.4f mean=%.4f (半解像度・α なし)", f, c, mean)
    (HERE / "results" / "ens_search.json").write_text(json.dumps(
        {"members": args.members, "singles": singles,
         "history": [{"members": h[0], "mean": h[1]} for h in history],
         "chosen": chosen, "fine": f, "coarse": c, "mean": mean}, indent=2))


if __name__ == "__main__":
    main()
