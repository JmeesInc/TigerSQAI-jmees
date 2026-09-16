"""実験比較 artifact 用のデータ収集: per-image weighted Dice + 代表画像の合成タイル."""
import json, sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import cv2, numpy as np, pandas as pd

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "reference" / "tigersqai_challenge"))
DP = REPO / "workspace/data_proc"
EXPS = {
    "A00": ("expA00_task12_baseline", "ベースライン"),
    "A05": ("expA05_maxvit", "MaxViT"),
    "A06": ("expA06_f2c_loss", "MaxViT + f2c loss"),
    "B03": ("expB03_dino_f2c", "+ DINOv3 融合"),
}
_G = {}

def _init():
    from metrics.classes import CLASSES
    _G["w"] = np.zeros(31, np.float32)
    for c in CLASSES:
        _G["w"][c.label_id] = c.weight
    _G["lut"] = np.load(DP / "lut_fine.npy")

def _pack(rgb):
    rgb = rgb.astype(np.uint32)
    return (rgb[..., 0] << 16) | (rgb[..., 1] << 8) | rgb[..., 2]

def dice_one(args):
    exp, name = args
    d = REPO / "workspace" / EXPS[exp][0] / "results" / EXPS[exp][0] / "oof/task1" / name
    gt = cv2.imread(str(DP / "labels_fine" / name), cv2.IMREAD_GRAYSCALE).astype(np.int64)
    pr = _G["lut"][_pack(cv2.cvtColor(cv2.imread(str(d)), cv2.COLOR_BGR2RGB))].astype(np.int64)
    cm = np.bincount(gt.ravel() * 31 + pr.ravel(), minlength=31 * 31).reshape(31, 31)
    tp, ngt, npr = np.diag(cm), cm.sum(1), cm.sum(0)
    dice = np.where((ngt == 0) & (npr == 0), 1.0, 2 * tp / np.maximum(ngt + npr, 1))
    w = _G["w"]
    return exp, name, float((dice * w).sum() / w.sum())

if __name__ == "__main__":
    names = sorted(p.name for p in (REPO / "workspace/expA06_f2c_loss/results/expA06_f2c_loss/oof/task1").glob("*.png"))
    jobs = [(e, n) for e in EXPS for n in names]
    rows = []
    with ProcessPoolExecutor(max_workers=12, initializer=_init) as ex:
        for exp, name, d in ex.map(dice_one, jobs, chunksize=8):
            rows.append({"exp": exp, "name": name, "dice": d})
    df = pd.DataFrame(rows).pivot(index="name", columns="exp", values="dice")
    df.to_csv(Path(__file__).parent / "per_image_dice.csv")
    print(df.mean().round(4).to_dict())
