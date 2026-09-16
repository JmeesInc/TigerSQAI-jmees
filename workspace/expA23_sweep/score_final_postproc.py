"""最終 7 レシピ (fold OOF 確率, TTA 5 枠) に提出コンテナと同じ後処理 (α on fine + 島除去 0.4%) を掛けて
公式 Dice/HD を測る。出力 PNG は results/final_postproc/{task1=fine 色, task2=coarse 色} (このリポジトリの慣習)。
   python3 score_final_postproc.py && python3 eval_official_hd.py --pred workspace/expA23_sweep/results/final_postproc --tag final_tta_alpha_island --jobs 32
"""
import json, sys, os
from pathlib import Path
import cv2, numpy as np, pandas as pd, torch, torch.nn.functional as F
from multiprocessing import Pool
HERE = Path(__file__).resolve().parent; REPO = HERE.parents[1]
sys.path.insert(0, str(REPO / "submit/v006_a23"))
ARMS = ["q_endovis18_dlv3_tta", "r_xl", "r_ft_fine_tta", "q_both_dlv3_tta", "s_kdr_seed43_tta", "l_dicedet_tta", "r_nohflip"]
ISLAND_PPM = float(os.environ.get("ISLAND_PPM", "4000"))
USE_ALPHA = os.environ.get("USE_ALPHA", "1") == "1"
OUT = HERE / os.environ.get("OUT_DIR", "results/final_postproc")
PPM_JSON = os.environ.get("ISLAND_PPM_JSON")   # {"fine": {id: ppm}, "coarse": {id: ppm}} -> per-class thresholds
PPM_MAP = json.load(open(PPM_JSON)) if PPM_JSON else None
alpha = json.load(open(REPO / "submit/v006_a23/model_v11/alpha.json"))
lm = pd.read_csv(REPO / "data/labelmap.csv")
id2rgb_f = np.zeros((256, 3), np.uint8); id2rgb_c = np.zeros((256, 3), np.uint8)
for r in lm.itertuples():
    id2rgb_f[int(r.fine_id)] = (r.fine_r, r.fine_g, r.fine_b); id2rgb_c[int(r.merged_id)] = (r.merged_r, r.merged_g, r.merged_b)
a_f = np.ones(31, np.float32)
if USE_ALPHA:
    for k, v in alpha["fine"].items(): a_f[int(k)] = v

def remove_islands(lab, ppm, per_class=None):
    out = lab.copy()
    for c in np.unique(lab):
        if c == 0: continue
        thr = int((per_class.get(str(int(c)), ppm) if per_class else ppm) * lab.size / 1_000_000)
        if thr <= 1: continue
        m = (lab == c).astype(np.uint8); n, cc, stats, _ = cv2.connectedComponentsWithStats(m, 8)
        if n <= 1: continue
        for i in np.where(stats[1:, cv2.CC_STAT_AREA] < thr)[0]:
            x, y, w_, h_, _ = stats[i + 1]
            sl = (slice(max(y - 2, 0), y + h_ + 2), slice(max(x - 2, 0), x + w_ + 2))
            comp = cc[sl] == (i + 1); ring = cv2.dilate(comp.astype(np.uint8), np.ones((5, 5), np.uint8)) > 0; ring &= ~comp
            vals = out[sl][ring]; vals = vals[vals != c]
            out[sl][comp] = np.bincount(vals).argmax() if len(vals) else 0
    return out

def one(stem):
    torch.set_num_threads(1)
    oh, ow = cv2.imread(str(REPO / "data/images" / f"{stem}.png")).shape[:2]
    res = {}
    for task, sub, id2rgb, a in (("fine", "task1", id2rgb_f, a_f), ("coarse", "task2", id2rgb_c, None)):
        acc = None
        for arm in ARMS:
            p = np.load(HERE / f"results/expA23_{arm}/probs/{task}/{stem}.npy").astype(np.float32)
            acc = p if acc is None else acc + p
        acc /= len(ARMS)
        if a is not None: acc *= a[:, None, None]
        up = F.interpolate(torch.from_numpy(acc)[None], size=(oh, ow), mode="bilinear", align_corners=False)
        ids = up.argmax(1)[0].numpy().astype(np.uint8)
        if PPM_MAP is not None: ids = remove_islands(ids, 0, PPM_MAP[task])
        elif ISLAND_PPM > 0: ids = remove_islands(ids, ISLAND_PPM)
        cv2.imwrite(str(OUT / sub / f"{stem}.png"), cv2.cvtColor(id2rgb[ids], cv2.COLOR_RGB2BGR))
    return stem

if __name__ == "__main__":
    (OUT / "task1").mkdir(parents=True, exist_ok=True); (OUT / "task2").mkdir(parents=True, exist_ok=True)
    stems = [p.stem for p in sorted((HERE / "results/expA23_r_xl/probs/fine").glob("*.npy"))]
    with Pool(32) as pool:
        for i, _ in enumerate(pool.imap_unordered(one, stems, chunksize=4), 1):
            if i % 100 == 0: print(i, flush=True)
    print("done", len(stems))
