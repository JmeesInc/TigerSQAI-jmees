"""FLF2V 用の station ペア候補を選ぶ.

同一 case 内で解剖学的に対応する station ペア (6L-7L, 6R-7R, 12L-13L, ...) を
画像 NCC とマスククラス Jaccard でスコアリングし, 生成の入力候補 CSV を吐く.
"""
import argparse, collections, csv, itertools, logging, os, re, sys
import numpy as np
from PIL import Image

LOG = logging.getLogger(__name__)

# 解剖学的に隣接 = 視野が重なると期待される station ペア (同側優先)
CANDIDATE_PAIRS = [
    ("6L", "7L"), ("6R", "7R"), ("10L", "6L"), ("10R", "6R"),
    ("12L", "13L"), ("12R", "13R"), ("11L", "9"), ("11R", "9"),
    ("10L", "7L"), ("10R", "7R"), ("11L", "11R"), ("10L", "8"), ("11L", "8"),
]
CONTROL_PAIRS = [("12L", "7R"), ("13L", "7R")]  # 低類似の対照群

FNAME = re.compile(r"^(center_\d+_case_\d+)_(.+)\.png$")


def index_cases(img_dir):
    by_case = collections.defaultdict(dict)
    for f in sorted(os.listdir(img_dir)):
        if not f.endswith(".png") or "_frame_" in f:
            continue
        m = FNAME.match(f)
        if m:
            by_case[m.group(1)][m.group(2)] = f
    return by_case


def gray(path, size=(160, 90)):
    a = np.asarray(Image.open(path).convert("L").resize(size), dtype=np.float32)
    return (a - a.mean()) / (a.std() + 1e-6)


def class_set(path, stride=4):
    a = np.asarray(Image.open(path).convert("RGB"))[::stride, ::stride].reshape(-1, 3)
    return set(map(tuple, np.unique(a, axis=0)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data")
    ap.add_argument("--out", default="workspace/expS02_flf2v/outputs/pairs.csv")
    ap.add_argument("--top", type=int, default=0, help=">0 なら NCC 上位 N 件に絞る")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")

    img_dir = os.path.join(args.data, "images")
    mask_dir = os.path.join(args.data, "masks_fine")
    by_case = index_cases(img_dir)
    LOG.info("cases=%d", len(by_case))

    gcache, ccache = {}, {}
    rows = []
    for case, st in sorted(by_case.items()):
        for group, pairs in (("candidate", CANDIDATE_PAIRS), ("control", CONTROL_PAIRS)):
            for s1, s2 in pairs:
                if s1 not in st or s2 not in st:
                    continue
                f1, f2 = st[s1], st[s2]
                for f in (f1, f2):
                    if f not in gcache:
                        gcache[f] = gray(os.path.join(img_dir, f))
                        ccache[f] = class_set(os.path.join(mask_dir, f))
                ncc = float((gcache[f1] * gcache[f2]).mean())
                ca, cb = ccache[f1], ccache[f2]
                jac = len(ca & cb) / max(1, len(ca | cb))
                w, h = Image.open(os.path.join(img_dir, f1)).size
                w2, h2 = Image.open(os.path.join(img_dir, f2)).size
                rows.append(dict(case=case, group=group, station_a=s1, station_b=s2,
                                 file_a=f1, file_b=f2, ncc=round(ncc, 4),
                                 jaccard=round(jac, 4), n_cls_a=len(ca), n_cls_b=len(cb),
                                 same_size=int((w, h) == (w2, h2)), width=w, height=h))

    rows.sort(key=lambda r: (-r["ncc"],))
    if args.top:
        keep = [r for r in rows if r["group"] == "candidate"][: args.top]
        keep += [r for r in rows if r["group"] == "control"][-3:]
        rows = keep
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", newline="") as fh:
        wtr = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        wtr.writeheader()
        wtr.writerows(rows)
    LOG.info("wrote %s (%d rows)", args.out, len(rows))
    cand = [r for r in rows if r["group"] == "candidate"]
    LOG.info("candidate NCC: max=%.3f median=%.3f min=%.3f",
             cand[0]["ncc"], np.median([r["ncc"] for r in cand]), cand[-1]["ncc"])
    for r in rows[:10]:
        LOG.info("  %-8s %s %-4s-%-4s ncc=%.3f jac=%.3f %dx%d", r["group"], r["case"],
                 r["station_a"], r["station_b"], r["ncc"], r["jaccard"], r["width"], r["height"])


if __name__ == "__main__":
    main()
