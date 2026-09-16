"""完全重複画像ペアについて 画像 / Task1(coarse) / Task2(fine) / Task3 を 1 枚にまとめる."""
import collections, hashlib, logging, os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import gridspec
from PIL import Image

LOG = logging.getLogger(__name__)
DATA, OUT = "data", "workspace/analysis/duplicate_pairs"


def md5(p):
    h = hashlib.md5()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def build_luts(lm):
    fine, merged = {}, {}
    for r in lm.itertuples():
        if not pd.isna(r.fine_id):
            fine[(int(r.fine_r), int(r.fine_g), int(r.fine_b))] = (int(r.fine_id), r.fine_name, int(r.weight))
        if not pd.isna(r.merged_id):
            merged[(int(r.merged_r), int(r.merged_g), int(r.merged_b))] = (int(r.merged_id), r.merged_name)
    return fine, merged


def to_label(path, lut):
    a = np.asarray(Image.open(path).convert("RGB"))
    out = np.full(a.shape[:2], -1, np.int16)
    for c, v in lut.items():
        out[(a == np.array(c, np.uint8)).all(-1)] = v[0]
    return out, a


def per_class_dice(A, B):
    """両方のマスクのいずれかに存在するクラスについて Dice を返す."""
    ids = sorted({i for i in np.unique(np.concatenate([A.ravel(), B.ravel()])) if i >= 0})
    out = []
    for i in ids:
        x, y = (A == i), (B == i)
        out.append((i, 2 * (x & y).sum() / max(1, x.sum() + y.sum())))
    return out


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    os.makedirs(OUT, exist_ok=True)
    lm = pd.read_csv(f"{DATA}/labelmap.csv")
    fine_lut, merged_lut = build_luts(lm)

    t3 = pd.read_csv(f"{DATA}/lymph_node_station_visibility.csv")
    t3.columns = ["case", "station", "visible"]
    t3key = {(r.case, str(r.station).strip()): str(r.visible) for r in t3.itertuples()}

    files = sorted(f for f in os.listdir(f"{DATA}/images") if f.endswith(".png"))
    groups = collections.defaultdict(list)
    for f in files:
        groups[md5(f"{DATA}/images/{f}")].append(f)
    dups = sorted((sorted(v) for v in groups.values() if len(v) > 1))
    LOG.info("画像 %d 枚 / ユニーク %d / 重複グループ %d", len(files), len(groups), len(dups))

    summary = []
    for grp in dups:
        a, b = grp[0], grp[1]
        stem_a, stem_b = a[:-4], b[:-4]
        case = "_".join(stem_a.split("_")[:4])
        st_a, st_b = stem_a[len(case) + 1:], stem_b[len(case) + 1:]

        img = np.asarray(Image.open(f"{DATA}/images/{a}").convert("RGB"))
        fa, rgb_fa = to_label(f"{DATA}/masks_fine/{a}", fine_lut)
        fb, rgb_fb = to_label(f"{DATA}/masks_fine/{b}", fine_lut)
        ca, rgb_ca = to_label(f"{DATA}/masks_coarse/{a}", merged_lut)
        cb, rgb_cb = to_label(f"{DATA}/masks_coarse/{b}", merged_lut)

        agree_f, agree_c = float((fa == fb).mean()), float((ca == cb).mean())
        df, dc = per_class_dice(fa, fb), per_class_dice(ca, cb)
        mdice_f = float(np.mean([d for _, d in df]))
        mdice_c = float(np.mean([d for _, d in dc]))

        fig = plt.figure(figsize=(18, 9.8))
        gs = gridspec.GridSpec(4, 3, height_ratios=[1.0, 1.0, 0.16, 0.24],
                               hspace=0.16, wspace=0.03)

        def show(r, c, arr, title):
            ax = fig.add_subplot(gs[r, c]); ax.imshow(arr)
            ax.set_title(title, fontsize=12); ax.axis("off")

        show(0, 0, img, f"image\n{a}")
        show(0, 1, rgb_ca, "Task1  coarse (16 classes)")
        show(0, 2, rgb_fa, "Task2  fine (31 classes)")
        show(1, 0, img, f"same image — byte-identical, MD5 match\n{b}")
        show(1, 1, rgb_cb, "Task1  coarse (16 classes)")
        show(1, 2, rgb_fb, "Task2  fine (31 classes)")

        # 列ごとの一致指標を, その列の真下に置く
        stats = [("", ""),
                 ("Task1 coarse", f"Pixel Accuracy  {agree_c:.3f}\nmacro Dice      {mdice_c:.3f}   ({len(dc)} classes)"),
                 ("Task2 fine",   f"Pixel Accuracy  {agree_f:.3f}\nmacro Dice      {mdice_f:.3f}   ({len(df)} classes)")]
        for col, (head, body) in enumerate(stats):
            ax = fig.add_subplot(gs[2, col]); ax.axis("off")
            if not head:
                ax.text(0.5, 1.0, "identical image bytes — the two rows differ only in the annotation\n"
                        "(both metrics are symmetric: neither annotation is treated as ground truth)",
                        ha="center", va="top", fontsize=10, style="italic",
                        transform=ax.transAxes, color="#333333")
                continue
            ax.text(0.5, 1.0, f"{head}  —  annotation A vs B", ha="center",
                    va="top", fontsize=11.5, weight="bold", transform=ax.transAxes)
            ax.text(0.5, 0.55, body, ha="center", va="top", fontsize=11,
                    family="monospace", transform=ax.transAxes)

        axt = fig.add_subplot(gs[3, :]); axt.axis("off")
        va, vb = t3key.get((case, st_a), "(no row)"), t3key.get((case, st_b), "(no row)")
        sa = {s.strip() for s in va.split(",")} if va != "(no row)" else set()
        sb = {s.strip() for s in vb.split(",")} if vb != "(no row)" else set()
        axt.text(0.0, 1.0,
                 f"Task3  {a}  visible = {va}\n"
                 f"Task3  {b}  visible = {vb}\n"
                 f"Task3  common={sorted(sa & sb)}   only in first={sorted(sa - sb)}   "
                 f"only in second={sorted(sb - sa)}   ->  Task3 labels are "
                 f"{'IDENTICAL' if sa == sb else 'DIFFERENT'}",
                 va="top", ha="left", fontsize=12.5, family="monospace",
                 transform=axt.transAxes, color="#b00020")

        fig.suptitle(f"BYTE-IDENTICAL IMAGE PUBLISHED TWICE:   {a}   ==   {b}", fontsize=15, y=0.975)
        dst = f"{OUT}/{stem_a}__VS__{st_b}.png"
        fig.savefig(dst, dpi=95, bbox_inches="tight")
        plt.close(fig)
        LOG.info("%-46s fine PA/mDice %.3f/%.3f  coarse PA/mDice %.3f/%.3f  t3 %s",
                 stem_a + " vs " + st_b, agree_f, mdice_f, agree_c, mdice_c,
                 "same" if sa == sb else "DIFF")
        summary.append(dict(case=case, file_a=a, file_b=b, station_a=st_a, station_b=st_b,
                            fine_pixel_accuracy=round(agree_f, 4), fine_macro_dice=round(mdice_f, 4),
                            fine_n_classes=len(df),
                            coarse_pixel_accuracy=round(agree_c, 4), coarse_macro_dice=round(mdice_c, 4),
                            coarse_n_classes=len(dc),
                            t3_a=va, t3_b=vb, t3_same=int(sa == sb), png=os.path.basename(dst)))

    pd.DataFrame(summary).to_csv(f"{OUT}/summary.csv", index=False)
    LOG.info("wrote %s/summary.csv and %d figures", OUT, len(summary))


if __name__ == "__main__":
    main()
