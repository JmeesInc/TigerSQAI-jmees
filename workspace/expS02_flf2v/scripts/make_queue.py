"""全 case の対応 station ペアを, 打ち切っても偏らない優先順で並べたキューを作る.

除外するもの:
  - バイト完全一致のペア (補間する中身が無い)
  - マスクが破損しているフレームを含むペア
順序は case をラウンドロビンする. 途中で打ち切っても 40 case が均等に含まれるようにするため.
"""
import argparse, collections, hashlib, logging, os
import pandas as pd

LOG = logging.getLogger(__name__)
BROKEN = {"center_7_case_2_13R.png"}   # マスクの 93.7% が背景 = 未完成注釈


def md5_all(img_dir, cache="/tmp/img_md5.txt"):
    """全画像の md5 を一括取得. data/ は共有マウントで遅いのでキャッシュを使う."""
    if os.path.exists(cache):
        out = {f.strip(): h for h, f in (l.split() for l in open(cache) if l.strip())}
        if len(out) > 400:
            LOG.info("md5 キャッシュを使用: %s (%d 件)", cache, len(out))
            return out
    LOG.info("md5 を計算中 (%s)", img_dir)
    os.system(f"cd {img_dir} && md5sum *.png > {cache}")
    return {f.strip(): h for h, f in (l.split() for l in open(cache) if l.strip())}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs", default="workspace/expS02_flf2v/outputs/pairs.csv")
    ap.add_argument("--out", default="workspace/expS02_flf2v/outputs/queue.csv")
    ap.add_argument("--ncc-min", type=float, default=0.45)
    ap.add_argument("--ncc-max", type=float, default=0.98)
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")

    d = pd.read_csv(args.pairs)
    d = d[d.group == "candidate"]
    n0 = len(d)
    d = d[(d.ncc >= args.ncc_min) & (d.ncc <= args.ncc_max)]
    LOG.info("candidate %d -> NCC[%.2f,%.2f] %d", n0, args.ncc_min, args.ncc_max, len(d))

    d = d[~d.file_a.isin(BROKEN) & ~d.file_b.isin(BROKEN)]
    LOG.info("破損マスクを含むペアを除外 -> %d", len(d))

    # 念のため md5 でも完全一致ペアを落とす (NCC の丸めで漏れる場合がある)
    h = md5_all("data/images")
    keep = [h.get(r.file_a, "a") != h.get(r.file_b, "b") for r in d.itertuples()]
    d = d[pd.Series(keep, index=d.index)]
    LOG.info("md5 一致ペアを除外 -> %d", len(d))

    # case ごとに NCC 降順, その後 case をラウンドロビン
    by_case = collections.defaultdict(list)
    for r in d.sort_values("ncc", ascending=False).itertuples():
        by_case[r.case].append(r)
    order = []
    for i in range(max(len(v) for v in by_case.values())):
        for c in sorted(by_case):
            if i < len(by_case[c]):
                order.append(by_case[c][i])
    out = pd.DataFrame([r._asdict() for r in order]).drop(columns=["Index"], errors="ignore")
    out.insert(0, "priority", range(len(out)))
    out.to_csv(args.out, index=False)
    LOG.info("wrote %s: %d pairs / %d cases", args.out, len(out), out.case.nunique())
    LOG.info("先頭40件のcase: %s", out.head(40).case.nunique())
    LOG.info("NCC 分布: min=%.3f p25=%.3f median=%.3f p75=%.3f max=%.3f",
             out.ncc.min(), out.ncc.quantile(.25), out.ncc.median(),
             out.ncc.quantile(.75), out.ncc.max())


if __name__ == "__main__":
    main()
