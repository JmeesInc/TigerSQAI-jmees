"""fold 割り当て生成 (v1): StratifiedGroupKFold(group=case, stratify=center).

- case 抽出は正規表現 `center_\\d+_case_\\d+`（例外 2 枚の `_frame_N` 形式にも対応）
- 公式評価が case 単位の階層平均のため、同一 case の train/val 跨ぎは厳禁
- center 分布が不均一 (center_1 が 16/40 case) なので center で層化する

出力: workspace/fold/v1/folds.csv  (columns: filename, case_id, center, station, fold)
"""

from __future__ import annotations

import logging
import re
from pathlib import Path

import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

REPO = Path(__file__).resolve().parents[2]
N_SPLITS = 5
SEED = 42
VERSION = "v1"

CASE_RE = re.compile(r"(center_\d+_case_\d+)")

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
log = logging.getLogger(__name__)


def main() -> None:
    rows = []
    for p in sorted((REPO / "data" / "images").glob("*.png")):
        m = CASE_RE.match(p.name)
        assert m, f"cannot parse case from {p.name}"
        case_id = m.group(1)
        center = "_".join(case_id.split("_")[:2])  # center_N
        station = p.stem[len(case_id) + 1 :]  # 残り (station または station_frame_N)
        rows.append({"filename": p.name, "case_id": case_id, "center": center, "station": station})
    df = pd.DataFrame(rows)
    log.info("%d images / %d cases / centers: %s",
             len(df), df.case_id.nunique(), df.groupby("center").case_id.nunique().to_dict())

    sgkf = StratifiedGroupKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)
    df["fold"] = -1
    for fold, (_, val_idx) in enumerate(sgkf.split(df, y=df.center, groups=df.case_id)):
        df.loc[val_idx, "fold"] = fold

    assert (df.fold >= 0).all()
    assert (df.groupby("case_id").fold.nunique() == 1).all(), "case が fold を跨いでいる"

    out = REPO / "workspace" / "fold" / VERSION
    out.mkdir(parents=True, exist_ok=True)
    df.to_csv(out / "folds.csv", index=False)

    for fold in range(N_SPLITS):
        sub = df[df.fold == fold]
        log.info("fold %d: %3d imgs / %2d cases / %s",
                 fold, len(sub), sub.case_id.nunique(),
                 sub.groupby("center").case_id.nunique().to_dict())
    log.info("saved %s", out / "folds.csv")


if __name__ == "__main__":
    main()
