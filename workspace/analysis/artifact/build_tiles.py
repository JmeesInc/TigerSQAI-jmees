"""代表画像を選び、元画像/GT/各モデル予測の合成タイルを JPEG base64 で出力."""
import base64, json
from pathlib import Path
import cv2, numpy as np, pandas as pd

REPO = Path(__file__).resolve().parents[3]
D = Path(__file__).parent
EXPS = {"A00": "expA00_task12_baseline", "A05": "expA05_maxvit",
        "A06": "expA06_f2c_loss", "B03": "expB03_dino_f2c"}
df = pd.read_csv(D / "per_image_dice.csv").set_index("name")
folds = pd.read_csv(REPO / "workspace/fold/v1/folds.csv").set_index("filename")

df["gain"] = df.A06 - df.A00
df["dino"] = df.B03 - df.A06
sel = {}
for label, idx in [
    ("A06 が苦手", df.A06.nsmallest(4).index),
    ("baseline から最も改善", df.gain.nlargest(4).index),
    ("DINOv3 が効く", df.dino.nlargest(2).index),
    ("DINOv3 が壊す", df.dino.nsmallest(2).index),
]:
    for n in idx:
        sel.setdefault(n, label)

PANEL_W = 300
items = []
for name, label in sel.items():
    img = cv2.imread(str(REPO / "data/images" / name))
    gt = cv2.imread(str(REPO / "data/masks_fine" / name))
    panels = [img, gt] + [cv2.imread(str(REPO / "workspace" / EXPS[e] / "results" / EXPS[e] / "oof/task1" / name)) for e in EXPS]
    h = int(panels[0].shape[0] * PANEL_W / panels[0].shape[1])
    tile = np.concatenate([cv2.resize(p, (PANEL_W, h), interpolation=cv2.INTER_AREA) for p in panels], axis=1)
    ok, buf = cv2.imencode(".jpg", tile, [cv2.IMWRITE_JPEG_QUALITY, 78])
    assert ok
    r = df.loc[name]
    items.append({
        "name": name, "group": label, "case": folds.loc[name, "case_id"],
        "center": folds.loc[name, "center"], "fold": int(folds.loc[name, "fold"]),
        "dice": {e: round(float(r[e]), 4) for e in EXPS},
        "jpg": base64.b64encode(buf).decode(),
    })
items.sort(key=lambda x: x["dice"]["A06"])
(D / "tiles.json").write_text(json.dumps(items))
print(f"{len(items)} tiles, total {sum(len(i['jpg']) for i in items)/1e6:.2f} MB base64")
