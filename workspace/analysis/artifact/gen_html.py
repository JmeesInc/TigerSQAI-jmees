import json
from pathlib import Path
import pandas as pd
D = Path(__file__).parent
REPO = D.parents[2]

scores = json.loads((D / "scores.json").read_text())
tiles = json.loads((D / "tiles.json").read_text())

# per-class: A00 / A05 / A06 / B03 を統合
a = pd.read_csv(D.parent / "per_class_task1.csv")[["cid","name","weight","dice_A00"]]
b = pd.read_csv(D.parent / "per_class_task1_A0506.csv")[["cid","dice_A05","dice_A06"]]
c = pd.read_csv(D.parent / "per_class_task1_A06B03.csv")[["cid","dice_B03"]]
pc = a.merge(b, on="cid").merge(c, on="cid")
lm = pd.read_csv(REPO / "data/labelmap.csv")[["fine_id","fine_r","fine_g","fine_b"]]
pc = pc.merge(lm, left_on="cid", right_on="fine_id")
pc["rgb"] = pc.apply(lambda r: f"rgb({int(r.fine_r)},{int(r.fine_g)},{int(r.fine_b)})", axis=1)
pc["delta"] = (pc.dice_A06 - pc.dice_A00).round(3)
# 出現しないクラス (全モデル同値かつ Dice>0.9) は absent-absent=1.0 の水増しなので印を付ける
pc["inflated"] = ((pc.dice_A00 - pc.dice_A06).abs() < 1e-6) & ((pc.dice_A06 - pc.dice_B03).abs() < 1e-6)
per_class = pc[["cid","name","weight","dice_A00","dice_A05","dice_A06","dice_B03","delta","rgb","inflated"]].to_dict("records")

payload = json.dumps({"scores": scores, "tiles": tiles, "perClass": per_class}, ensure_ascii=False)
html = (D / "template.html").read_text().replace("__DATA__", payload)
(D / "index.html").write_text(html)
print(f"{len(html)/1e6:.2f} MB written")
