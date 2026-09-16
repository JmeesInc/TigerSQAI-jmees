"""model_v10 -> model_v11: 指定 group 以外のメンバーに tta=true を付ける（実体はハードリンク）."""
import json, os, shutil, sys
NO_TTA = set(sys.argv[1:]) or {"r_nohflip"}
src, dst = "model_v10", "model_v11"
if os.path.isdir(dst): shutil.rmtree(dst)
os.makedirs(dst)
order = json.load(open(f"{src}/priority.json"))
n_tta = 0
for m in order:
    os.makedirs(f"{dst}/{m}")
    for f in os.listdir(f"{src}/{m}"):
        if f != "config.json": os.link(f"{src}/{m}/{f}", f"{dst}/{m}/{f}")
    c = json.load(open(f"{src}/{m}/config.json"))
    c["tta"] = c["group"] not in NO_TTA; n_tta += c["tta"]
    json.dump(c, open(f"{dst}/{m}/config.json", "w"), indent=2)
os.link(f"{src}/alpha.json", f"{dst}/alpha.json")
json.dump(order, open(f"{dst}/priority.json", "w"), indent=1)
print(f"model_v11: {len(order)} 本, tta=true {n_tta} 本, TTA なし group: {sorted(NO_TTA)}")
