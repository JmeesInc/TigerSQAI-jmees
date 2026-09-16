"""model_v13 = model_v11 (hardlink) + model_v13_new の seed 違い（group/tta を枠に合わせ、priority 末尾）."""
import json, os, shutil
src, new, dst = "model_v11", "model_v13_new", "model_v13"
SPEC = {
  "full_r_xl_s43": ("r_xl", False), "full_q_both_dlv3_s43": ("q_both_dlv3", True),
  "full_l_dicedet_s43": ("l_dicedet", True), "full_r_nohflip_s43": ("r_nohflip", False),
  "full_k_dicedet_rules_s44": ("s_kdr_seed43", True),
  # r_ft_fine は fold 検証の best_epoch が 5/5 fold とも 0（追学習は 1 epoch 目が最良）。
  # 8 epoch 版を出荷すると CV が検証した重みと別物になるので、1 epoch 版 2 seed に差し替える。
  "full_r_ft_fine_ep1": ("r_ft_fine", True), "full_r_ft_fine_s43_ep1": ("r_ft_fine", True),
}
# v11 から外すメンバー（上の差し替えで置き換わるもの）
DROP = {"full_r_ft_fine"}
if os.path.isdir(dst): shutil.rmtree(dst)
# .pt は hardlink（容量ゼロ）、json は **実体コピー**。
# json まで hardlink すると dst 側の書き換え・削除が src(model_v11) に波及する（一度踏んだ）。
def _cp(a, b):
    (shutil.copyfile if a.endswith(".json") else os.link)(a, b)
shutil.copytree(src, dst, copy_function=_cp)
order = json.load(open(f"{dst}/priority.json"))
for m in list(DROP):
    if m in order:
        order.remove(m); shutil.rmtree(f"{dst}/{m}"); print("drop:", m)
for name, (group, tta) in SPEC.items():
    if not os.path.isfile(f"{new}/{name}/fold0.pt"): print("skip (not exported):", name); continue
    os.makedirs(f"{dst}/{name}"); os.link(f"{new}/{name}/fold0.pt", f"{dst}/{name}/fold0.pt")
    c = json.load(open(f"{new}/{name}/config.json")); c["group"] = group; c["tta"] = tta
    json.dump(c, open(f"{dst}/{name}/config.json", "w"), indent=2)
    order.append(name)
json.dump(order, open(f"{dst}/priority.json", "w"), indent=1)
groups = {}
for m in order:
    g = json.load(open(f"{dst}/{m}/config.json"))["group"]; groups[g] = groups.get(g, 0) + 1
print(len(order), "checkpoints;", groups); assert len(groups) == 7, "枠数が 7 でない"
