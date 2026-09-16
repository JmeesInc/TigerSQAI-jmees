"""Regenerate the ensemble-member table of the write-up from the packaged container
(submit/v006_a23/model/*/config.json). Run from repo root:
    python3 submit/writeup/make_member_table.py [model_dir]
Paste the printed Markdown into writeup_draft.md (section "Ensemble members").
Recipe details (loss / AnatomyLoss / aug) are read from workspace/expA23_sweep/configs/<name>.yaml.
"""
import json, os, sys
from pathlib import Path
import yaml

REPO = Path(__file__).resolve().parents[2]
root = Path(sys.argv[1]) if len(sys.argv) > 1 else REPO / "submit/v006_a23/model"
cfgs = REPO / "workspace/expA23_sweep/configs"
ARCH = {"unetplusplus": "Unet++", "deeplabv3plus": "DeepLabV3+", "smp-hub/upernet-swin-large": "UPerNet (ADE20k)"}
ENC = {"tu-convnext_large.fb_in22k_ft_in1k_384": "ConvNeXt-L (IN-22k→1k, 384)",
       "tu-convnext_xlarge.fb_in22k_ft_in1k_384": "ConvNeXt-XL (IN-22k→1k, 384)"}
PRE = {"pre_cholec_deeplabv3plus/weights.pt": "CholecSeg8k", "pre_cholec_unetplusplus/weights.pt": "CholecSeg8k",
       "pre_endovis18_deeplabv3plus/weights.pt": "EndoVis18", "pre_endovis18_unetplusplus/weights.pt": "EndoVis18",
       "pre_both_deeplabv3plus/weights.pt": "CholecSeg8k+EndoVis18"}
rows = []
for name in sorted(os.listdir(root)):
    p = root / name / "config.json"
    if not p.exists():
        continue
    c = json.load(open(p)); m = c["model"]
    n_pt = len([f for f in os.listdir(root / name) if f.endswith(".pt")])
    arch = ARCH.get(m.get("arch") or m.get("hub_id", ""), m.get("arch") or m.get("hub_id", "?"))
    enc = ENC.get(m.get("encoder_name", ""), m.get("encoder_name", "Swin-L (ADE20k)") or "Swin-L (ADE20k)")
    if m.get("decoder_attention_type") == "scse" or "scse" in name:
        arch += " + scSE"
    y = cfgs / f"expA23_{name}.yaml"
    loss = anat = aug = "?"
    if y.exists():
        yc = yaml.safe_load(open(y))
        loss = yc["loss"]["name"] + (f" + f2c {yc['loss'].get('fine2coarse_weight', 0)}" if yc["loss"].get("fine2coarse_weight", 0) else "")
        r = yc.get("rules", {}) or {}
        anat = ("3D-knowledge" if "anat3d" in str(r.get("rules_dir", "")) else "GT-statistics") if r.get("enabled") else "–"
        aug = yc["data"].get("aug", "normal") + ("" if yc["data"].get("hflip", True) else ", no hflip") + (", ToolPaste" if yc.get("toolpaste", {}).get("enabled") else "")
    if m.get("init_from") and not m.get("init_from_partial"):
        loss = (loss if loss != "?" else "dicedet") + " (fine-only fine-tune, 8 ep)"
    rows.append((name.replace("full_", ""), c.get("group", "–"), "on" if c.get("tta") else "off", arch, enc,
                 PRE.get(m.get("init_from_partial", ""), "CholecSeg8k (via base)" if "cholec" in str(m.get("init_from", "")) else "–"),
                 loss, anat, aug, f"{c['img_size'][1]}×{c['img_size'][0]}", n_pt))
print(f"| # | checkpoint | slot | TTA | decoder | encoder | surgical pre-training | loss | AnatomyLoss | augmentation | input | ckpts |")
print("|---|---|---|---|---|---|---|---|---|---|---|---|")
for i, r in enumerate(rows, 1):
    print(f"| {i} | `{r[0]}` | " + " | ".join(map(str, r[1:])) + " |")
print(f"\nTotal: {len(rows)} members, {sum(r[-1] for r in rows)} checkpoints")
