"""GPU 版指標を公式実装と突き合わせる（fine/coarse、解像度混在の 24 枚）."""
import json, sys, time
from pathlib import Path
import numpy as np, torch, torch.nn.functional as F
HERE = Path(__file__).resolve().parent; REPO = HERE.parents[1]
sys.path.insert(0, str(REPO / "reference/tigersqai_challenge")); sys.path.insert(0, str(HERE))
from metrics import classes as CF, classes_merged as CC
from metrics.metrics import weighted_image_scores
from gpu_metrics import weighted_scores_gpu
CACHE = HERE / "_cvcache"; sizes = json.loads((CACHE / "sizes.json").read_text())
stems = sorted(sizes, key=lambda s: -sizes[s][0])[:8] + sorted(sizes, key=lambda s: sizes[s][0])[:8] + sorted(sizes)[100:108]
arms = ["q_endovis18_dlv3", "r_xl"]
dev = "cuda"; rows = []; t_cpu = t_gpu = 0.0
for s in stems:
    oh, ow = sizes[s]
    for task, mod, cls in (("fine", CF, CF.CLASSES), ("coarse", CC, CC.CLASSES_MERGED)):
        acc = sum(torch.from_numpy(np.load(HERE / f"results/expA23_{a}/probs/{task}/{s}.npy")).to(dev).float() for a in arms) / len(arms)
        pred = F.interpolate(acc[None], size=(oh, ow), mode="bilinear", align_corners=False).argmax(1)[0].to(torch.uint8)
        gt = torch.from_numpy(np.load(CACHE / f"gt_{task}" / f"{s}.npy")).to(dev)
        t = time.time(); ref = weighted_image_scores(pred.cpu().numpy(), gt.cpu().numpy(), cls, sum(c.weight for c in cls)); t_cpu += time.time() - t
        torch.cuda.synchronize(); t = time.time(); d, h = weighted_scores_gpu(pred, gt, cls); torch.cuda.synchronize(); t_gpu += time.time() - t
        rows.append((s, task, oh, ref["dice"], d, ref["hd"], h))
dd = np.array([abs(r[3] - r[4]) for r in rows]); dh = np.array([r[6] - r[5] for r in rows])
print(f"{len(rows)} 件 (fine+coarse, 720p/1080p/4K 混在)")
print(f"Dice: max|Δ| = {dd.max():.2e}")
print(f"HD  : GPU−公式  mean {dh.mean():+.4f}  max|Δ| {np.abs(dh).max():.4f}  (公式 HD の平均 {np.mean([r[5] for r in rows]):.4f})")
print(f"順位相関(HD, 画像単位): {np.corrcoef([r[5] for r in rows], [r[6] for r in rows])[0,1]:.4f}")
print(f"時間: 公式 CPU {t_cpu:.1f}s / GPU {t_gpu:.1f}s  ({len(rows)} 件)")
for r in rows[:6]: print(f"  {r[0][:24]:24s} {r[1]:6s} {r[2]:4d}p  dice {r[3]:.4f}/{r[4]:.4f}  hd {r[5]:.4f}/{r[6]:.4f}")
