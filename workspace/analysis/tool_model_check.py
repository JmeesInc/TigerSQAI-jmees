"""前提検証: STIR tool model の器具 Dice が expA06 の Instrument Dice を上回るか.

比較は公式と同じ階層集約 (画像 -> case -> 全体)、空集合規約 (両方空=1.0)。
対象: fine_id=1 Instrument 単独 / merged_id=12 Non-anatomical Other (Instrument+Other)。
"""
import sys
from pathlib import Path
import cv2, numpy as np, pandas as pd, torch
import segmentation_models_pytorch as smp

REPO = Path(__file__).resolve().parents[2]
STIR = REPO.parent / "STIR/reference/stitch_track/weights/convnext-unet-best.pth"
DP = REPO / "workspace/data_proc"
A06 = REPO / "workspace/expA06_f2c_loss/results/expA06_f2c_loss/oof"
MEAN = np.array([0.485, 0.456, 0.406], np.float32); STD = np.array([0.229, 0.224, 0.225], np.float32)

def dice(p, g):
    ps, gs = p.sum(), g.sum()
    if ps == 0 and gs == 0: return 1.0
    return float(2 * (p & g).sum() / (ps + gs)) if (ps + gs) else 1.0

@torch.no_grad()
def main():
    thr = float(sys.argv[1]) if len(sys.argv) > 1 else 0.5
    dev = "cuda"
    m = smp.Unet(encoder_name="tu-convnext_base.dinov3_lvd1689m", encoder_weights=None,
                 in_channels=3, classes=1, activation="sigmoid")
    st = torch.load(STIR, map_location="cpu", weights_only=False)
    st = st.get("model_state_dict", st)
    m.load_state_dict(st, strict=True); m = m.half().to(dev).eval()

    lut = np.load(DP / "lut_fine.npy")
    folds = pd.read_csv(REPO / "workspace/fold/v1/folds.csv")
    rows = []
    for i, r in enumerate(folds.itertuples()):
        gt = cv2.imread(str(DP / "labels_fine" / r.filename), cv2.IMREAD_GRAYSCALE)
        img = cv2.imread(str(REPO / "data/images" / r.filename))
        oh, ow = img.shape[:2]
        x = cv2.cvtColor(cv2.resize(img, (512, 512), interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2RGB)
        x = (x.astype(np.float32) / 255.0 - MEAN) / STD
        x = torch.from_numpy(x.transpose(2, 0, 1))[None].half().to(dev)
        pr = m(x).float()
        pr = torch.nn.functional.interpolate(pr, size=(oh, ow), mode="bilinear", align_corners=False)
        tool = (pr[0, 0].cpu().numpy() > thr)

        a06 = lut[((lambda a: (a[..., 0].astype(np.uint32) << 16) | (a[..., 1].astype(np.uint32) << 8) | a[..., 2])(
            cv2.cvtColor(cv2.imread(str(A06 / "task1" / r.filename)), cv2.COLOR_BGR2RGB)))]
        rows.append({
            "case": r.case_id,
            "stir_instr": dice(tool, gt == 1),
            "a06_instr": dice(a06 == 1, gt == 1),
            "stir_nonanat": dice(tool, (gt == 1) | (gt == 2)),
            "a06_nonanat": dice((a06 == 1) | (a06 == 2), (gt == 1) | (gt == 2)),
        })
        if (i + 1) % 100 == 0: print(f"  {i+1}/{len(folds)}", flush=True)
    df = pd.DataFrame(rows)
    agg = df.groupby("case").mean().mean()
    print(f"\n=== thr={thr} / 526 枚・case 階層集約 ===")
    print(f"Instrument (fine_id=1)     STIR {agg.stir_instr:.4f}  vs  expA06 {agg.a06_instr:.4f}  → {'STIR 勝ち' if agg.stir_instr > agg.a06_instr else 'expA06 勝ち'} ({agg.stir_instr-agg.a06_instr:+.4f})")
    print(f"Non-anatomical (id 1 or 2) STIR {agg.stir_nonanat:.4f}  vs  expA06 {agg.a06_nonanat:.4f}  → {agg.stir_nonanat-agg.a06_nonanat:+.4f}")
    df.to_csv(Path(__file__).parent / f"tool_model_check_thr{thr}.csv", index=False)


if __name__ == "__main__":
    main()
