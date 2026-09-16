"""STIR tool model で全 526 枚の器具マスクを生成し 1024x576 でキャッシュする（学習用）."""
from pathlib import Path
import cv2, numpy as np, pandas as pd, torch
import segmentation_models_pytorch as smp

REPO = Path(__file__).resolve().parents[2]
STIR = REPO.parent / "STIR/reference/stitch_track/weights/convnext-unet-best.pth"
OUT = REPO / "workspace/data_proc/tool_masks_1024"; OUT.mkdir(exist_ok=True)
MEAN = np.array([0.485, 0.456, 0.406], np.float32); STD = np.array([0.229, 0.224, 0.225], np.float32)
THR = 0.5

@torch.no_grad()
def main():
    dev = "cuda"
    m = smp.Unet(encoder_name="tu-convnext_base.dinov3_lvd1689m", encoder_weights=None,
                 in_channels=3, classes=1, activation="sigmoid")
    st = torch.load(STIR, map_location="cpu", weights_only=False)
    m.load_state_dict(st.get("model_state_dict", st), strict=True)
    m = m.half().to(dev).eval()
    names = sorted(p.name for p in (REPO / "workspace/data_proc/images_1024").glob("*.png"))
    frac = []
    for i, n in enumerate(names):
        img = cv2.imread(str(REPO / "data/images" / n))
        x = cv2.cvtColor(cv2.resize(img, (512, 512), interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2RGB)
        x = (x.astype(np.float32) / 255.0 - MEAN) / STD
        x = torch.from_numpy(x.transpose(2, 0, 1))[None].half().to(dev)
        pr = torch.nn.functional.interpolate(m(x).float(), size=(576, 1024), mode="bilinear", align_corners=False)
        mask = ((pr[0, 0].cpu().numpy() > THR) * 255).astype(np.uint8)
        cv2.imwrite(str(OUT / n), mask)
        frac.append(mask.mean() / 255)
        if (i + 1) % 150 == 0: print(f"  {i+1}/{len(names)}", flush=True)
    print(f"done: {len(names)} masks, 器具画素率 mean={np.mean(frac):.3f} max={np.max(frac):.3f}")

if __name__ == "__main__":
    main()
