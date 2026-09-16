"""expA23: 各 arm が 576x1024 / bs2 で **構築 → forward → backward** できるかを先に潰す.

ViT/BeiT/Swin は入力サイズ固定だったり、Unet++ は C=0 ダミー段で落ちたりと、
学習を投げてから 1 分で死ぬ候補が混ざる。GPU キューを無駄に回さないための事前選別。

各 arm について VRAM ピークと 1 step の所要時間も測り、どの GPU (48/24/16GB) に
割り当てられるかを決める材料にする。

Usage:
    CUDA_VISIBLE_DEVICES=2 python3 smoke_arch.py [--configs configs/expA23_*.yaml]
"""

from __future__ import annotations

import argparse
import glob
import json
import sys
import time
import traceback
from pathlib import Path

import torch
import yaml

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from losses import build_loss  # noqa: E402
from model import build_model  # noqa: E402


def probe(cfg: dict, device: str = "cuda") -> dict:
    h, w = cfg["data"]["img_h"], cfg["data"]["img_w"]
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    t0 = time.time()
    model = build_model(cfg["model"], (h, w)).to(device)
    t_build = time.time() - t0
    n_par = sum(p.numel() for p in model.parameters()) / 1e6

    cw = json.loads((HERE.parents[1] / cfg["paths"]["class_weights"]).read_text())
    params = dict(cfg["loss"].get("params", {}) or {})
    lf = build_loss(cfg["loss"]["name"], cw["fine"], **params).to(device)
    lc = build_loss(cfg["loss"]["name"], cw["coarse"], **params).to(device)

    x = torch.randn(2, 3, h, w, device=device)
    tf = torch.randint(0, 31, (2, h, w), device=device)
    tc = torch.randint(0, 16, (2, h, w), device=device)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-6)
    scaler = torch.amp.GradScaler("cuda")

    t0 = time.time()
    for _ in range(2):
        opt.zero_grad(set_to_none=True)
        with torch.amp.autocast("cuda", dtype=torch.float16):
            of, oc = model(x)
            loss = lf(of, tf.unsqueeze(1)) + lc(oc, tc.unsqueeze(1))
        scaler.scale(loss).backward()
        scaler.step(opt)
        scaler.update()
    torch.cuda.synchronize()
    t_step = (time.time() - t0) / 2
    peak = torch.cuda.max_memory_allocated() / 2**30
    del model, opt, x, of, oc, loss
    torch.cuda.empty_cache()
    return {"ok": True, "params_M": round(n_par, 1), "build_s": round(t_build, 1),
            "step_s": round(t_step, 2), "peak_GB": round(peak, 1)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--configs", nargs="*", default=sorted(glob.glob(str(HERE / "configs" / "expA23_*.yaml"))))
    ap.add_argument("--out", default=str(HERE / "smoke_results.json"))
    args = ap.parse_args()

    results = {}
    if Path(args.out).exists():
        results = json.loads(Path(args.out).read_text())
    for c in args.configs:
        name = Path(c).stem
        cfg = yaml.safe_load(Path(c).read_text())
        try:
            r = probe(cfg)
            print(f"OK   {name:30s} {r['params_M']:7.1f}M  peak {r['peak_GB']:5.1f}GB  "
                  f"{r['step_s']:5.2f}s/step")
        except Exception as e:
            r = {"ok": False, "error": f"{type(e).__name__}: {e}"[:300]}
            print(f"FAIL {name:30s} {r['error'][:160]}")
            traceback.print_exc(limit=2, file=sys.stderr)
            torch.cuda.empty_cache()
        results[name] = r
        Path(args.out).write_text(json.dumps(results, indent=2))
    ok = [k for k, v in results.items() if v.get("ok")]
    print(f"\n{len(ok)}/{len(results)} arms 実行可能 -> {args.out}")


if __name__ == "__main__":
    main()
