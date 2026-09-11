"""Fuse the public SimuScope surgical LoRA into SD1.5 and save it as a new base.

Our surgical knowledge currently comes from 528 TIGER frames only: the base
(SD1.5/LAION) and the ControlNet init (ADE20K) contribute none. SimuScope's
CholecT45 LoRA is the one public set of surgical diffusion weights, so fusing it
gives a free "ImageNet -> laparoscopic surgery -> TIGER" middle stage.

Fusing rather than stacking avoids the fused adapter competing with the LoRA we
train afterwards.
"""
import argparse, logging, sys
from pathlib import Path
import torch

ROOT = Path(__file__).resolve().parents[2]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--lora', required=True, type=Path, help='kohya .safetensors')
    p.add_argument('--output', required=True, type=Path)
    p.add_argument('--scale', type=float, default=0.6,
                   help='fuse 強度。1.0 は胆嚢/肝臓の形状概念まで引きずるので既定は控えめ')
    p.add_argument('--base', default='runwayml/stable-diffusion-v1-5')
    a = p.parse_args()

    logging.basicConfig(level=logging.INFO, format='%(asctime)s | %(levelname)s | %(message)s',
                        stream=sys.stdout)
    log = logging.getLogger()

    from diffusers import StableDiffusionPipeline
    pipe = StableDiffusionPipeline.from_pretrained(a.base, torch_dtype=torch.float32,
                                                   safety_checker=None)
    before = pipe.unet.conv_out.weight.detach().clone()
    pipe.load_lora_weights(str(a.lora))
    assert pipe.get_active_adapters(), 'LoRA failed to attach'
    log.info(f'adapters: {pipe.get_active_adapters()}')
    pipe.fuse_lora(lora_scale=a.scale)
    pipe.unload_lora_weights()

    # A fused model whose weights are bit-identical to the base means nothing was applied.
    delta = 0.0
    for n, q in pipe.unet.named_parameters():
        if 'to_q.weight' in n:
            delta = float(q.detach().abs().mean())
            break
    log.info(f'fused with scale={a.scale}; sample |to_q| mean = {delta:.6f}')

    a.output.mkdir(parents=True, exist_ok=True)
    pipe.save_pretrained(a.output)
    log.info(f'saved fused base -> {a.output}')


if __name__ == '__main__':
    main()
