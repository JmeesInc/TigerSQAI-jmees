"""img2img from a Tier1 render, conditioned on seg+depth.

Every text2img variant so far (v1..v4) sat at 0.14-0.16 weighted Dice on synthetic
layouts while reaching 0.35 on real ones, i.e. the generator rejects the synthetic
layout rather than lacking capacity. Seeding from the Blender render gives the
initial latent the correct structure outright, so the model only has to restyle it.
This is the SimuScope recipe and it is the one configuration never tested here.
"""
import argparse, logging, sys
from pathlib import Path
import numpy as np
import torch
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
PROMPT = 'thoracoscopic esophagectomy surgical field, endoscopic view'


def build_palette(csv_path):
    pal = np.zeros((31, 3), np.uint8)
    for line in open(csv_path).read().strip().split('\n')[1:]:
        f = line.split(',')
        pal[int(f[0])] = [int(f[2]), int(f[3]), int(f[4])]
    return pal


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--controlnet', required=True, type=Path)
    p.add_argument('--unet-lora', type=Path, default=None)
    p.add_argument('--base', default='runwayml/stable-diffusion-v1-5')
    p.add_argument('--batch-dir', required=True, type=Path,
                   help='Tier1 出力 (label/ depth/ rgb/ を含む)')
    p.add_argument('--depth-cond', type=Path, default=None,
                   help='省略時は batch-dir/depth の EXR から変換')
    p.add_argument('--output', required=True, type=Path)
    p.add_argument('--labelmap', default=str(ROOT / 'data/labelmap.csv'))
    p.add_argument('--n', type=int, default=24)
    p.add_argument('--width', type=int, default=768)
    p.add_argument('--height', type=int, default=448)
    p.add_argument('--steps', type=int, default=40)
    p.add_argument('--strength', type=float, default=0.75,
                   help='1.0 は種を捨てて text2img と等価になる')
    p.add_argument('--guidance', type=float, default=7.0)
    p.add_argument('--cond-scale', type=float, default=1.0)
    p.add_argument('--seed', type=int, default=0)
    a = p.parse_args()

    a.output.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(level=logging.INFO, format='%(asctime)s | %(levelname)s | %(message)s',
                        handlers=[logging.StreamHandler(sys.stdout),
                                  logging.FileHandler(a.output / 'generate.log')])
    log = logging.getLogger()

    from diffusers import (ControlNetModel, StableDiffusionControlNetImg2ImgPipeline,
                           UniPCMultistepScheduler)
    from safetensors import safe_open
    with safe_open(next(Path(a.controlnet).glob('*.safetensors')), framework='pt') as fh:
        ncond = fh.get_slice('controlnet_cond_embedding.conv_in.weight').get_shape()[1]
    net = ControlNetModel.from_pretrained(a.controlnet, torch_dtype=torch.float16,
                                          conditioning_channels=ncond)
    pipe = StableDiffusionControlNetImg2ImgPipeline.from_pretrained(
        a.base, controlnet=net, torch_dtype=torch.float16, safety_checker=None)
    pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
    if a.unet_lora:
        from safetensors.torch import load_file
        sd = load_file(Path(a.unet_lora) / 'pytorch_lora_weights.safetensors')
        pipe.load_lora_weights({f'unet.{k}': v for k, v in sd.items()})
        assert pipe.get_active_adapters(), 'UNet LoRA failed to attach'
    pipe = pipe.to('cuda')
    pipe.set_progress_bar_config(disable=True)
    log.info(f'conditioning_channels={ncond}, strength={a.strength}')

    pal = build_palette(a.labelmap)
    depth_dir = a.depth_cond
    rgbs = sorted((a.batch_dir / 'rgb').glob('*.png'))[:a.n]
    assert rgbs, f'no Tier1 rgb in {a.batch_dir}'

    for i, rp in enumerate(rgbs):
        stem = rp.stem
        seed_img = Image.open(rp).convert('RGB').resize((a.width, a.height), Image.BICUBIC)
        lab = np.array(Image.open(a.batch_dir / 'label' / f'{stem}.png')
                       .resize((a.width, a.height), Image.NEAREST))
        seg = torch.from_numpy(pal[lab]).permute(2, 0, 1).float() / 255.0
        if ncond == 6:
            dp = Image.open(depth_dir / f'{stem}.png').convert('L') if depth_dir else None
            if dp is None:
                raise SystemExit('6ch ControlNet needs --depth-cond')
            d = torch.from_numpy(np.array(dp.resize((a.width, a.height), Image.BILINEAR))) \
                .float()[None].repeat(3, 1, 1) / 255.0
            cond = torch.cat([seg, d], 0)[None]
        else:
            cond = seg[None]

        g = torch.Generator('cuda').manual_seed(a.seed + i)
        img = pipe(PROMPT, image=seed_img, control_image=cond,
                   num_inference_steps=a.steps, strength=a.strength,
                   guidance_scale=a.guidance, controlnet_conditioning_scale=a.cond_scale,
                   generator=g).images[0]
        img.save(a.output / f'{stem}_gen.png')
        seed_img.save(a.output / f'{stem}_seed.png')
        Image.fromarray(pal[lab]).save(a.output / f'{stem}_cond.png')
        log.info(f'[{i+1}/{len(rgbs)}] {stem}')
    log.info('done')


if __name__ == '__main__':
    main()
