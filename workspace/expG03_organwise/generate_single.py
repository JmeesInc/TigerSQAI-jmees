"""Generate surgical images from label maps with the finetuned ControlNet.

Two modes:
  --labels real     reconstruct from real label maps (sanity: does it look like the source centre?)
  --labels <dir>    generate from synthetic atlas label maps (the actual goal)
"""
import argparse, logging, random, re, sys, time
from pathlib import Path
import numpy as np
import torch
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]


def build_palette(csv):
    rows = [l.split(',') for l in open(csv).read().strip().split('\n')[1:]]
    pal = np.zeros((31, 3), np.uint8)
    for r in rows:
        pal[int(r[0])] = [int(r[2]), int(r[3]), int(r[4])]
    return pal


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--controlnet', required=True, type=Path)
    p.add_argument('--labels', required=True, type=Path)
    p.add_argument('--output', required=True, type=Path)
    p.add_argument('--labelmap', default=str(ROOT / 'data/labelmap.csv'))
    p.add_argument('--n', type=int, default=12)
    p.add_argument('--width', type=int, default=768)
    p.add_argument('--height', type=int, default=448)
    p.add_argument('--steps', type=int, default=30)
    p.add_argument('--guidance', type=float, default=7.0)
    p.add_argument('--cond-scale', type=float, default=1.0)
    p.add_argument('--unet-lora', type=Path, default=None, help='UNet LoRA adapter dir (v2)')
    p.add_argument('--depth', type=Path, default=None, help='6ch 条件用の深度ディレクトリ')
    p.add_argument('--seed', type=int, default=0)
    a = p.parse_args()

    a.output.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(level=logging.INFO, format='%(asctime)s | %(levelname)s | %(message)s',
                        handlers=[logging.StreamHandler(sys.stdout),
                                  logging.FileHandler(a.output / 'generate.log')])
    log = logging.getLogger()

    from diffusers import ControlNetModel, StableDiffusionControlNetPipeline, UniPCMultistepScheduler
    from safetensors import safe_open
    wf = next(Path(a.controlnet).glob('*.safetensors'))
    with safe_open(wf, framework='pt') as fh:
        ncond = fh.get_slice('controlnet_cond_embedding.conv_in.weight').get_shape()[1]
    net = ControlNetModel.from_pretrained(a.controlnet, torch_dtype=torch.float16,
                                          conditioning_channels=ncond)
    pipe = StableDiffusionControlNetPipeline.from_pretrained(
        'runwayml/stable-diffusion-v1-5', controlnet=net, torch_dtype=torch.float16, safety_checker=None)
    pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
    if a.unet_lora:
        # save_lora_adapter() writes bare peft keys; the pipeline loader needs the
        # "unet." prefix or it silently attaches nothing (verified: zero pixel diff).
        from safetensors.torch import load_file
        sd = load_file(Path(a.unet_lora) / 'pytorch_lora_weights.safetensors')
        pipe.load_lora_weights({f'unet.{k}': v for k, v in sd.items()})
        assert pipe.get_active_adapters(), 'UNet LoRA failed to attach'
        log.info(f'loaded UNet LoRA from {a.unet_lora}: {pipe.get_active_adapters()}')
    pipe = pipe.to('cuda')
    pipe.set_progress_bar_config(disable=True)

    pal = build_palette(a.labelmap)
    files = sorted(a.labels.glob('*.png'))
    random.Random(a.seed).shuffle(files)
    files = files[:a.n]
    log.info(f'{len(files)} label maps from {a.labels}')

    for i, f in enumerate(files):
        lab = np.array(Image.open(f).resize((a.width, a.height), Image.NEAREST))
        cond = Image.fromarray(pal[lab])
        if a.depth is not None and ncond == 6:
            import torch as _t
            dp = np.array(Image.open(a.depth / f.name).convert('L')
                          .resize((a.width, a.height), Image.BILINEAR))
            seg_t = _t.from_numpy(np.array(cond)).permute(2, 0, 1).float() / 255.0
            d3 = _t.from_numpy(dp).float()[None].repeat(3, 1, 1) / 255.0
            cond = _t.cat([seg_t, d3], 0)[None]
        prompt = 'thoracoscopic esophagectomy surgical field, endoscopic view'
        g = torch.Generator('cuda').manual_seed(a.seed + i)
        img = pipe(prompt, image=cond, num_inference_steps=a.steps,
                   guidance_scale=a.guidance, controlnet_conditioning_scale=a.cond_scale,
                   generator=g).images[0]
        img.save(a.output / f'{f.stem}_gen.png')
        Image.fromarray(pal[lab]).save(a.output / f'{f.stem}_cond.png')
        log.info(f'[{i+1}/{len(files)}] {f.stem}')
    log.info('done')


if __name__ == '__main__':
    main()
