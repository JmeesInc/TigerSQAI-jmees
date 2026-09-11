"""Depth-ordered, organ-wise progressive generation from a synthetic label map.

Rather than denoising the whole frame once (v1/v2), each anatomical class is painted
in its own pass, back to front, with the already-painted region held fixed through
masked latent blending (RePaint-style). This is the NCT anatomy-aware recipe, but the
ordering and the occlusion come from the Blender depth buffer instead of an ad-hoc
blend, and every pass is conditioned on seg+depth so the passes share one 3D scene.

Using latent blending rather than an inpainting checkpoint means the surgical LoRA
trained in v3 applies unchanged.
"""
import argparse, json, logging, sys
from pathlib import Path
import numpy as np
import torch
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
PROMPT = 'thoracoscopic esophagectomy surgical field, endoscopic view'


def build_palette(csv_path):
    rows = [l.split(',') for l in open(csv_path).read().strip().split('\n')[1:]]
    pal = np.zeros((31, 3), np.uint8)
    for r in rows:
        pal[int(r[0])] = [int(r[2]), int(r[3]), int(r[4])]
    return pal


def depth_order(label, depth_u8, min_pixels):
    """Classes sorted far -> near by median condition value (bright = near)."""
    order = []
    for c in np.unique(label):
        if c == 0:
            continue
        m = label == c
        if m.sum() < min_pixels:
            continue
        order.append((float(np.median(depth_u8[m])), int(c), int(m.sum())))
    order.sort()                      # ascending brightness == far first
    return order


@torch.no_grad()
def generate(pipe, cond, mask_stack, steps, guidance, cond_scale, generator, device, dtype):
    """One reverse diffusion run; after each step the latent outside the class being
    painted is reset to the previously accepted latent, so earlier passes survive."""
    sched = pipe.scheduler
    h, w = cond.shape[-2:]
    shape = (1, 4, h // 8, w // 8)
    emb = pipe.encode_prompt(PROMPT, device, 1, True, '')
    emb = torch.cat([emb[1], emb[0]])          # (negative, positive)
    accepted = None
    for m in mask_stack:
        ml = torch.nn.functional.interpolate(m[None, None].to(device, dtype), shape[-2:],
                                             mode='nearest')
        sched.set_timesteps(steps, device=device)   # resets multistep history between stages
        cur = torch.randn(shape, generator=generator, device=device, dtype=dtype) * sched.init_noise_sigma
        for t in sched.timesteps:
            if accepted is not None:
                # RePaint: the frozen region must re-enter at the SAME noise level as the
                # region being painted. Feeding a clean latent into a high-t step diverges.
                n = torch.randn(shape, generator=generator, device=device, dtype=dtype)
                cur = cur * ml + sched.add_noise(accepted, n, t.reshape(1)) * (1 - ml)
            inp = sched.scale_model_input(torch.cat([cur] * 2), t)
            down, mid = pipe.controlnet(inp, t, encoder_hidden_states=emb,
                                        controlnet_cond=torch.cat([cond] * 2),
                                        conditioning_scale=cond_scale, return_dict=False)
            noise = pipe.unet(inp, t, encoder_hidden_states=emb,
                              down_block_additional_residuals=down,
                              mid_block_additional_residual=mid).sample
            uncond, text = noise.chunk(2)
            noise = uncond + guidance * (text - uncond)
            cur = sched.step(noise, t, cur).prev_sample
        if accepted is not None:
            cur = cur * ml + accepted * (1 - ml)     # final composite, both clean
        assert torch.isfinite(cur).all(), 'latent diverged'
        accepted = cur
    return accepted


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--controlnet', required=True, type=Path)
    p.add_argument('--unet-lora', type=Path, default=None)
    p.add_argument('--labels', required=True, type=Path)
    p.add_argument('--depth', required=True, type=Path)
    p.add_argument('--output', required=True, type=Path)
    p.add_argument('--labelmap', default=str(ROOT / 'data/labelmap.csv'))
    p.add_argument('--n', type=int, default=8)
    p.add_argument('--width', type=int, default=768)
    p.add_argument('--height', type=int, default=448)
    p.add_argument('--steps', type=int, default=30)
    p.add_argument('--guidance', type=float, default=7.0)
    p.add_argument('--cond-scale', type=float, default=1.0)
    p.add_argument('--max-stages', type=int, default=6,
                   help='奥から数えて何クラスまで個別パスにするか (残りは最終パスでまとめて)')
    p.add_argument('--min-pixels', type=int, default=400)
    p.add_argument('--seed', type=int, default=0)
    a = p.parse_args()

    a.output.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(level=logging.INFO, format='%(asctime)s | %(levelname)s | %(message)s',
                        handlers=[logging.StreamHandler(sys.stdout),
                                  logging.FileHandler(a.output / 'generate.log')])
    log = logging.getLogger()

    from diffusers import ControlNetModel, StableDiffusionControlNetPipeline, UniPCMultistepScheduler
    # A checkpoint written before the config fix still declares 3 channels, so trust
    # the weight shape rather than the config.
    from safetensors import safe_open
    wf = next(Path(a.controlnet).glob('*.safetensors'))
    with safe_open(wf, framework='pt') as fh:
        ncond = fh.get_slice('controlnet_cond_embedding.conv_in.weight').get_shape()[1]
    net = ControlNetModel.from_pretrained(a.controlnet, torch_dtype=torch.float16,
                                          conditioning_channels=ncond)
    log.info(f'controlnet conditioning_channels={ncond}')
    pipe = StableDiffusionControlNetPipeline.from_pretrained(
        'runwayml/stable-diffusion-v1-5', controlnet=net, torch_dtype=torch.float16,
        safety_checker=None)
    pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
    if a.unet_lora:
        from safetensors.torch import load_file
        sd = load_file(Path(a.unet_lora) / 'pytorch_lora_weights.safetensors')
        pipe.load_lora_weights({f'unet.{k}': v for k, v in sd.items()})
        assert pipe.get_active_adapters(), 'UNet LoRA failed to attach'
        log.info(f'UNet LoRA: {pipe.get_active_adapters()}')
    pipe = pipe.to('cuda')
    pipe.set_progress_bar_config(disable=True)
    dev, dtype = 'cuda', torch.float16

    pal = build_palette(a.labelmap)
    files = sorted(a.labels.glob('*.png'))[:a.n]
    log.info(f'{len(files)} label maps, up to {a.max_stages} depth-ordered stages each')

    for i, f in enumerate(files):
        lab = np.array(Image.open(f).resize((a.width, a.height), Image.NEAREST))
        dep = np.array(Image.open(a.depth / f.name).convert('L')
                       .resize((a.width, a.height), Image.BILINEAR))
        order = depth_order(lab, dep, a.min_pixels)
        if not order:
            log.warning(f'{f.stem}: no class above min-pixels, skipped'); continue

        seg = torch.from_numpy(pal[lab]).permute(2, 0, 1).float() / 255.0
        d3 = torch.from_numpy(dep).float()[None].repeat(3, 1, 1) / 255.0
        cond = torch.cat([seg, d3], 0)[None].to(dev, dtype)

        # Far classes get their own pass; the rest are finished in one final full-frame pass.
        stages = []
        cum = np.zeros_like(lab, bool)
        for _, c, _ in order[:a.max_stages]:
            cum = cum | (lab == c)
            stages.append(torch.from_numpy(cum.astype(np.float32)))
        stages.append(torch.ones(a.height, a.width))

        g = torch.Generator(dev).manual_seed(a.seed + i)
        lat = generate(pipe, cond, stages, a.steps, a.guidance, a.cond_scale, g, dev, dtype)
        with torch.no_grad():
            img = pipe.vae.decode(lat / pipe.vae.config.scaling_factor).sample
        img = ((img / 2 + 0.5).clamp(0, 1)[0].permute(1, 2, 0).float().cpu().numpy() * 255)
        Image.fromarray(img.astype(np.uint8)).save(a.output / f'{f.stem}_gen.png')
        Image.fromarray(pal[lab]).save(a.output / f'{f.stem}_cond.png')
        (a.output / f'{f.stem}_stages.json').write_text(json.dumps(
            {'order_far_to_near': [{'class': c, 'depth_median': d, 'pixels': n}
                                   for d, c, n in order]}, indent=2))
        log.info(f'[{i+1}/{len(files)}] {f.stem} | {len(stages)} stages | '
                 f'far->near {[c for _, c, _ in order[:a.max_stages]]}')
    log.info('done')


if __name__ == '__main__':
    main()
