"""v3: depth-conditioned, class-aware ControlNet + UNet LoRA.

Combines the two published recipes that match this task:
  - SimuScope (WACV25): add DEPTH to the conditioning for sim-to-real consistency.
    Real frames have no depth, so they use Depth-Anything-V2 estimates; synthetic
    frames use exact Blender depth, both normalised to inverse depth.
  - CASDM (HTL25): class-aware weighting so the weight-3 small structures are not
    drowned out by pleura/lung, which is exactly how v2 failed.

Condition is 6 channels (seg RGB + depth RGB) through ONE ControlNet rather than two
stacked ones, so the branches cannot fight each other. conv_in is widened from the
trained v2 seg ControlNet with the depth half zero-initialised: generation starts
identical to v2 and learns to use depth from there.

Original v2 docstring:
v2: ControlNet + UNet LoRA finetune on real (image, label) pairs.

v1 trained ControlNet alone with the UNet frozen. Fidelity to the conditioning
mask came out at 0.27 weighted Dice against a 0.49 real-image baseline, and
small structures scored ~0: the frozen UNet has no surgical prior, so it
renders plausible tissue while ignoring what the condition asks for. v2 also
adapts the UNet with LoRA so the backbone learns the domain.

Prompt is a single fixed caption - the v1 per-centre style token had no effect
(centre_1/3/6/7 produced near-identical images), so it is dropped.
"""
import argparse, json, logging, math, os, random, re, sys, time
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import Dataset, DataLoader

ROOT = Path(__file__).resolve().parents[2]
PROMPT = 'thoracoscopic esophagectomy surgical field, endoscopic view'


def build_palette(labelmap_csv):
    rows = [l.split(',') for l in open(labelmap_csv).read().strip().split('\n')[1:]]
    pal = np.zeros((31, 3), np.uint8)
    for r in rows:
        pal[int(r[0])] = [int(r[2]), int(r[3]), int(r[4])]
    return pal


class PairDataset(Dataset):
    """Real image + colourised label condition. Center id is carried as a prompt token."""

    def __init__(self, image_dir, label_dir, palette, size, holdout_center=None, train=True,
                 depth_dir=None, class_weights=None):
        self.pal, self.size, self.train = palette, size, train
        self.depth_dir = Path(depth_dir) if depth_dir else None
        self.cw = class_weights
        labels = sorted(Path(label_dir).glob('*.png'))
        self.items = []
        for lp in labels:
            ip = Path(image_dir) / lp.name
            if not ip.exists():
                continue
            center = re.match(r'(center_\d+)', lp.name).group(1)
            if holdout_center and ((center == holdout_center) == train):
                continue
            self.items.append((ip, lp, center))

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        ip, lp, center = self.items[i]
        w, h = self.size
        img = Image.open(ip).convert('RGB').resize((w, h), Image.BICUBIC)
        lab = np.array(Image.open(lp).resize((w, h), Image.NEAREST))
        flipped = self.train and random.random() < 0.5
        if flipped:
            img = img.transpose(Image.FLIP_LEFT_RIGHT)
            lab = lab[:, ::-1].copy()
        cond = self.pal[lab]
        x = torch.from_numpy(np.array(img)).permute(2, 0, 1).float() / 127.5 - 1.0
        c = torch.from_numpy(cond).permute(2, 0, 1).float() / 255.0
        if self.depth_dir is not None:
            dp = Image.open(self.depth_dir / lp.name).convert('L').resize((w, h), Image.BILINEAR)
            dnp = np.array(dp)
            if flipped:
                dnp = dnp[:, ::-1].copy()
            d = torch.from_numpy(dnp).float()[None].repeat(3, 1, 1) / 255.0
            c = torch.cat([c, d], 0)
        # Per-pixel loss weight from the official 3/2/1 class weights (CASDM-style).
        wmap = torch.from_numpy(self.cw[lab]).float()[None] if self.cw is not None \
            else torch.ones(1, h, w)
        return {'pixel_values': x, 'conditioning': c, 'weight_map': wmap, 'prompt': PROMPT}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--images', default=str(ROOT / 'data/images'))
    p.add_argument('--labels', default=str(ROOT / 'workspace/data_proc/labels_fine_1024'))
    p.add_argument('--labelmap', default=str(ROOT / 'data/labelmap.csv'))
    p.add_argument('--output', required=True, type=Path)
    p.add_argument('--width', type=int, default=768)
    p.add_argument('--height', type=int, default=448)
    p.add_argument('--steps', type=int, default=6000)
    p.add_argument('--batch', type=int, default=2)
    p.add_argument('--accum', type=int, default=4)
    p.add_argument('--lr', type=float, default=1e-5)
    p.add_argument('--depth', default=str(ROOT / 'workspace/expG03_organwise/depth_real'),
                   help='depth condition dir, same filenames as the labels')
    p.add_argument('--init-controlnet', default='lllyasviel/control_v11p_sd15_seg',
                   help='v3 は学習済み v2 ControlNet から開始するのが既定の使い方')
    p.add_argument('--init-unet-lora', default=None, help='v2 の UNet LoRA から継続する場合')
    p.add_argument('--class-aware', action='store_true', help='CASDM 風のクラス重み付き loss')
    p.add_argument('--base', default='runwayml/stable-diffusion-v1-5',
                   help='SimuScope LoRA を fuse 済みのローカルベースを指定できる')
    p.add_argument('--lora-rank', type=int, default=32)
    p.add_argument('--lora-lr', type=float, default=1e-4)
    p.add_argument('--holdout-center', default=None, help='exclude this center from training (generalisation check)')
    p.add_argument('--seed', type=int, default=42)
    a = p.parse_args()

    a.output.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(level=logging.INFO, format='%(asctime)s | %(levelname)s | %(message)s',
                        handlers=[logging.StreamHandler(sys.stdout),
                                  logging.FileHandler(a.output / f'train_{time.strftime("%Y%m%d_%H%M%S")}.log')])
    log = logging.getLogger()
    torch.manual_seed(a.seed); random.seed(a.seed); np.random.seed(a.seed)

    from diffusers import AutoencoderKL, ControlNetModel, DDPMScheduler, UNet2DConditionModel
    from transformers import CLIPTextModel, CLIPTokenizer

    base = a.base
    dev = 'cuda'
    tok = CLIPTokenizer.from_pretrained(base, subfolder='tokenizer')
    txt = CLIPTextModel.from_pretrained(base, subfolder='text_encoder').to(dev).eval().requires_grad_(False)
    vae = AutoencoderKL.from_pretrained(base, subfolder='vae').to(dev).eval().requires_grad_(False)
    unet = UNet2DConditionModel.from_pretrained(base, subfolder='unet').to(dev).requires_grad_(False)
    from peft import LoraConfig
    unet.add_adapter(LoraConfig(r=a.lora_rank, lora_alpha=a.lora_rank, init_lora_weights='gaussian',
                                target_modules=['to_k', 'to_q', 'to_v', 'to_out.0']))
    if a.init_unet_lora:
        from safetensors.torch import load_file
        sd = load_file(Path(a.init_unet_lora) / 'pytorch_lora_weights.safetensors')
        got = unet.load_state_dict({f'{k.replace(".weight", ".default.weight")}': v
                                    for k, v in sd.items()}, strict=False)
        log.info(f'unet lora init: unexpected {len(got.unexpected_keys)}')
    lora_params = [q for q in unet.parameters() if q.requires_grad]
    unet.train()
    log_n = sum(q.numel() for q in lora_params)
    # A checkpoint saved before the register_to_config fix still declares 3 channels,
    # so read the real width off the weights instead of trusting the config.
    ncond_init = 3
    init_path = Path(a.init_controlnet)
    if init_path.exists():
        from safetensors import safe_open
        with safe_open(next(init_path.glob('*.safetensors')), framework='pt') as fh:
            ncond_init = fh.get_slice('controlnet_cond_embedding.conv_in.weight').get_shape()[1]
    net = ControlNetModel.from_pretrained(a.init_controlnet, conditioning_channels=ncond_init)
    log.info(f'init controlnet conditioning_channels={ncond_init}')
    conv = net.controlnet_cond_embedding.conv_in
    if conv.in_channels == 3:
        wide = torch.nn.Conv2d(6, conv.out_channels, conv.kernel_size, conv.stride, conv.padding)
        with torch.no_grad():
            wide.weight.zero_()
            wide.weight[:, :3] = conv.weight          # keep the trained seg behaviour
            wide.bias.copy_(conv.bias)                # depth half starts at zero -> no-op
        net.controlnet_cond_embedding.conv_in = wide
        net.register_to_config(conditioning_channels=6)  # config.x = y は FrozenDict に残らない
    net = net.to(dev).train()
    sched = DDPMScheduler.from_pretrained(base, subfolder='scheduler')

    pal = build_palette(a.labelmap)
    cw = None
    if a.class_aware:
        cw = np.ones(31, np.float32)
        for r in open(a.labelmap).read().strip().split('\n')[1:]:
            f = r.split(',')
            cw[int(f[0])] = float(f[5])
        cw /= cw.mean()
        log.info(f'class-aware loss on: weights {cw.min():.2f}..{cw.max():.2f}')
    ds = PairDataset(a.images, a.labels, pal, (a.width, a.height), a.holdout_center, train=True,
                     depth_dir=a.depth, class_weights=cw)
    log.info(f'train pairs: {len(ds)} (holdout_center={a.holdout_center})')
    dl = DataLoader(ds, batch_size=a.batch, shuffle=True, num_workers=8, drop_last=True, pin_memory=True)

    log.info(f'UNet LoRA trainable params: {log_n/1e6:.1f}M (rank {a.lora_rank})')
    opt = torch.optim.AdamW([{'params': net.parameters(), 'lr': a.lr},
                             {'params': lora_params, 'lr': a.lora_lr}], weight_decay=1e-2)
    scaler = torch.amp.GradScaler('cuda')
    step = 0
    t0 = time.time()
    history = []
    while step < a.steps:
        for b in dl:
            with torch.no_grad():
                x = b['pixel_values'].to(dev, non_blocking=True)
                lat = vae.encode(x).latent_dist.sample() * vae.config.scaling_factor
                ids = tok(b['prompt'], padding='max_length', truncation=True,
                          max_length=tok.model_max_length, return_tensors='pt').input_ids.to(dev)
                emb = txt(ids)[0]
            noise = torch.randn_like(lat)
            t = torch.randint(0, sched.config.num_train_timesteps, (lat.shape[0],), device=dev).long()
            noisy = sched.add_noise(lat, noise, t)
            cond = b['conditioning'].to(dev, non_blocking=True)
            with torch.amp.autocast('cuda', dtype=torch.float16):
                down, mid = net(noisy, t, encoder_hidden_states=emb,
                                controlnet_cond=cond, return_dict=False)
                pred = unet(noisy, t, encoder_hidden_states=emb,
                            down_block_additional_residuals=down,
                            mid_block_additional_residual=mid).sample
                target = noise if sched.config.prediction_type == 'epsilon' else sched.get_velocity(lat, noise, t)
                if a.class_aware:
                    # Downsample the pixel-space weight map onto the latent grid and
                    # renormalise so the loss scale does not drift with class mix.
                    wm = F.adaptive_avg_pool2d(b['weight_map'].to(dev), pred.shape[-2:])
                    se = (pred.float() - target.float()) ** 2
                    loss = (se * wm).sum() / (wm.expand_as(se).sum() + 1e-8) / a.accum
                else:
                    loss = F.mse_loss(pred.float(), target.float()) / a.accum
            scaler.scale(loss).backward()
            if (step + 1) % a.accum == 0:
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(list(net.parameters()) + lora_params, 1.0)
                scaler.step(opt); scaler.update(); opt.zero_grad(set_to_none=True)
            step += 1
            if step % 50 == 0:
                log.info(f'step {step}/{a.steps} loss {loss.item()*a.accum:.4f} '
                         f'{(time.time()-t0)/step:.2f}s/step')
                history.append({'step': step, 'loss': loss.item() * a.accum})
            if step % 2000 == 0 or step == a.steps:
                net.save_pretrained(a.output / 'controlnet')
                unet.save_lora_adapter(a.output / 'unet_lora')
                (a.output / 'training_log.json').write_text(json.dumps(history, indent=2))
                log.info(f'saved checkpoint at step {step}')
            if step >= a.steps:
                break
    net.save_pretrained(a.output / 'controlnet')
    unet.save_lora_adapter(a.output / 'unet_lora')
    (a.output / 'training_log.json').write_text(json.dumps(history, indent=2))
    log.info('done')


if __name__ == '__main__':
    main()
