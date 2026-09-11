"""Mask-conditioned surgical image generation: ControlNet finetune on real (image, label) pairs.

Source: real pairs from data/images + workspace/data_proc/labels_fine_1024
Condition: fine label map colourised with the official labelmap RGB table.
Base: runwayml/stable-diffusion-v1-5 (public), ControlNet init from
      lllyasviel/control_v11p_sd15_seg (public). UNet frozen; ControlNet trained.
"""
import argparse, json, logging, math, os, random, re, sys, time
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import Dataset, DataLoader

ROOT = Path(__file__).resolve().parents[2]


def build_palette(labelmap_csv):
    rows = [l.split(',') for l in open(labelmap_csv).read().strip().split('\n')[1:]]
    pal = np.zeros((31, 3), np.uint8)
    for r in rows:
        pal[int(r[0])] = [int(r[2]), int(r[3]), int(r[4])]
    return pal


class PairDataset(Dataset):
    """Real image + colourised label condition. Center id is carried as a prompt token."""

    def __init__(self, image_dir, label_dir, palette, size, holdout_center=None, train=True):
        self.pal, self.size, self.train = palette, size, train
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
        if self.train and random.random() < 0.5:
            img = img.transpose(Image.FLIP_LEFT_RIGHT)
            lab = lab[:, ::-1].copy()
        cond = self.pal[lab]
        x = torch.from_numpy(np.array(img)).permute(2, 0, 1).float() / 127.5 - 1.0
        c = torch.from_numpy(cond).permute(2, 0, 1).float() / 255.0
        return {'pixel_values': x, 'conditioning': c, 'prompt': f'{center} thoracoscopic esophagectomy surgical field'}


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

    base = 'runwayml/stable-diffusion-v1-5'
    dev = 'cuda'
    tok = CLIPTokenizer.from_pretrained(base, subfolder='tokenizer')
    txt = CLIPTextModel.from_pretrained(base, subfolder='text_encoder').to(dev).eval().requires_grad_(False)
    vae = AutoencoderKL.from_pretrained(base, subfolder='vae').to(dev).eval().requires_grad_(False)
    unet = UNet2DConditionModel.from_pretrained(base, subfolder='unet').to(dev).eval().requires_grad_(False)
    net = ControlNetModel.from_pretrained('lllyasviel/control_v11p_sd15_seg').to(dev).train()
    sched = DDPMScheduler.from_pretrained(base, subfolder='scheduler')

    pal = build_palette(a.labelmap)
    ds = PairDataset(a.images, a.labels, pal, (a.width, a.height), a.holdout_center, train=True)
    log.info(f'train pairs: {len(ds)} (holdout_center={a.holdout_center})')
    dl = DataLoader(ds, batch_size=a.batch, shuffle=True, num_workers=8, drop_last=True, pin_memory=True)

    opt = torch.optim.AdamW(net.parameters(), lr=a.lr, weight_decay=1e-2)
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
                loss = F.mse_loss(pred.float(), target.float()) / a.accum
            scaler.scale(loss).backward()
            if (step + 1) % a.accum == 0:
                scaler.unscale_(opt); torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
                scaler.step(opt); scaler.update(); opt.zero_grad(set_to_none=True)
            step += 1
            if step % 50 == 0:
                log.info(f'step {step}/{a.steps} loss {loss.item()*a.accum:.4f} '
                         f'{(time.time()-t0)/step:.2f}s/step')
                history.append({'step': step, 'loss': loss.item() * a.accum})
            if step % 2000 == 0 or step == a.steps:
                net.save_pretrained(a.output / 'controlnet')
                (a.output / 'training_log.json').write_text(json.dumps(history, indent=2))
                log.info(f'saved checkpoint at step {step}')
            if step >= a.steps:
                break
    net.save_pretrained(a.output / 'controlnet')
    (a.output / 'training_log.json').write_text(json.dumps(history, indent=2))
    log.info('done')


if __name__ == '__main__':
    main()
