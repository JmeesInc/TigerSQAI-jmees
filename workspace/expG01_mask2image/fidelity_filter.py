"""Fidelity filter: run the expA06 segmentation model on generated images and
score agreement with the conditioning label map.

Generated frames whose weighted Dice against their own condition falls below a
threshold are rejected: the diffusion model invented content that contradicts
the label, so the pair would be mislabelled training data.
"""
import argparse, json, logging, sys
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'workspace/expA06_f2c_loss'))


def load_model(ckpt, device):
    import yaml
    from model import DualHeadUnetPP
    cfg = yaml.safe_load((ckpt.parent / 'config.yaml').read_text())['model']
    net = DualHeadUnetPP(
        encoder_name=cfg['encoder_name'], encoder_weights=None,
        decoder_channels=tuple(cfg['decoder_channels']),
        num_classes_fine=cfg['num_classes_fine'],
        num_classes_coarse=cfg['num_classes_coarse'],
        img_size=tuple(cfg.get('img_size', (576, 1024))))
    sd = torch.load(ckpt, map_location='cpu', weights_only=False)['state_dict']
    net.load_state_dict({k.replace('model.', '', 1): v for k, v in sd.items() if k.startswith('model.')})
    return net.eval().to(device), tuple(cfg.get('img_size', (576, 1024)))


def weighted_dice(pred, gt, weights, n=31):
    """Official convention: classes absent from both count as 1.0; weights from labelmap."""
    num = den = 0.0
    per = {}
    for c in range(1, n):
        p, g = pred == c, gt == c
        if not p.any() and not g.any():
            continue
        d = 2 * np.count_nonzero(p & g) / max(np.count_nonzero(p) + np.count_nonzero(g), 1)
        per[c] = float(d)
        num += weights[c] * d
        den += weights[c]
    return (num / den if den else 0.0), per


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--generated', required=True, type=Path, help='dir with *_gen.png and *_cond.png')
    p.add_argument('--labels', required=True, type=Path, help='dir with the source class-id label PNGs')
    p.add_argument('--ckpt', type=Path,
                   default=ROOT / 'workspace/expA06_f2c_loss/results/expA06_f2c_loss/fold0/best.ckpt')
    p.add_argument('--labelmap', default=str(ROOT / 'data/labelmap.csv'))
    p.add_argument('--threshold', type=float, default=0.35)
    p.add_argument('--output', type=Path, default=None)
    a = p.parse_args()

    logging.basicConfig(level=logging.INFO, format='%(asctime)s | %(levelname)s | %(message)s',
                        stream=sys.stdout)
    log = logging.getLogger()
    dev = 'cuda'
    rows = [l.split(',') for l in open(a.labelmap).read().strip().split('\n')[1:]]
    weights = np.ones(31)
    for r in rows:
        weights[int(r[0])] = float(r[5])

    net, img_size = load_model(a.ckpt, dev)
    H, W = img_size
    mean = torch.tensor([0.485, 0.456, 0.406], device=dev).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=dev).view(1, 3, 1, 1)

    results = []
    for g in sorted(a.generated.glob('*_gen.png')):
        stem = g.name[:-8]
        lp = a.labels / f'{stem}.png'
        if not lp.exists():
            log.warning(f'no label for {stem}'); continue
        img = Image.open(g).convert('RGB').resize((W, H), Image.BICUBIC)
        x = torch.from_numpy(np.array(img)).permute(2, 0, 1)[None].float().to(dev) / 255.0
        x = (x - mean) / std
        with torch.no_grad(), torch.amp.autocast('cuda', dtype=torch.float16):
            out = net(x)
            logits = out[0] if isinstance(out, (tuple, list)) else out['fine']
            pred = logits.float().argmax(1)[0].cpu().numpy().astype(np.uint8)
        gt = np.array(Image.open(lp).resize((W, H), Image.NEAREST))
        d, per = weighted_dice(pred, gt, weights)
        results.append({'frame': stem, 'weighted_dice': d, 'accepted': bool(d >= a.threshold),
                        'per_class': per})
        log.info(f'{stem:<34} weighted Dice {d:.4f} {"OK" if d >= a.threshold else "REJECT"}')

    if results:
        v = np.array([r['weighted_dice'] for r in results])
        acc = sum(r['accepted'] for r in results)
        log.info(f'\n{len(results)} frames | mean {v.mean():.4f} median {np.median(v):.4f} '
                 f'| accepted {acc}/{len(results)} (threshold {a.threshold})')
    out = a.output or a.generated / 'fidelity.json'
    out.write_text(json.dumps({'threshold': a.threshold, 'ckpt': str(a.ckpt),
                               'results': results}, indent=2))
    log.info(f'saved {out}')


if __name__ == '__main__':
    main()
