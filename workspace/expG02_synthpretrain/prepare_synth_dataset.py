"""Turn generated (image, synthetic label) pairs into an expA06-compatible dataset.

Produces images_1024 / labels_fine_1024 / labels_coarse_1024 plus a folds.csv so the
existing expA06 training code can consume the synthetic set unchanged.

Fine -> coarse is the official fine_id -> merged_id mapping from data/labelmap.csv.
"""
import argparse, csv, logging, sys
from pathlib import Path
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]


def fine_to_coarse_lut(labelmap_csv):
    lut = np.zeros(31, np.uint8)
    for r in csv.DictReader(open(labelmap_csv)):
        lut[int(r['fine_id'])] = int(r['merged_id'])
    return lut


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--generated', required=True, type=Path, help='dir with *_gen.png from generate_v2')
    p.add_argument('--labels', required=True, type=Path, help='dir with the source class-id label PNGs')
    p.add_argument('--output', required=True, type=Path)
    p.add_argument('--labelmap', default=str(ROOT / 'data/labelmap.csv'))
    p.add_argument('--width', type=int, default=1024)
    p.add_argument('--height', type=int, default=576)
    p.add_argument('--val-fraction', type=float, default=0.05,
                   help='held-out slice so the trainer has a val loader; not a real evaluation')
    a = p.parse_args()

    logging.basicConfig(level=logging.INFO, format='%(asctime)s | %(levelname)s | %(message)s',
                        stream=sys.stdout)
    log = logging.getLogger()
    for sub in ['images_1024', 'labels_fine_1024', 'labels_coarse_1024']:
        (a.output / sub).mkdir(parents=True, exist_ok=True)

    lut = fine_to_coarse_lut(a.labelmap)
    rows = []
    for i, g in enumerate(sorted(a.generated.glob('*_gen.png'))):
        stem = g.name[:-8]
        lp = a.labels / f'{stem}.png'
        if not lp.exists():
            log.warning(f'no label for {stem}'); continue
        name = f'synth_{stem}.png'
        # Resumable: a run killed part-way should not redo the images it already wrote.
        done = all((a.output / sub / name).exists()
                   for sub in ['images_1024', 'labels_fine_1024', 'labels_coarse_1024'])
        if not done:
            Image.open(g).convert('RGB').resize((a.width, a.height), Image.BICUBIC).save(
                a.output / 'images_1024' / name)
            fine = np.array(Image.open(lp).resize((a.width, a.height), Image.NEAREST))
            Image.fromarray(fine, mode='L').save(a.output / 'labels_fine_1024' / name)
            Image.fromarray(lut[fine], mode='L').save(a.output / 'labels_coarse_1024' / name)
        # Synthetic frames have no real case structure; group by index so the split is clean.
        rows.append({'filename': name, 'case_id': f'synth_case_{i // 14}', 'center': 'synth',
                     'station': 'NA', 'fold': 0 if (i % int(1 / a.val_fraction)) == 0 else 1})

    out = a.output / 'folds.csv'
    with out.open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=['filename', 'case_id', 'center', 'station', 'fold'])
        w.writeheader(); w.writerows(rows)
    n_val = sum(r['fold'] == 0 for r in rows)
    log.info(f'{len(rows)} pairs -> {a.output} (val {n_val} / train {len(rows) - n_val})')


if __name__ == '__main__':
    main()
