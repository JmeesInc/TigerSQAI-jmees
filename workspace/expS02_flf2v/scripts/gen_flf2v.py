"""Wan2.1-FLF2V で station ペア間の中間フレームを生成する (スパイク用).

同一 case の解剖学的対応 station ペア (例 6L -> 7L) を first/last frame として与え,
間の動画を生成する. 生成物は擬似データの候補であり, この段階では目視評価が目的.
"""
import argparse, csv, logging, os, time
import numpy as np
import torch
from PIL import Image

LOG = logging.getLogger(__name__)

STATION_DESC = {
    "6L": "left paratracheal region", "6R": "right paratracheal region",
    "7L": "left subcarinal region", "7R": "right subcarinal region",
    "8": "middle paraesophageal region", "9": "pulmonary ligament region",
    "10L": "left hilar region", "10R": "right hilar region",
    "11L": "left lower paraesophageal region", "11R": "right lower paraesophageal region",
    "12L": "left diaphragmatic region", "12R": "right diaphragmatic region",
    "13L": "left paracardial region", "13R": "right paracardial region",
}
NEG = ("blurry, low quality, jpeg artifacts, cartoon, illustration, text, watermark, "
       "sudden cut, scene change, duplicated organs, deformed anatomy, flickering")


def build_prompt(sa, sb):
    a = STATION_DESC.get(sa, "mediastinal region")
    b = STATION_DESC.get(sb, "mediastinal region")
    return (f"Thoracoscopic esophagectomy, endoscopic view inside the right chest cavity. "
            f"The endoscope moves smoothly and continuously from the {a} to the {b}. "
            f"Wet glistening tissue with specular highlights, subtle tissue deformation from "
            f"retraction, surgical instruments moving slowly, realistic surgical video, "
            f"single continuous shot, no cuts.")


def load_resized(path, w, h):
    return Image.open(path).convert("RGB").resize((w, h), Image.LANCZOS)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs", default="workspace/expS02_flf2v/outputs/pairs.csv")
    ap.add_argument("--queue", default=None, help="make_queue.py が作った優先順 CSV")
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--num-shards", type=int, default=1)
    ap.add_argument("--out", default="workspace/expS02_flf2v/outputs/clips")
    ap.add_argument("--model", default="Wan-AI/Wan2.1-FLF2V-14B-720P-diffusers")
    ap.add_argument("--n", type=int, default=0, help="生成するペア数 (0=全部)")
    ap.add_argument("--width", type=int, default=848)
    ap.add_argument("--height", type=int, default=480)
    ap.add_argument("--frames", type=int, default=25, help="4n+1")
    ap.add_argument("--steps", type=int, default=30)
    ap.add_argument("--guidance", type=float, default=5.0)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--ncc-max", type=float, default=0.98,
                    help="完全重複 (NCC~1.0) のペアは補間対象にならないので除外")
    ap.add_argument("--ncc-min", type=float, default=0.55)
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s",
        handlers=[logging.StreamHandler(),
                  logging.FileHandler(os.path.join(args.out, "gen.log"))])

    if args.queue:
        rows = list(csv.DictReader(open(args.queue)))          # 既に優先順に並んでいる
    else:
        rows = [r for r in csv.DictReader(open(args.pairs))
                if r["group"] == "candidate"
                and args.ncc_min <= float(r["ncc"]) <= args.ncc_max]
        rows.sort(key=lambda r: -float(r["ncc"]))
    if args.n:
        if args.queue:
            rows = rows[: args.n]
        else:   # NCC 帯を散らして選ぶ (高類似だけ見ると楽観バイアスがかかる)
            idx = np.linspace(0, len(rows) - 1, args.n).round().astype(int)
            rows = [rows[i] for i in idx]
    # ラウンドロビンでシャードすると各ワーカーの担当 case が均等になる
    picks = rows[args.shard :: args.num_shards]
    LOG.info("queue=%d shard %d/%d -> %d pairs (ncc %.3f..%.3f)",
             len(rows), args.shard, args.num_shards, len(picks),
             min(float(r["ncc"]) for r in picks), max(float(r["ncc"]) for r in picks))

    from diffusers import WanImageToVideoPipeline
    from diffusers.utils import export_to_video

    LOG.info("loading %s (fp16)", args.model)
    t0 = time.time()
    pipe = WanImageToVideoPipeline.from_pretrained(args.model, torch_dtype=torch.float16)
    pipe.enable_model_cpu_offload()
    LOG.info("loaded in %.1fs", time.time() - t0)

    for k, r in enumerate(picks):
        tag = f'{r["case"]}_{r["station_a"]}_to_{r["station_b"]}'
        dst = os.path.join(args.out, tag)
        if os.path.exists(os.path.join(dst, "f000.png")):
            LOG.info("[%d/%d] %s already done, skip", k + 1, len(picks), tag)
            continue
        os.makedirs(dst, exist_ok=True)
        first = load_resized(os.path.join("data/images", r["file_a"]), args.width, args.height)
        last = load_resized(os.path.join("data/images", r["file_b"]), args.width, args.height)
        first.save(os.path.join(dst, "first.png"))
        last.save(os.path.join(dst, "last.png"))
        prompt = build_prompt(r["station_a"], r["station_b"])
        LOG.info("[%d/%d] %s ncc=%s jac=%s", k + 1, len(picks), tag, r["ncc"], r["jaccard"])
        t0 = time.time()
        out = pipe(image=first, last_image=last, prompt=prompt, negative_prompt=NEG,
                   height=args.height, width=args.width, num_frames=args.frames,
                   num_inference_steps=args.steps, guidance_scale=args.guidance,
                   generator=torch.Generator("cuda").manual_seed(args.seed)).frames[0]
        LOG.info("    generated %d frames in %.1fs (%.1fs/frame)",
                 len(out), time.time() - t0, (time.time() - t0) / len(out))
        export_to_video(out, os.path.join(dst, "clip.mp4"), fps=12)
        for i, fr in enumerate(out):
            if not isinstance(fr, Image.Image):
                fr = Image.fromarray(np.clip(np.asarray(fr) * (255 if np.asarray(fr).max() <= 1.0 else 1),
                                             0, 255).astype(np.uint8))
            fr.save(os.path.join(dst, f"f{i:03d}.png"))
        with open(os.path.join(dst, "meta.txt"), "w") as fh:
            fh.write(f"{tag}\nncc={r['ncc']} jaccard={r['jaccard']}\nprompt={prompt}\n")

    LOG.info("done -> %s", args.out)


if __name__ == "__main__":
    main()
