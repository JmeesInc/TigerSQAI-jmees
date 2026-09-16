"""v006: workspace の学習結果を提出コンテナ用に書き出す.

各メンバーについて `model/<name>/` に
  - `config.json` : model_def.build_model に渡す dict（smp_hub の場合は学習時の
                    `.config` を `hub_config` として埋め込み、**オフラインで再構築**できる形にする）
  - `fold{N}.pt`  : `model.` 接頭辞を外した **fp16** state_dict
を置く。

Usage:
    python3 export.py --members A23:expA23_h_upernet_swin_l A23:expA23_d_base \
                      --folds 0 1 2 3 4
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import yaml

REPO = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent


def resolve(member: str) -> tuple[Path, Path, str]:
    """メンバー指定 -> (config パス, results ディレクトリ, 出力名)。"""
    if member.startswith("A23:"):
        arm = member.split(":", 1)[1]
        return (REPO / "workspace/expA23_sweep/configs" / f"{arm}.yaml",
                REPO / "workspace/expA23_sweep/results" / arm,
                arm.replace("expA23_", ""))
    exp = member                      # 旧世代 (expA06_f2c_loss など)
    cfg_path = REPO / "workspace" / exp / "config.yaml"
    cfg = yaml.safe_load(cfg_path.read_text())
    return cfg_path, REPO / cfg["paths"]["results_root"] / cfg["experiment"]["name"], exp


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--members", nargs="+", required=True)
    ap.add_argument("--folds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    ap.add_argument("--out", default=str(HERE / "model"))
    ap.add_argument("--prefer-latest", action="store_true",
                    help="latest.ckpt (最終 epoch) を優先する。全データ学習は val が学習データなので "
                         "best.ckpt の選択根拠が無く、終盤プラトーの最終 epoch を使う")
    args = ap.parse_args()

    out_root = Path(args.out)
    out_root.mkdir(parents=True, exist_ok=True)
    index = []
    for member in args.members:
        cfg_path, res_dir, name = resolve(member)
        cfg = yaml.safe_load(cfg_path.read_text())
        m = dict(cfg["model"])
        m["encoder_weights"] = None
        img = [cfg["data"]["img_h"], cfg["data"]["img_w"]]

        if m.get("source") == "smp_hub":
            # 学習時と同じ構造を **ネット無しで**組めるよう、hub の config を焼き込む
            import segmentation_models_pytorch as smp
            kw = {"img_size": tuple(img)} if "swin" in m["hub_id"] else {}
            hub = smp.from_pretrained(m["hub_id"], **kw)
            m["hub_config"] = {k: v for k, v in dict(hub.config).items()}
            del hub

        d = out_root / name
        d.mkdir(exist_ok=True)
        (d / "config.json").write_text(json.dumps({"model": m, "img_size": img}, indent=2))

        n = 0
        for f in args.folds:
            ckpt = res_dir / f"fold{f}" / "best.ckpt"
            if args.prefer_latest:
                # 再開した run は Lightning が latest-v1.ckpt を作るので、
                # 「最終」は名前でなく **更新時刻が最新**のものを取る
                cands = [q for q in (res_dir / f"fold{f}").glob("*")
                         if q.name in ("last_fp16.pt",) or (q.name.startswith("latest") and q.suffix == ".ckpt")]
                if cands:
                    ckpt = max(cands, key=lambda q: q.stat().st_mtime)
            if not ckpt.exists():
                ckpt = res_dir / f"fold{f}" / "best_fp16.pt"
            if not ckpt.exists():
                print(f"  !! {name} fold{f}: ckpt が無い ({res_dir}/fold{f})")
                continue
            obj = torch.load(ckpt, map_location="cpu", weights_only=False)
            sd = obj["state_dict"] if isinstance(obj, dict) and "state_dict" in obj else obj
            sd = {k.removeprefix("model."): v.half() for k, v in sd.items() if k.startswith("model.")}
            torch.save(sd, d / f"fold{f}.pt")
            ep = obj.get("epoch") if isinstance(obj, dict) else None
            print(f"  {name} fold{f}: {ckpt.name} (epoch {ep})")
            n += 1
        size = sum(p.stat().st_size for p in d.glob("*.pt")) / 2 ** 30
        print(f"{name}: {n} folds, {size:.2f} GB")
        index.append({"name": name, "folds": n, "gb": round(size, 2)})
    (out_root / "index.json").write_text(json.dumps(index, indent=2))
    total = sum(x["gb"] for x in index)
    print(f"合計 {sum(x['folds'] for x in index)} モデル / {total:.2f} GB -> {out_root}")


if __name__ == "__main__":
    main()
