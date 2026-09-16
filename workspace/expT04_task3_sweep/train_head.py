"""expT04: Task3 ヘッドのスイープ（凍結 seg encoder + プーリング/loss を振る）.

expT03 (F1@0.5 0.7070 / AUROC 0.8867) からの変更点を **1 本につき 1 つだけ**動かす:
  --encoder  どの seg 実験の encoder を使うか (expA06 / expA23 の任意 arm)
  --pool     gap | gem | ms(最終2段の GAP 連結) | attn(学習可能な注意プーリング)
  --loss     bce | focal | asl(非対称: 正例の取りこぼしを重く)
  --select   last(固定 epoch) | best(val mAUROC 最良)
             ※ expT03 は best を使っており **OOF が楽観的**。既定は last にした

**station one-hot は一切使わない**（公式でファイル名 station の利用は禁止）。

Usage:
    python3 train_head.py --encoder expA06 --pool gap --loss bce --tag base
    python3 train_head.py --encoder expA23_d_base --pool ms --loss asl --tag ms_asl
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import yaml
from torch.utils.data import DataLoader, Dataset

REPO = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO / "workspace" / "expA23_sweep"))
sys.path.insert(0, str(REPO / "reference" / "tigersqai_challenge"))

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
log = logging.getLogger("t4")

STATIONS = ["6L", "6R", "7L", "7R", "8", "9", "10L", "10R", "11L", "11R", "12L", "12R", "13L", "13R"]
IMG_H, IMG_W = 576, 1024
MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


class T3Dataset(Dataset):
    def __init__(self, df: pd.DataFrame, train: bool):
        self.df = df.reset_index(drop=True)
        self.train = train
        self.img_dir = REPO / "workspace/data_proc/images_1024"

    def __len__(self):
        return len(self.df)

    def __getitem__(self, i):
        r = self.df.iloc[i]
        img = cv2.cvtColor(cv2.imread(str(self.img_dir / f"{r.case_id}.png")), cv2.COLOR_BGR2RGB)
        if self.train:
            if np.random.rand() < 0.5:
                img = img[:, ::-1].copy()
            a = 1.0 + np.random.uniform(-0.2, 0.2)
            img = np.clip(img.astype(np.float32) * a, 0, 255).astype(np.uint8)
        x = (img.astype(np.float32) / 255.0 - MEAN) / STD
        return (torch.from_numpy(x.transpose(2, 0, 1)),
                torch.from_numpy(r[STATIONS].values.astype(np.float32)))


class GeM(nn.Module):
    def __init__(self, p: float = 3.0):
        super().__init__()
        self.p = nn.Parameter(torch.tensor(p))

    def forward(self, f):
        return f.clamp(min=1e-6).pow(self.p).mean((2, 3)).pow(1.0 / self.p)


class AttnPool(nn.Module):
    """station ごとに見る場所が違うはずなので、注意重みで空間を畳む。"""

    def __init__(self, ch: int):
        super().__init__()
        self.score = nn.Conv2d(ch, 1, 1)

    def forward(self, f):
        w = self.score(f).flatten(2).softmax(-1)          # (B,1,HW)
        return (f.flatten(2) * w).sum(-1)


class T3Head(nn.Module):
    def __init__(self, encoder: nn.Module, channels: list[int], pool: str, hidden: int = 256):
        super().__init__()
        self.encoder = encoder
        for p in encoder.parameters():
            p.requires_grad_(False)
        self.pool = pool
        if pool == "ms":
            feat = channels[-1] + channels[-2]
        else:
            feat = channels[-1]
        self.gem = GeM() if pool == "gem" else None
        self.attn = AttnPool(channels[-1]) if pool == "attn" else None
        self.head = nn.Sequential(nn.Linear(feat, hidden), nn.ReLU(inplace=True),
                                  nn.Dropout(0.3), nn.Linear(hidden, len(STATIONS)))

    def _pool(self, f):
        if self.gem is not None:
            return self.gem(f.float())
        if self.attn is not None:
            return self.attn(f.float())
        return f.float().mean((2, 3))

    def forward(self, x):
        with torch.no_grad():
            feats = self.encoder(x)
        g = self._pool(feats[-1])
        if self.pool == "ms":
            g = torch.cat([g, feats[-2].float().mean((2, 3))], 1)
        return self.head(g)


def asymmetric_loss(logit, y, gamma_neg: float = 4.0, gamma_pos: float = 0.0, clip: float = 0.05):
    """ASL: 負例を確信度で減衰させ、正例の取りこぼしを相対的に重くする。"""
    p = torch.sigmoid(logit)
    pm = (p - clip).clamp(min=0)
    los_pos = y * torch.log(p.clamp(min=1e-8)) * (1 - p).pow(gamma_pos)
    los_neg = (1 - y) * torch.log((1 - pm).clamp(min=1e-8)) * pm.pow(gamma_neg)
    return -(los_pos + los_neg).mean()


def focal_loss(logit, y, gamma: float = 2.0):
    bce = nn.functional.binary_cross_entropy_with_logits(logit, y, reduction="none")
    pt = torch.exp(-bce)
    return ((1 - pt).pow(gamma) * bce).mean()


def build_encoder(spec: str, fold: int, device: str) -> tuple[nn.Module, list[int]]:
    """seg 実験の学習済み encoder を取り出す。spec は実験フォルダ名 or expA23 の arm 名。"""
    from model import build_model
    if spec == "expA06":
        cfg_path = REPO / "workspace/expA06_f2c_loss/config.yaml"
        res = REPO / "workspace/expA06_f2c_loss/results/expA06_f2c_loss" / f"fold{fold}"
        # dl2 には A06 の full ckpt が無い。expT03 が置いた encoder 単体の重みを使う
        light = REPO / "workspace/expT03_task3_noleak/enc_weights" / f"fold{fold}_encoder.pt"
        if not (res / "best.ckpt").exists() and light.exists():
            from segmentation_models_pytorch.encoders import get_encoder
            cfg = yaml.safe_load(cfg_path.read_text())
            enc = get_encoder(cfg["model"]["encoder_name"], in_channels=3, depth=5, weights=None,
                              img_size=(IMG_H, IMG_W))
            enc.load_state_dict({k: v.float() for k, v in
                                 torch.load(light, map_location="cpu").items()}, strict=True)
            ch = [c for c in enc.out_channels if c > 0]
            return enc.to(device).eval(), ch
    else:
        cfg_path = REPO / "workspace/expA23_sweep/configs" / f"{spec}.yaml"
        res = REPO / "workspace/expA23_sweep/results" / spec / f"fold{fold}"
    cfg = yaml.safe_load(cfg_path.read_text())
    m = dict(cfg["model"])
    m["encoder_weights"] = None
    model = build_model(m, (IMG_H, IMG_W))
    ckpt = res / "best.ckpt"
    if not ckpt.exists():
        ckpt = res / "best_fp16.pt"
    assert ckpt.exists(), f"missing ckpt: {res}"
    obj = torch.load(ckpt, map_location="cpu", weights_only=False)
    state = obj["state_dict"] if isinstance(obj, dict) and "state_dict" in obj else obj
    state = {k.removeprefix("model."): v.float() for k, v in state.items() if k.startswith("model.")}
    model.load_state_dict(state, strict=True)
    core = model.core
    enc = core.encoder if hasattr(core, "encoder") else core.m_fine.encoder
    ch = [c for c in enc.out_channels if c > 0]
    return enc.to(device).eval(), ch


def official_eval(pred_csv: Path) -> dict:
    from metrics.classes_stations import CLASSES_STATIONS
    from metrics.evaluate_cls import evaluate
    return evaluate(gt_csv=REPO / "workspace/data_proc/task3_gt_wide.csv",
                    pred_csv=pred_csv, classes=CLASSES_STATIONS)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--encoder", default="expA06")
    ap.add_argument("--pool", choices=["gap", "gem", "ms", "attn"], default="gap")
    ap.add_argument("--loss", choices=["bce", "focal", "asl"], default="bce")
    ap.add_argument("--select", choices=["last", "best"], default="last")
    ap.add_argument("--folds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--tag", default=None)
    args = ap.parse_args()
    tag = args.tag or f"{args.encoder}_{args.pool}_{args.loss}_{args.select}"
    torch.manual_seed(42)
    np.random.seed(42)

    gt = pd.read_csv(REPO / "workspace/data_proc/task3_gt_wide.csv")
    folds = pd.read_csv(REPO / "workspace/fold/v2/folds.csv")
    stem2fold = {f.rsplit(".", 1)[0]: fo for f, fo in zip(folds.filename, folds.fold)}
    gt["fold"] = gt.case_id.map(stem2fold)
    device = "cuda"
    out_dir = HERE / "results"
    out_dir.mkdir(exist_ok=True)

    oof = []
    for fold in args.folds:
        tr, va = gt[gt.fold != fold], gt[gt.fold == fold]
        enc, ch = build_encoder(args.encoder, fold, device)
        model = T3Head(enc, ch, args.pool).to(device)
        dl_tr = DataLoader(T3Dataset(tr, True), batch_size=args.batch, shuffle=True,
                           num_workers=6, drop_last=True)
        dl_va = DataLoader(T3Dataset(va, False), batch_size=args.batch, shuffle=False, num_workers=6)
        opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],
                                lr=args.lr, weight_decay=1e-2)
        sched = torch.optim.lr_scheduler.SequentialLR(
            opt, [torch.optim.lr_scheduler.LinearLR(opt, start_factor=0.01, total_iters=3),
                  torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(1, args.epochs - 3))],
            milestones=[3])

        best_auc, best_probs, last_probs = -1.0, None, None
        for ep in range(args.epochs):
            model.train()
            model.encoder.eval()
            for x, y in dl_tr:
                with torch.autocast("cuda", torch.float16):
                    logit = model(x.to(device))
                    yy = y.to(device)
                    loss = (nn.functional.binary_cross_entropy_with_logits(logit.float(), yy)
                            if args.loss == "bce" else
                            focal_loss(logit.float(), yy) if args.loss == "focal" else
                            asymmetric_loss(logit.float(), yy))
                opt.zero_grad()
                loss.backward()
                opt.step()
            sched.step()
            model.eval()
            ps, ys = [], []
            with torch.no_grad():
                for x, y in dl_va:
                    with torch.autocast("cuda", torch.float16):
                        logit = model(x.to(device))
                    ps.append(torch.sigmoid(logit.float()).cpu().numpy())
                    ys.append(y.numpy())
            P, Y = np.concatenate(ps), np.concatenate(ys)
            from sklearn.metrics import roc_auc_score
            aucs = [roc_auc_score(Y[:, k], P[:, k]) for k in range(len(STATIONS))
                    if len(np.unique(Y[:, k])) > 1]
            mauc = float(np.mean(aucs))
            last_probs = P
            if mauc > best_auc:
                best_auc, best_probs = mauc, P
            if ep % 10 == 0 or ep == args.epochs - 1:
                log.info("[%s] fold%d ep%d loss=%.4f mAUROC=%.4f", tag, fold, ep, loss.item(), mauc)
        P = last_probs if args.select == "last" else best_probs
        df = pd.DataFrame(P, columns=STATIONS)
        df.insert(0, "case_id", va.case_id.values)
        oof.append(df)
        del model, enc
        torch.cuda.empty_cache()

    path = out_dir / f"oof_{tag}.csv"
    pd.concat(oof).to_csv(path, index=False)
    res = official_eval(path)
    log.info("[%s] Weighted F1@0.5 = %.4f / AUROC = %.4f  (expT03 基準 0.7070 / 0.8867)",
             tag, res["final_f1"], res["final_auroc"])
    (out_dir / f"score_{tag}.json").write_text(json.dumps(
        {"tag": tag, "f1": res["final_f1"], "auroc": res["final_auroc"],
         "encoder": args.encoder, "pool": args.pool, "loss": args.loss,
         "select": args.select, "epochs": args.epochs}, indent=2))


if __name__ == "__main__":
    main()
