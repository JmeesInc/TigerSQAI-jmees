"""expT03: Task3 特化ヘッド — expA06 学習済み encoder (凍結) + MLP（**station one-hot なし**）.

**expT01 からの変更点は「ファイル名 station の one-hot を入力から外した」ことのみ。**
2026-09-08 の公式仕様確定により、**ファイル名に含まれる station 情報を予測に使うことは禁止**
された（expT01 は 518/518 で「自 station は必ず可視」が成立するリークに依存していたため違反）。

- 入力: 1024x576 キャッシュ画像のみ
- backbone: expA06 の fold 対応 best.ckpt から encoder のみロード・凍結
- head: encoder 最終特徴 GAP (768) + one-hot(15) → MLP(256) → 14 sigmoid, BCE
- fold v2 の case 分割。OOF 確率を CSV に書き、公式 evaluate_cls で CV

Usage: python3 train_t3.py [--folds 0 1 2 3 4] [--epochs 30] [--finetune]
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent))
from seg_model import DualHeadUnetPP  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
log = logging.getLogger("t3")

STATIONS = ["6L", "6R", "7L", "7R", "8", "9", "10L", "10R", "11L", "11R", "12L", "12R", "13L", "13R"]
IMG_H, IMG_W = 576, 1024
MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)
A06 = REPO / "workspace/expA06_f2c_loss/results/expA06_f2c_loss"


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
        return (
            torch.from_numpy(x.transpose(2, 0, 1)),
            torch.from_numpy(r[STATIONS].values.astype(np.float32)),
        )


class T3Head(nn.Module):
    def __init__(self, encoder: nn.Module, feat_ch: int, finetune: bool):
        super().__init__()
        self.encoder = encoder
        self.finetune = finetune
        if not finetune:
            for p in encoder.parameters():
                p.requires_grad_(False)
        self.head = nn.Sequential(
            nn.Linear(feat_ch, 256), nn.ReLU(inplace=True),
            nn.Dropout(0.3), nn.Linear(256, len(STATIONS)),
        )

    def forward(self, x):
        if self.finetune:
            f = self.encoder(x)[-1]
        else:
            with torch.no_grad():
                f = self.encoder(x)[-1]
        g = f.float().mean(dim=(2, 3))
        return self.head(g)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--folds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--finetune", action="store_true")
    ap.add_argument("--out", default="results")
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--ema", type=float, default=0.0, help="EMA decay (0=無効, 例 0.999)")
    ap.add_argument("--alldata", action="store_true", help="518枚フルで学習 (val は train と同一=参考値。最終ep重みを保存)")
    args = ap.parse_args()
    torch.manual_seed(42); np.random.seed(42)

    gt = pd.read_csv(REPO / "workspace/data_proc/task3_gt_wide.csv")
    folds = pd.read_csv(REPO / "workspace/fold/v2/folds.csv")
    stem2fold = {f.rsplit(".", 1)[0]: fo for f, fo in zip(folds.filename, folds.fold)}
    gt["fold"] = gt.case_id.map(stem2fold)
    device = "cuda"
    out_dir = Path(__file__).parent / args.out
    out_dir.mkdir(exist_ok=True)

    oof = []
    for fold in args.folds:
        if args.alldata:
            tr, va = gt, gt.sample(n=100, random_state=0)  # va は経過観察のみ (train に含まれる=参考値)
        else:
            tr, va = gt[gt.fold != fold], gt[gt.fold == fold]
        seg = DualHeadUnetPP("tu-maxvit_base_tf_512.in21k_ft_in1k", encoder_weights=None,
                             img_size=(IMG_H, IMG_W))
        enc_file = Path(__file__).parent / "enc_weights" / f"fold{fold}_encoder.pt"
        if enc_file.exists():  # 軽量 encoder-only 重み (dl2 等の別マシン用)
            enc_state = {k: v.float() for k, v in torch.load(enc_file, map_location="cpu").items()}
        else:
            state = torch.load(A06 / f"fold{fold}/best.ckpt", map_location="cpu", weights_only=False)["state_dict"]
            enc_state = {k.removeprefix("model.encoder."): v for k, v in state.items() if k.startswith("model.encoder.")}
        seg.encoder.load_state_dict(enc_state, strict=True)
        model = T3Head(seg.encoder, feat_ch=768, finetune=args.finetune).to(device)
        model.encoder.eval()

        dl_tr = DataLoader(T3Dataset(tr, True), batch_size=args.batch, shuffle=True, num_workers=6, drop_last=True)
        dl_va = DataLoader(T3Dataset(va, False), batch_size=args.batch, shuffle=False, num_workers=6)
        if args.finetune:
            opt = torch.optim.AdamW([
                {"params": model.encoder.parameters(), "lr": args.lr * 0.1},
                {"params": model.head.parameters(), "lr": args.lr},
            ], weight_decay=1e-2)
        else:
            params = [p for p in model.parameters() if p.requires_grad]
            opt = torch.optim.AdamW(params, lr=args.lr, weight_decay=1e-2)
        warmup_ep = 3
        sched = torch.optim.lr_scheduler.SequentialLR(
            opt,
            [torch.optim.lr_scheduler.LinearLR(opt, start_factor=0.01, total_iters=warmup_ep),
             torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(1, args.epochs - warmup_ep))],
            milestones=[warmup_ep],
        )
        crit = nn.BCEWithLogitsLoss()

        best_auc, best_probs = -1.0, None
        ema_state = None
        if args.ema > 0:
            ema_state = {k: v.detach().clone().float() for k, v in model.state_dict().items() if v.dtype.is_floating_point}
        final_probs, final_auc = None, None
        for ep in range(args.epochs):
            model.train()
            if not args.finetune:
                model.encoder.eval()
            for x, y in dl_tr:
                with torch.autocast("cuda", torch.float16):
                    logit = model(x.to(device))
                    loss = crit(logit.float(), y.to(device))
                opt.zero_grad(); loss.backward(); opt.step()
                if ema_state is not None:
                    with torch.no_grad():
                        msd = model.state_dict()
                        for k in ema_state:
                            ema_state[k].mul_(args.ema).add_(msd[k].float(), alpha=1 - args.ema)
            sched.step()
            model.eval()
            backup = None
            if ema_state is not None:
                backup = {k: v.detach().clone() for k, v in model.state_dict().items()}
                model.load_state_dict({k: ema_state.get(k, v).to(v.dtype) for k, v in backup.items()})
            ps, ys = [], []
            with torch.no_grad():
                for x, y in dl_va:
                    with torch.autocast("cuda", torch.float16):
                        logit = model(x.to(device))
                    ps.append(torch.sigmoid(logit.float()).cpu().numpy()); ys.append(y.numpy())
            if backup is not None:
                pass  # EMA 重みのまま評価済み。学習用に戻すのは下で
            P, Y = np.concatenate(ps), np.concatenate(ys)
            aucs = []
            for k in range(len(STATIONS)):
                if len(np.unique(Y[:, k])) > 1:
                    from sklearn.metrics import roc_auc_score
                    aucs.append(roc_auc_score(Y[:, k], P[:, k]))
            mauc = float(np.mean(aucs))
            if (mauc > best_auc and not args.alldata) or (args.alldata and ep == args.epochs - 1):
                best_auc, best_probs = mauc, P
                torch.save({k: v for k, v in model.state_dict().items() if not k.startswith("encoder.")} if not args.finetune else model.state_dict(),
                           out_dir / f"fold{fold}_head.pt")
            final_probs, final_auc = P, mauc  # 最終 epoch の (EMA) 評価を全データ学習の代理指標に
            if backup is not None:
                model.load_state_dict(backup)
            if ep % 5 == 0 or ep == args.epochs - 1:
                log.info("fold%d ep%d loss=%.4f val mAUROC=%.4f (best %.4f)", fold, ep, loss.item(), mauc, best_auc)
        use_probs = final_probs if args.ema > 0 else best_probs  # EMA 時は最終ep重み (=全データ学習と同条件)
        log.info("fold%d final-ep mAUROC=%.4f vs best=%.4f", fold, final_auc, best_auc)
        df = pd.DataFrame(use_probs, columns=STATIONS)
        df.insert(0, "case_id", va.case_id.values)
        oof.append(df)
        log.info("fold%d done best mAUROC=%.4f", fold, best_auc)

    pd.concat(oof).to_csv(out_dir / "oof_task3.csv", index=False)
    log.info("OOF saved -> 公式 evaluate_cls で採点してください")


if __name__ == "__main__":
    main()
