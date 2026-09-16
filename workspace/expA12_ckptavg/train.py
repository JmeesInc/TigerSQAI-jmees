"""Task1+2 同時学習 (shared encoder + dual Unet++ decoder) の学習エントリポイント.

Usage:
    python3 train.py --fold 0 [--config config.yaml] [--epochs N] [--fast-dev]

出力: {results_root}/{experiment.name}/fold{N}/
    best.ckpt / last.ckpt / swa.ckpt / config.yaml / train_YYYYmmdd_HHMMSS.log / training_log.json
last.ckpt が存在すれば自動で再開する。

expA11 (v2 基準線) + **checkpoint averaging**。train.swa.start_epoch 以降の重みを
等重み平均し、BN running stats を train データで再推定して swa.ckpt に保存する。
best.ckpt と swa.ckpt の val スコアを同一 run 内で並べるため、ペア比較になる。
"""

from __future__ import annotations

import argparse
import json
import logging
import shutil
import sys
from datetime import datetime
from pathlib import Path

import lightning.pytorch as pl
import pandas as pd
import torch
import torch.nn.functional as F  # noqa: F401  (将来の deep supervision 用)
import yaml
from lightning.pytorch.callbacks import LearningRateMonitor, ModelCheckpoint
from lightning.pytorch.loggers import CSVLogger, WandbLogger
from monai.losses import DiceCELoss, DiceFocalLoss, DiceLoss
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from torch.utils.data import DataLoader

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent))

from dataset import TigerDataset  # noqa: E402
from model import DualHeadUnetPP  # noqa: E402

log = logging.getLogger("train")


def setup_logging(out_dir: Path) -> None:
    fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    root = logging.getLogger()
    root.setLevel(logging.DEBUG)
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(fmt)
    fh = logging.FileHandler(out_dir / f"train_{datetime.now():%Y%m%d_%H%M%S}.log")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(fmt)
    root.addHandler(ch)
    root.addHandler(fh)


def build_loss(name: str, class_weights: list[int]) -> torch.nn.Module:
    w = torch.tensor(class_weights, dtype=torch.float32)
    kwargs = dict(softmax=True, to_onehot_y=True, include_background=True, weight=w)
    if name == "dice":
        return DiceLoss(**kwargs)
    if name == "dicece":
        return DiceCELoss(**kwargs)
    if name == "dicefocal":
        return DiceFocalLoss(**kwargs)
    raise ValueError(f"unknown loss: {name}")


def weighted_dice_per_image(
    pred: torch.Tensor, target: torch.Tensor, num_classes: int, weights: torch.Tensor
) -> torch.Tensor:
    """公式 Dice 規約 (両方無し=1.0, 片方無し=0.0) の weighted マクロ平均。pred/target: (H,W) long."""
    idx = target.reshape(-1) * num_classes + pred.reshape(-1)
    cm = torch.bincount(idx, minlength=num_classes * num_classes).reshape(num_classes, num_classes)
    tp = cm.diag().float()
    n_gt = cm.sum(1).float()
    n_pred = cm.sum(0).float()
    dice = torch.where(
        (n_gt == 0) & (n_pred == 0),
        torch.ones_like(tp),
        2 * tp / (n_gt + n_pred).clamp(min=1),
    )
    return (dice * weights).sum() / weights.sum()


class CheckpointAveraging(pl.Callback):
    """最終 N epoch の重みを等重み平均する (checkpoint averaging / SWA).

    - start_epoch 以降、train epoch 終了ごとに CPU fp32 の running mean を更新する
    - 非浮動小数バッファ (num_batches_tracked 等) は平均せず最新値を保持する
    - crash / レジューム対策で毎回 swa_state.pt に退避し、再開時に読み直す
      (Lightning の ckpt には入れない。1.6GB の latest.ckpt をこれ以上太らせないため)
    """

    def __init__(self, out_dir: Path, start_epoch: int):
        super().__init__()
        self.state_file = Path(out_dir) / "swa_state.pt"
        self.start_epoch = start_epoch
        self.avg: dict[str, torch.Tensor] | None = None
        self.n = 0
        if self.state_file.exists():
            d = torch.load(self.state_file, map_location="cpu", weights_only=False)
            self.avg, self.n = d["avg"], d["n"]
            log.info("SWA state restored from %s (n=%d)", self.state_file, self.n)

    def on_train_epoch_end(self, trainer, pl_module) -> None:
        ep = trainer.current_epoch
        if ep < self.start_epoch:
            return
        sd = {
            k: (v.detach().float().cpu().clone() if v.is_floating_point() else v.detach().cpu().clone())
            for k, v in pl_module.model.state_dict().items()
        }
        if self.avg is None:
            self.avg, self.n = sd, 1
        else:
            self.n += 1
            a = 1.0 / self.n
            for k, v in sd.items():
                if v.is_floating_point():
                    self.avg[k].mul_(1.0 - a).add_(v, alpha=a)
                else:
                    self.avg[k] = v
        torch.save({"avg": self.avg, "n": self.n}, self.state_file)
        log.info("SWA update: epoch %d averaged (n=%d)", ep, self.n)


@torch.no_grad()
def update_bn(model: torch.nn.Module, loader, device: str) -> int:
    """重み平均後の BN running stats を train データで再推定する (momentum=None = 累積平均)."""
    bns = [m for m in model.modules() if isinstance(m, torch.nn.modules.batchnorm._BatchNorm)]
    if not bns:
        return 0
    momenta = {m: m.momentum for m in bns}
    for m in bns:
        m.reset_running_stats()
        m.momentum = None
    model.train()
    n = 0
    for batch in loader:
        with torch.autocast("cuda", torch.float16):
            model(batch["image"].to(device, non_blocking=True))
        n += 1
    for m, mom in momenta.items():
        m.momentum = mom
    model.eval()
    return n


@torch.no_grad()
def eval_val(model: torch.nn.Module, loader, cfg: dict, wf: torch.Tensor, wc: torch.Tensor,
             device: str) -> dict:
    """monitor val と同一の指標 (画像→case→全体の weighted Dice) を平均重みで測る."""
    nf, nc = cfg["model"]["num_classes_fine"], cfg["model"]["num_classes_coarse"]
    model.eval()
    rec = []
    for batch in loader:
        with torch.autocast("cuda", torch.float16):
            lf, lc = model(batch["image"].to(device, non_blocking=True))
        pf, pc = lf.argmax(1).cpu(), lc.argmax(1).cpu()
        for i, cid in enumerate(batch["case_id"]):
            rec.append((cid,
                        weighted_dice_per_image(pf[i], batch["fine"][i], nf, wf).item(),
                        weighted_dice_per_image(pc[i], batch["coarse"][i], nc, wc).item()))
    cm = pd.DataFrame(rec, columns=["case_id", "dice_fine", "dice_coarse"]).groupby("case_id").mean()
    df, dc = float(cm.dice_fine.mean()), float(cm.dice_coarse.mean())
    return {"val_dice_fine": df, "val_dice_coarse": dc, "val_score": (df + dc) / 2}


class SegModule(pl.LightningModule):
    def __init__(self, cfg: dict, class_weights: dict):
        super().__init__()
        self.cfg = cfg
        m = cfg["model"]
        self.model = DualHeadUnetPP(
            encoder_name=m["encoder_name"],
            encoder_weights=m["encoder_weights"],
            decoder_channels=tuple(m["decoder_channels"]),
            num_classes_fine=m["num_classes_fine"],
            num_classes_coarse=m["num_classes_coarse"],
            img_size=(cfg["data"]["img_h"], cfg["data"]["img_w"]),
        )
        lc = cfg["loss"]
        self.loss_fine = build_loss(lc["name"], class_weights["fine"])
        self.loss_coarse = build_loss(lc["name"], class_weights["coarse"])
        self.w1, self.w2 = float(lc["task1_weight"]), float(lc["task2_weight"])
        # fine logits を merged_id 単位で確率合算 -> coarse GT に対する階層一貫性 loss
        self.w_f2c = float(lc.get("fine2coarse_weight", 0.0))
        if self.w_f2c > 0:
            import pandas as pd
            lm = pd.read_csv(REPO / "data" / "labelmap.csv")
            M = torch.zeros(len(class_weights["fine"]), len(class_weights["coarse"]))
            for r in lm.itertuples():
                M[int(r.fine_id), int(r.merged_id)] = 1.0
            self.register_buffer("f2c_mat", M)
            self.loss_f2c = DiceLoss(
                softmax=False, to_onehot_y=True, include_background=True,
                weight=torch.tensor(class_weights["coarse"], dtype=torch.float32),
            )
        self.register_buffer("wf", torch.tensor(class_weights["fine"], dtype=torch.float32))
        self.register_buffer("wc", torch.tensor(class_weights["coarse"], dtype=torch.float32))
        self._val_records: list[tuple[str, float, float]] = []
        self.history: list[dict] = []

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, _):
        lf_logit, lc_logit = self.model(batch["image"])
        loss_f = self.loss_fine(lf_logit, batch["fine"].unsqueeze(1))
        loss_c = self.loss_coarse(lc_logit, batch["coarse"].unsqueeze(1))
        loss = self.w1 * loss_f + self.w2 * loss_c
        logs = {"train/loss_fine": loss_f, "train/loss_coarse": loss_c}
        if self.w_f2c > 0:
            probs_c_from_f = torch.einsum("bfhw,fc->bchw", lf_logit.softmax(1), self.f2c_mat)
            loss_f2c = self.loss_f2c(probs_c_from_f, batch["coarse"].unsqueeze(1))
            loss = loss + self.w_f2c * loss_f2c
            logs["train/loss_f2c"] = loss_f2c
        logs["train/loss"] = loss
        self.log_dict(
            logs,
            on_step=True, on_epoch=True, prog_bar=True, batch_size=batch["image"].size(0),
        )
        return loss

    def validation_step(self, batch, _):
        lf_logit, lc_logit = self.model(batch["image"])
        pred_f = lf_logit.argmax(1)
        pred_c = lc_logit.argmax(1)
        nf = self.cfg["model"]["num_classes_fine"]
        nc = self.cfg["model"]["num_classes_coarse"]
        for i, case_id in enumerate(batch["case_id"]):
            df = weighted_dice_per_image(pred_f[i], batch["fine"][i], nf, self.wf)
            dc = weighted_dice_per_image(pred_c[i], batch["coarse"][i], nc, self.wc)
            self._val_records.append((case_id, df.item(), dc.item()))

    def on_validation_epoch_end(self):
        if not self._val_records:
            return
        df = pd.DataFrame(self._val_records, columns=["case_id", "dice_fine", "dice_coarse"])
        self._val_records = []
        # 公式集約: 画像 → case 平均 → 全体平均
        case_mean = df.groupby("case_id").mean()
        dice_fine = float(case_mean.dice_fine.mean())
        dice_coarse = float(case_mean.dice_coarse.mean())
        score = (dice_fine + dice_coarse) / 2
        self.log_dict(
            {"val/dice_fine": dice_fine, "val/dice_coarse": dice_coarse, "val/score": score},
            prog_bar=True,
        )
        self.history.append(
            {"epoch": self.current_epoch, "val_dice_fine": dice_fine,
             "val_dice_coarse": dice_coarse, "val_score": score}
        )
        log.info("epoch %d | val dice_fine=%.4f dice_coarse=%.4f score=%.4f",
                 self.current_epoch, dice_fine, dice_coarse, score)

    def configure_optimizers(self):
        tc = self.cfg["train"]
        opt = torch.optim.AdamW(
            self.parameters(), lr=float(tc["lr"]), weight_decay=float(tc["weight_decay"])
        )
        warmup = int(tc["warmup_epochs"])
        sched = SequentialLR(
            opt,
            [LinearLR(opt, start_factor=0.01, total_iters=warmup),
             CosineAnnealingLR(opt, T_max=max(1, int(tc["epochs"]) - warmup))],
            milestones=[warmup],
        )
        return {"optimizer": opt, "lr_scheduler": {"scheduler": sched, "interval": "epoch"}}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fold", type=int, required=True)
    ap.add_argument("--config", default=str(Path(__file__).parent / "config.yaml"))
    ap.add_argument("--epochs", type=int, default=None, help="config の epochs を上書き (smoke test 用)")
    ap.add_argument("--limit-batches", type=float, default=None, help="smoke test 用")
    ap.add_argument("--smoke", action="store_true", help="experiment 名に _smoke を付けて本走と分離")
    ap.add_argument("--swa-start", type=int, default=None, help="config の train.swa.start_epoch を上書き")
    args = ap.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text())
    if args.epochs is not None:
        cfg["train"]["epochs"] = args.epochs
    if args.swa_start is not None:
        cfg["train"].setdefault("swa", {})["start_epoch"] = args.swa_start
    if args.smoke:
        cfg["experiment"]["name"] += "_smoke"

    def pick_resume(d: Path) -> Path | None:
        """レジューム地点: 毎 epoch 更新の latest.ckpt を優先、無ければ last.ckpt (= best のコピー)。"""
        for name in ("latest.ckpt", "last.ckpt"):
            if (d / name).exists():
                return d / name
        return None

    out_dir = REPO / cfg["paths"]["results_root"] / cfg["experiment"]["name"] / f"fold{args.fold}"
    if out_dir.exists() and any(out_dir.iterdir()) and pick_resume(out_dir) is None:
        # 完了済み/中途半端なディレクトリは上書きしない (CLAUDE.md)
        n = 1
        while (alt := out_dir.with_name(f"{out_dir.name}_{n:03d}")).exists():
            n += 1
        out_dir = alt
    resume_ckpt = pick_resume(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    setup_logging(out_dir)
    shutil.copy(args.config, out_dir / "config.yaml")

    pl.seed_everything(cfg["experiment"]["seed"], workers=True)

    folds = pd.read_csv(REPO / cfg["cv"]["folds_csv"])
    train_df = folds[folds.fold != args.fold]
    val_df = folds[folds.fold == args.fold]
    log.info("fold %d | train %d imgs / val %d imgs (%d cases)",
             args.fold, len(train_df), len(val_df), val_df.case_id.nunique())

    dl_kwargs = dict(
        batch_size=cfg["data"]["batch_size"],
        num_workers=cfg["data"]["num_workers"],
        pin_memory=True,
        persistent_workers=cfg["data"]["num_workers"] > 0,
    )
    train_dl = DataLoader(TigerDataset(train_df, REPO, cfg, train=True), shuffle=True,
                          drop_last=True, **dl_kwargs)
    val_dl = DataLoader(TigerDataset(val_df, REPO, cfg, train=False), shuffle=False, **dl_kwargs)

    class_weights = json.loads((REPO / cfg["paths"]["class_weights"]).read_text())
    module = SegModule(cfg, class_weights)

    loggers = [CSVLogger(save_dir=out_dir, name="csv_logs")]
    wandb_cfg = cfg.get("wandb", {})
    if wandb_cfg.get("enabled") and not args.smoke:
        loggers.append(
            WandbLogger(
                project=wandb_cfg["project"],
                name=cfg["experiment"]["name"],  # 全 fold で同じ run 名
                group=str(args.fold),            # fold 分けは group
                save_dir=str(out_dir),
                config=cfg,
            )
        )

    ckpt_cb = ModelCheckpoint(
        dirpath=out_dir, filename="best", monitor="val/score", mode="max",
        save_last=True, save_top_k=1,
    )
    # 注意: Lightning 2.x の save_last は「best が保存された時のコピー」でしかない。
    # 毎 epoch のレジューム地点は monitor=None の rolling checkpoint で別途持つ。
    latest_cb = ModelCheckpoint(
        dirpath=out_dir, filename="latest", monitor=None, save_top_k=1, every_n_epochs=1,
    )
    swa_cfg = cfg["train"].get("swa", {})
    swa_cb = None
    callbacks = [ckpt_cb, latest_cb, LearningRateMonitor()]
    if swa_cfg.get("enabled"):
        swa_cb = CheckpointAveraging(out_dir, int(swa_cfg["start_epoch"]))
        callbacks.append(swa_cb)
        log.info("checkpoint averaging: start_epoch=%d bn_update=%s",
                 swa_cb.start_epoch, swa_cfg.get("bn_update", True))

    trainer = pl.Trainer(
        logger=loggers,
        accelerator="gpu",
        devices=1,
        precision=cfg["train"]["precision"],
        max_epochs=cfg["train"]["epochs"],
        accumulate_grad_batches=cfg["train"]["accumulate_grad_batches"],
        callbacks=callbacks,
        default_root_dir=out_dir,
        log_every_n_steps=10,
        limit_train_batches=args.limit_batches,
        limit_val_batches=args.limit_batches,
        num_sanity_val_steps=0,
    )
    if resume_ckpt is not None:
        log.info("resuming from %s", resume_ckpt)
    trainer.fit(module, train_dl, val_dl, ckpt_path=str(resume_ckpt) if resume_ckpt else None)

    swa_result = None
    if swa_cb is not None and swa_cb.avg is not None:
        device = "cuda"
        module.model.load_state_dict(swa_cb.avg, strict=True)
        module.model.to(device)
        if swa_cfg.get("bn_update", True):
            nb = update_bn(module.model, train_dl, device)
            log.info("SWA: BN running stats を %d バッチで再推定", nb)
        swa_result = eval_val(module.model, val_dl, cfg, module.wf.cpu(), module.wc.cpu(), device)
        swa_result["n_averaged"] = swa_cb.n
        # predict_oof.py が読める Lightning 形式 (state_dict / "model." prefix) で保存
        torch.save({"state_dict": {f"model.{k}": v for k, v in module.model.state_dict().items()}},
                   out_dir / "swa.ckpt")
        best_score = float(ckpt_cb.best_model_score) if ckpt_cb.best_model_score is not None else -1.0
        log.info("SWA: n=%d score=%.4f (fine=%.4f coarse=%.4f) | best.ckpt=%.4f | diff=%+.4f",
                 swa_cb.n, swa_result["val_score"], swa_result["val_dice_fine"],
                 swa_result["val_dice_coarse"], best_score, swa_result["val_score"] - best_score)
        swa_cb.state_file.unlink(missing_ok=True)  # 完走したら中間退避は不要

    (out_dir / "training_log.json").write_text(json.dumps(module.history, indent=2))
    if swa_result is not None:
        (out_dir / "swa_result.json").write_text(json.dumps(swa_result, indent=2))
    log.info("done. best=%s (val/score=%.4f)", ckpt_cb.best_model_path, ckpt_cb.best_model_score or -1)


if __name__ == "__main__":
    main()
