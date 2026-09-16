"""Task1+2 同時学習 (shared encoder + dual Unet++ decoder) の学習エントリポイント.

Usage:
    python3 train.py --fold 0 [--config config.yaml] [--epochs N] [--fast-dev]

出力: {results_root}/{experiment.name}/fold{N}/
    best.ckpt / last.ckpt / config.yaml / train_YYYYmmdd_HHMMSS.log / training_log.json
last.ckpt が存在すれば自動で再開する。
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
import numpy as np
from torch.utils.data import DataLoader, WeightedRandomSampler

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


class MaskedDiceLoss(torch.nn.Module):
    """MONAI DiceLoss(softmax, to_onehot_y, include_background, weight) と等価だが,
    valid マスクが 0 の画素を分子・分母の双方から除外する。

    擬似ラベルには採用できなかった画素 (ignore) が含まれるため, それを学習に使わない。
    valid が全 1 のときは MONAI 版と数値一致する (verify_masked_dice で検証済み)。
    """

    def __init__(self, class_weights: list[int], softmax: bool = True,
                 smooth_nr: float = 1e-5, smooth_dr: float = 1e-5):
        super().__init__()
        self.register_buffer("w", torch.tensor(class_weights, dtype=torch.float32))
        self.softmax = softmax
        self.smooth_nr, self.smooth_dr = smooth_nr, smooth_dr

    def forward(self, logits: torch.Tensor, target: torch.Tensor,
                valid: torch.Tensor | None = None,
                sample_w: torch.Tensor | None = None) -> torch.Tensor:
        # logits: (B,C,H,W) / target: (B,1,H,W) long / valid: (B,H,W) float
        p = logits.softmax(1) if self.softmax else logits
        t = torch.zeros_like(p).scatter_(1, target.clamp(min=0), 1.0)
        if valid is not None:
            m = valid.unsqueeze(1)
            p, t = p * m, t * m
        dims = (2, 3)
        inter = (p * t).sum(dims)
        denom = p.sum(dims) + t.sum(dims)
        f = 1.0 - (2.0 * inter + self.smooth_nr) / (denom + self.smooth_dr)   # (B,C)
        f = f * self.w.view(1, -1)
        if sample_w is not None:
            f = f * sample_w.view(-1, 1)
            return f.sum() / (sample_w.sum() * f.size(1)).clamp(min=1e-8)
        return f.mean()


class PseudoScheduleSampler(torch.utils.data.Sampler):
    """擬似ラベルを混ぜて学習し, 最後の数 epoch は実データだけに切り替えるサンプラー.

    擬似ラベルに引っ張られたまま収束するのを避けるため, 表現学習の段階では
    ノイズを許容し, 仕上げはクリーンなラベルだけで行う。
    1 epoch の抽出数は常に実データ枚数に固定するので, ベースラインと
    最適化ステップ数が一致したままになる。
    """

    def __init__(self, real_idx, pseudo_idx, n_draw: int, frac: float,
                 real_only_last: int, total_epochs: int, seed: int = 0):
        self.real_idx = np.asarray(real_idx)
        self.pseudo_idx = np.asarray(pseudo_idx)
        self.n_draw = int(n_draw)
        self.frac = float(frac)
        self.switch_epoch = total_epochs - int(real_only_last)
        self.epoch = 0
        self.rng = np.random.default_rng(seed)

    def set_epoch(self, e: int) -> None:
        self.epoch = int(e)

    def pseudo_active(self) -> bool:
        return len(self.pseudo_idx) > 0 and self.epoch < self.switch_epoch

    def __iter__(self):
        if not self.pseudo_active():
            idx = self.rng.choice(self.real_idx, self.n_draw, replace=True)
        else:
            n_p = int(round(self.n_draw * self.frac))
            idx = np.concatenate([
                self.rng.choice(self.real_idx, self.n_draw - n_p, replace=True),
                self.rng.choice(self.pseudo_idx, n_p, replace=True),
            ])
            self.rng.shuffle(idx)
        return iter(idx.tolist())

    def __len__(self) -> int:
        return self.n_draw


class PseudoScheduleCallback(pl.Callback):
    """epoch をサンプラーに伝え, 切り替わりをログに残す。"""

    def __init__(self, sampler: PseudoScheduleSampler):
        self.sampler = sampler
        self._logged = False

    def on_train_epoch_start(self, trainer, pl_module):
        self.sampler.set_epoch(trainer.current_epoch)
        if not self.sampler.pseudo_active() and not self._logged:
            log.info("epoch %d: 擬似ラベルを外し, 実データのみの fine-tune に切り替え",
                     trainer.current_epoch)
            self._logged = True


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
        self.loss_fine = MaskedDiceLoss(class_weights["fine"])
        self.loss_coarse = MaskedDiceLoss(class_weights["coarse"])
        self.pseudo_w = float(cfg.get("pseudo", {}).get("loss_weight", 1.0))
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
            self.loss_f2c = MaskedDiceLoss(class_weights["coarse"], softmax=False)
        self.register_buffer("wf", torch.tensor(class_weights["fine"], dtype=torch.float32))
        self.register_buffer("wc", torch.tensor(class_weights["coarse"], dtype=torch.float32))
        self._val_records: list[tuple[str, float, float]] = []
        self.history: list[dict] = []

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, _):
        lf_logit, lc_logit = self.model(batch["image"])
        valid = batch.get("valid")
        # 擬似サンプルは loss 重みを下げられるようにする (実データを 1.0 とした相対値)
        sw = torch.where(batch["is_pseudo"] > 0.5,
                         torch.full_like(batch["is_pseudo"], self.pseudo_w),
                         torch.ones_like(batch["is_pseudo"]))
        loss_f = self.loss_fine(lf_logit, batch["fine"].unsqueeze(1), valid, sw)
        loss_c = self.loss_coarse(lc_logit, batch["coarse"].unsqueeze(1), valid, sw)
        loss = self.w1 * loss_f + self.w2 * loss_c
        logs = {"train/loss_fine": loss_f, "train/loss_coarse": loss_c,
                "train/pseudo_frac": batch["is_pseudo"].mean()}
        if self.w_f2c > 0:
            probs_c_from_f = torch.einsum("bfhw,fc->bchw", lf_logit.softmax(1), self.f2c_mat)
            loss_f2c = self.loss_f2c(probs_c_from_f, batch["coarse"].unsqueeze(1), valid, sw)
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
    args = ap.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text())
    if args.epochs is not None:
        cfg["train"]["epochs"] = args.epochs
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
    train_df = train_df.copy()
    train_df["is_pseudo"] = False
    pcfg = cfg.get("pseudo", {})
    sampler = None
    if pcfg.get("enabled"):
        pi = pd.read_csv(REPO / pcfg["index_csv"])
        val_cases = set(val_df.case_id)
        # 擬似ラベルは元 case の実 GT と, その case を学習に含む ens5 から作られている。
        # したがって val case 由来の擬似フレームを学習に入れるのは直接のリークになる。
        n_all = len(pi)
        pi = pi[~pi.case_id.isin(val_cases)]
        pi = pi[pi.coverage >= float(pcfg.get("min_coverage", 0.0))]
        pi = pi.assign(is_pseudo=True)[["filename", "case_id", "is_pseudo"]]
        log.info("pseudo: %d frames -> %d after removing val cases (%d) and coverage filter",
                 n_all, len(pi), len(val_cases))
        if len(pi):
            frac = float(pcfg.get("fraction", 0.5))     # 1 epoch 内で擬似が占める割合
            train_df = pd.concat([train_df[["filename", "case_id", "is_pseudo"]], pi],
                                 ignore_index=True)
            is_p = train_df.is_pseudo.values
            n_r, n_p = int((~is_p).sum()), int(is_p.sum())
            real_only_last = int(pcfg.get("real_only_last_epochs", 0))
            # ステップ数はベースラインと同じ (= 実データ枚数) に固定して比較可能にする
            sampler = PseudoScheduleSampler(
                real_idx=np.flatnonzero(~is_p), pseudo_idx=np.flatnonzero(is_p),
                n_draw=n_r, frac=frac, real_only_last=real_only_last,
                total_epochs=int(cfg["train"]["epochs"]), seed=cfg["experiment"]["seed"])
            log.info("train set: real %d + pseudo %d, draws %d/epoch (pseudo frac %.2f, "
                     "loss weight %.2f, 最後の %d epoch は実データのみ)",
                     n_r, n_p, n_r, frac, float(pcfg.get("loss_weight", 1.0)), real_only_last)
    train_dl = DataLoader(TigerDataset(train_df, REPO, cfg, train=True),
                          shuffle=(sampler is None), sampler=sampler,
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
    trainer = pl.Trainer(
        logger=loggers,
        accelerator="gpu",
        devices=1,
        precision=cfg["train"]["precision"],
        max_epochs=cfg["train"]["epochs"],
        accumulate_grad_batches=cfg["train"]["accumulate_grad_batches"],
        callbacks=[ckpt_cb, latest_cb, LearningRateMonitor()]
        + ([PseudoScheduleCallback(sampler)] if sampler is not None else []),
        default_root_dir=out_dir,
        log_every_n_steps=10,
        limit_train_batches=args.limit_batches,
        limit_val_batches=args.limit_batches,
        num_sanity_val_steps=0,
    )
    if resume_ckpt is not None:
        log.info("resuming from %s", resume_ckpt)
    trainer.fit(module, train_dl, val_dl, ckpt_path=str(resume_ckpt) if resume_ckpt else None)

    (out_dir / "training_log.json").write_text(json.dumps(module.history, indent=2))
    log.info("done. best=%s (val/score=%.4f)", ckpt_cb.best_model_path, ckpt_cb.best_model_score or -1)


if __name__ == "__main__":
    main()
