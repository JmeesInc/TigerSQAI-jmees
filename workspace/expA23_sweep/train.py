"""expA23_sweep: アーキ / loss / 擬似ラベル を 1 本のコードで振るスイープ用学習スクリプト.

expS03_pseudo/train.py（擬似ラベル対応）を土台に、
  * model を汎用ビルダ (model.build_model) に差し替え
  * loss を loss 動物園 (losses.build_loss) に差し替え
  * 解剖ルール loss (expA22) を optional で合流
  * val に **正規化 Hausdorff の proxy** を追加（公式スコアの半分は HD なのに
    これまでのゲートは Dice しか見ていなかった）
  * ゲート用に学習後 ckpt を間引く `train.prune_ckpt`
を足したもの。

Usage:
    python3 train.py --fold 0 --config configs/xxx.yaml
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import lightning.pytorch as pl
import numpy as np
import pandas as pd
import torch
import yaml
from lightning.pytorch.callbacks import LearningRateMonitor, ModelCheckpoint
from lightning.pytorch.loggers import CSVLogger, WandbLogger
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from torch.utils.data import DataLoader

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent))

from dataset import TigerDataset  # noqa: E402
from fast_hd import fast_binary_normalized_hausdorff  # noqa: E402
from losses import MaskedDiceLoss, build_loss  # noqa: E402
from model import build_model  # noqa: E402

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


def weighted_dice_per_image(pred, target, num_classes, weights):
    """公式 Dice 規約 (両方無し=1.0, 片方無し=0.0) の weighted マクロ平均。pred/target: (H,W) long."""
    idx = target.reshape(-1) * num_classes + pred.reshape(-1)
    cm = torch.bincount(idx, minlength=num_classes * num_classes).reshape(num_classes, num_classes)
    tp = cm.diag().float()
    n_gt = cm.sum(1).float()
    n_pred = cm.sum(0).float()
    dice = torch.where((n_gt == 0) & (n_pred == 0), torch.ones_like(tp),
                       2 * tp / (n_gt + n_pred).clamp(min=1))
    return (dice * weights).sum() / weights.sum()


def weighted_hd_per_image(pred: np.ndarray, target: np.ndarray, num_classes: int,
                          weights: np.ndarray, sub: int = 2) -> float:
    """正規化 Hausdorff の proxy（公式と同じ規約・同じ式、ただし 1/sub 解像度で計算）.

    公式は元解像度で測るので値そのものは一致しないが、**学習中の相対比較**には足りる。
    出現しないクラス (GT も pred も空) は公式どおり 0.0。
    """
    if sub > 1:
        pred, target = pred[::sub, ::sub], target[::sub, ::sub]
    diag = float(np.hypot(*pred.shape))
    present = set(np.unique(pred).tolist()) | set(np.unique(target).tolist())
    num, den = 0.0, 0.0
    for c in range(num_classes):
        w = float(weights[c])
        den += w
        if c not in present:
            continue  # 両方空 = 0.0
        num += w * fast_binary_normalized_hausdorff(pred == c, target == c, diag)
    return num / max(den, 1e-8)


class PseudoScheduleSampler(torch.utils.data.Sampler):
    """擬似ラベルを混ぜ、最後の数 epoch は実データだけに切り替えるサンプラー (expS03 と同一)。"""

    def __init__(self, real_idx, pseudo_idx, n_draw, frac, real_only_last, total_epochs, seed=0):
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
    def __init__(self, sampler: PseudoScheduleSampler):
        self.sampler = sampler
        self._logged = False

    def on_train_epoch_start(self, trainer, pl_module):
        self.sampler.set_epoch(trainer.current_epoch)
        if not self.sampler.pseudo_active() and not self._logged:
            log.info("epoch %d: 擬似ラベルを外し、実データのみの fine-tune に切り替え",
                     trainer.current_epoch)
            self._logged = True


class SegModule(pl.LightningModule):
    def __init__(self, cfg: dict, class_weights: dict):
        super().__init__()
        self.cfg = cfg
        m = cfg["model"]
        self.model = build_model(m, (cfg["data"]["img_h"], cfg["data"]["img_w"]))
        # 既存レシピの重みから始める (タスク特化の fine-tune 用)。
        # `model.init_from: <config 名>` を指定すると results/<名>/fold{k}/best_fp16.pt を読む
        if m.get("init_from"):
            src = (REPO / cfg["paths"]["results_root"] / m["init_from"]
                   / f"fold{cfg['_fold']}" / "best_fp16.pt")
            assert src.exists(), f"init_from の重みが無い: {src}"
            sd = torch.load(src, map_location="cpu", weights_only=False)
            sd = {k.removeprefix("model."): v.float() for k, v in sd.items()
                  if k.startswith("model.")}
            self.model.load_state_dict(sd, strict=True)
            log.info("init_from %s fold%d の重みから開始", m["init_from"], cfg["_fold"])
        # `model.init_from_partial: <results 下のパス>` は head を除いた重み (外部データ事前学習)
        if m.get("init_from_partial"):
            src = REPO / cfg["paths"]["results_root"] / m["init_from_partial"]
            assert src.exists(), f"init_from_partial の重みが無い: {src}"
            sd = {k: v.float() for k, v in torch.load(src, map_location="cpu").items()}
            # head はクラス数が違うので落とす（外部データは 12-13 クラス）
            cur = self.model.state_dict()
            sd = {k: v for k, v in sd.items() if k in cur and cur[k].shape == v.shape}
            missing, unexpected = self.model.load_state_dict(sd, strict=False)
            log.info("init_from_partial %s: 読込 %d / 未設定 %d / 余分 %d",
                     m["init_from_partial"], len(sd), len(missing), len(unexpected))
            assert len(unexpected) == 0, f"想定外のキー: {unexpected[:5]}"

        lc = cfg["loss"]
        params = dict(lc.get("params", {}) or {})
        self.loss_fine = build_loss(lc["name"], class_weights["fine"], **params)
        self.loss_coarse = build_loss(lc["name"], class_weights["coarse"], **params)
        self.w1, self.w2 = float(lc["task1_weight"]), float(lc["task2_weight"])
        self.pseudo_w = float(cfg.get("pseudo", {}).get("loss_weight", 1.0))

        # fine logits を merged_id 単位で確率合算 -> coarse GT に対する階層一貫性 loss (expA06)
        self.w_f2c = float(lc.get("fine2coarse_weight", 0.0))
        if self.w_f2c > 0:
            lm = pd.read_csv(REPO / "data" / "labelmap.csv")
            M = torch.zeros(len(class_weights["fine"]), len(class_weights["coarse"]))
            for r in lm.itertuples():
                M[int(r.fine_id), int(r.merged_id)] = 1.0
            self.register_buffer("f2c_mat", M)
            self.loss_f2c = MaskedDiceLoss(class_weights["coarse"], softmax=False)

        # 解剖ルール loss (expA22)。fold 別ルール行列 = 学習側 GT のみから作るのでリーク無し
        rc = cfg.get("rules", {}) or {}
        self.rules_on = bool(rc.get("enabled", False))
        if self.rules_on:
            from anatomy_rules import AnatomyRuleLoss
            fold = int(cfg["_fold"])
            self.rule_fine = AnatomyRuleLoss(REPO / rc["rules_dir"] / f"rules_fine_fold{fold}.npz",
                                             presence_pool=int(rc.get("presence_pool", 16)))
            self.rule_coarse = AnatomyRuleLoss(REPO / rc["rules_dir"] / f"rules_coarse_fold{fold}.npz",
                                               presence_pool=int(rc.get("presence_pool", 16)))
            self.w_adj = float(rc.get("adj_weight", 1.0))
            self.w_excl = float(rc.get("excl_weight", 0.1))
            self.rule_start = int(rc.get("start_epoch", 0))
            self.rule_ramp = max(1, int(rc.get("ramp_epochs", 1)))
            self.w_rule_coarse = float(rc.get("coarse_scale", 1.0))

        self.register_buffer("wf", torch.tensor(class_weights["fine"], dtype=torch.float32))
        self.register_buffer("wc", torch.tensor(class_weights["coarse"], dtype=torch.float32))
        self.hd_last_n = int(cfg.get("val", {}).get("hd_last_epochs", 3))
        self.hd_sub = int(cfg.get("val", {}).get("hd_subsample", 2))
        self._val_records: list[tuple] = []
        self.history: list[dict] = []

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, _):
        lf_logit, lc_logit = self.model(batch["image"])
        valid = batch.get("valid")
        is_p = batch.get("is_pseudo")
        sw = None
        if is_p is not None and self.pseudo_w != 1.0:
            sw = torch.where(is_p > 0.5, torch.full_like(is_p, self.pseudo_w), torch.ones_like(is_p))
        loss_f = self.loss_fine(lf_logit, batch["fine"].unsqueeze(1), valid, sw,
                                batch.get("dist_fine"))
        loss_c = self.loss_coarse(lc_logit, batch["coarse"].unsqueeze(1), valid, sw,
                                  batch.get("dist_coarse"))
        loss = self.w1 * loss_f + self.w2 * loss_c
        logs = {"train/loss_fine": loss_f, "train/loss_coarse": loss_c}
        if self.w_f2c > 0:
            probs_c_from_f = torch.einsum("bfhw,fc->bchw", lf_logit.float().softmax(1), self.f2c_mat)
            loss_f2c = self.loss_f2c(probs_c_from_f, batch["coarse"].unsqueeze(1), valid, sw)
            loss = loss + self.w_f2c * loss_f2c
            logs["train/loss_f2c"] = loss_f2c
        if self.rules_on:
            ramp = min(1.0, max(0.0, (self.current_epoch - self.rule_start + 1) / self.rule_ramp))
            rf = self.rule_fine(lf_logit)
            rcl = self.rule_coarse(lc_logit)
            loss = loss + ramp * (self.w_adj * rf["adj"] + self.w_excl * rf["excl"]
                                  + self.w_rule_coarse * (self.w_adj * rcl["adj"]
                                                          + self.w_excl * rcl["excl"]))
            logs["train/rule_adj_fine"] = rf["adj"]
        logs["train/loss"] = loss
        self.log_dict(logs, on_step=True, on_epoch=True, prog_bar=True,
                      batch_size=batch["image"].size(0))
        return loss

    def _hd_epoch(self) -> bool:
        return self.current_epoch >= self.cfg["train"]["epochs"] - self.hd_last_n

    def validation_step(self, batch, _):
        lf_logit, lc_logit = self.model(batch["image"])
        pred_f = lf_logit.argmax(1)
        pred_c = lc_logit.argmax(1)
        nf = self.cfg["model"]["num_classes_fine"]
        nc = self.cfg["model"]["num_classes_coarse"]
        want_hd = self._hd_epoch()
        for i, case_id in enumerate(batch["case_id"]):
            df = weighted_dice_per_image(pred_f[i], batch["fine"][i], nf, self.wf).item()
            dc = weighted_dice_per_image(pred_c[i], batch["coarse"][i], nc, self.wc).item()
            hf = hc = float("nan")
            if want_hd:
                pf = pred_f[i].cpu().numpy().astype(np.int16)
                pc = pred_c[i].cpu().numpy().astype(np.int16)
                gf = batch["fine"][i].cpu().numpy().astype(np.int16)
                gc = batch["coarse"][i].cpu().numpy().astype(np.int16)
                hf = weighted_hd_per_image(pf, gf, nf, self.wf.cpu().numpy(), self.hd_sub)
                hc = weighted_hd_per_image(pc, gc, nc, self.wc.cpu().numpy(), self.hd_sub)
            self._val_records.append((case_id, df, dc, hf, hc))

    def on_validation_epoch_end(self):
        if not self._val_records:
            return
        df = pd.DataFrame(self._val_records,
                          columns=["case_id", "dice_fine", "dice_coarse", "hd_fine", "hd_coarse"])
        self._val_records = []
        case_mean = df.groupby("case_id").mean()   # 公式集約: 画像 → case → 全体
        dice_fine = float(case_mean.dice_fine.mean())
        dice_coarse = float(case_mean.dice_coarse.mean())
        score = (dice_fine + dice_coarse) / 2
        rec = {"epoch": self.current_epoch, "val_dice_fine": dice_fine,
               "val_dice_coarse": dice_coarse, "val_score": score}
        logs = {"val/dice_fine": dice_fine, "val/dice_coarse": dice_coarse, "val/score": score}
        if case_mean.hd_fine.notna().all():
            hd_fine = float(case_mean.hd_fine.mean())
            hd_coarse = float(case_mean.hd_coarse.mean())
            rec.update(val_hd_fine=hd_fine, val_hd_coarse=hd_coarse)
            logs.update({"val/hd_fine": hd_fine, "val/hd_coarse": hd_coarse})
            log.info("epoch %d | val HD(proxy) fine=%.4f coarse=%.4f", self.current_epoch,
                     hd_fine, hd_coarse)
        self.log_dict(logs, prog_bar=True)
        self.history.append(rec)
        log.info("epoch %d | val dice_fine=%.4f dice_coarse=%.4f score=%.4f",
                 self.current_epoch, dice_fine, dice_coarse, score)

    def configure_optimizers(self):
        tc = self.cfg["train"]
        opt = torch.optim.AdamW(self.parameters(), lr=float(tc["lr"]),
                                weight_decay=float(tc["weight_decay"]))
        warmup = int(tc["warmup_epochs"])
        sched = SequentialLR(
            opt,
            [LinearLR(opt, start_factor=0.01, total_iters=warmup),
             CosineAnnealingLR(opt, T_max=max(1, int(tc["epochs"]) - warmup))],
            milestones=[warmup],
        )
        return {"optimizer": opt, "lr_scheduler": {"scheduler": sched, "interval": "epoch"}}


def ensure_rules(cfg: dict, fold: int) -> None:
    """fold 別ルール行列が無ければ GT 統計 (val 除外) から生成する (expA22 と同一)。"""
    rc = cfg["rules"]
    rules_dir = REPO / rc["rules_dir"]
    ag = REPO / "workspace/anatomy_graph"
    for task in ("fine", "coarse"):
        if (rules_dir / f"rules_{task}_fold{fold}.npz").exists():
            continue
        if not (rules_dir / f"{task}_fold{fold}" / "graph.json").exists():
            subprocess.run([sys.executable, str(ag / "build_graph.py"), "--task", task,
                            "--fold", str(fold), "--folds-csv", cfg["cv"]["folds_csv"]], check=True)
        subprocess.run([sys.executable, str(ag / "rules.py"), "--task", task, "--fold", str(fold)],
                       check=True)


def export_fp16(ckpt: Path, out: Path) -> None:
    """Lightning ckpt から **重みだけ fp16** を取り出す (optimizer state を捨てて 1/6 以下に)。"""
    sd = torch.load(ckpt, map_location="cpu", weights_only=False)["state_dict"]
    sd = {k: (v.half() if v.is_floating_point() else v) for k, v in sd.items()}
    torch.save(sd, out)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fold", type=int, required=True)
    ap.add_argument("--config", default=str(Path(__file__).parent / "configs" / "base.yaml"))
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--limit-batches", type=float, default=None)
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text())
    cfg["_fold"] = args.fold
    if args.epochs is not None:
        cfg["train"]["epochs"] = args.epochs
    if args.smoke:
        cfg["experiment"]["name"] += "_smoke"
    if cfg.get("rules", {}).get("enabled"):
        ensure_rules(cfg, args.fold)

    def pick_resume(d: Path) -> Path | None:
        for name in ("latest.ckpt", "last.ckpt"):
            if (d / name).exists():
                return d / name
        return None

    out_dir = REPO / cfg["paths"]["results_root"] / cfg["experiment"]["name"] / f"fold{args.fold}"
    if out_dir.exists() and any(out_dir.iterdir()) and pick_resume(out_dir) is None:
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
    train_df = folds[folds.fold != args.fold].copy()
    # 学習データ量の効果を測るため、学習側の fold を間引けるようにする
    # (val fold は固定。`cv.train_fold_limit: 3` なら学習 4 fold のうち 3 つだけ使う)
    # 全データ学習: val fold も学習に含める (held-out 無し。検証不可なので用途は提出のみ)
    if cfg["cv"].get("use_all_data"):
        train_df = folds.copy()
        log.info("use_all_data: 全 %d 枚で学習 (val は参考値。held-out 無し)", len(train_df))
    lim = cfg["cv"].get("train_fold_limit")
    if lim:
        keep = sorted(f for f in folds.fold.unique() if f != args.fold)[:int(lim)]
        train_df = train_df[train_df.fold.isin(keep)]
        log.info("train_fold_limit=%s -> 学習 fold %s のみ使用 (%d 枚)",
                 lim, keep, len(train_df))
    val_df = folds[folds.fold == args.fold]
    log.info("fold %d | train %d imgs / val %d imgs (%d cases)",
             args.fold, len(train_df), len(val_df), val_df.case_id.nunique())

    dl_kwargs = dict(batch_size=cfg["data"]["batch_size"], num_workers=cfg["data"]["num_workers"],
                     pin_memory=True, persistent_workers=cfg["data"]["num_workers"] > 0)

    train_df["is_pseudo"] = False
    pcfg = cfg.get("pseudo", {}) or {}
    sampler = None
    if pcfg.get("enabled"):
        pi = pd.read_csv(REPO / pcfg["index_csv"])
        val_cases = set(val_df.case_id)
        n_all = len(pi)
        # 擬似ラベルは元 case の実 GT と ens5 予測から作られている
        # -> val case 由来の擬似フレームを学習に入れるのは直接のリーク
        pi = pi[~pi.case_id.isin(val_cases)]
        pi = pi[pi.coverage >= float(pcfg.get("min_coverage", 0.0))]
        pi = pi.assign(is_pseudo=True)[["filename", "case_id", "is_pseudo"]]
        log.info("pseudo: %d frames -> %d (val case %d 件と coverage で除外後)",
                 n_all, len(pi), len(val_cases))
        if len(pi):
            train_df = pd.concat([train_df[["filename", "case_id", "is_pseudo"]], pi],
                                 ignore_index=True)
            is_p = train_df.is_pseudo.values
            n_r = int((~is_p).sum())
            sampler = PseudoScheduleSampler(
                real_idx=np.flatnonzero(~is_p), pseudo_idx=np.flatnonzero(is_p),
                n_draw=n_r, frac=float(pcfg.get("fraction", 0.5)),
                real_only_last=int(pcfg.get("real_only_last_epochs", 0)),
                total_epochs=int(cfg["train"]["epochs"]), seed=cfg["experiment"]["seed"])
            log.info("train: real %d + pseudo %d, %d draws/epoch (frac %.2f, loss_w %.2f)",
                     n_r, int(is_p.sum()), n_r, float(pcfg.get("fraction", 0.5)),
                     float(pcfg.get("loss_weight", 1.0)))

    # 希少クラス（weight=3, 出現 100 枚未満）を含むフレームを重点サンプリングする。
    # OOF 目視で Pulmonary Artery / 左反回神経 などが Dice 0.000（一度も当たらない）と分かっており、
    # 公式規約では「GT にあるのに出さない」= そのクラス 0 点なので、ここが最大の失点源。
    rare = cfg["data"].get("rare_oversample", 0.0)
    if rare and sampler is None:
        inv = pd.read_csv(REPO / "workspace/expT04_task3_sweep/features_gt_fix.csv").set_index("case_id")
        lm = pd.read_csv(REPO / "data/labelmap.csv").drop_duplicates("fine_id")
        w3 = [int(r.fine_id) for _, r in lm.iterrows() if int(r.weight) == 3]
        pres = {c: inv[f"f_has_{c}"] for c in w3 if f"f_has_{c}" in inv.columns}
        freq = {c: float(v.mean()) for c, v in pres.items()}
        stems = [f.rsplit(".", 1)[0] for f in train_df.filename]
        w = np.ones(len(stems), dtype=np.float64)
        for c, v in pres.items():
            if freq[c] <= 0 or freq[c] > 0.2:      # 出現 20% 超のクラスは希少ではない
                continue
            hit = np.array([float(v.get(st, 0.0)) > 0 for st in stems])
            w[hit] *= (1.0 + rare * (0.2 / max(freq[c], 1e-3) - 1.0))
        w = np.clip(w, 1.0, 8.0)
        sampler = torch.utils.data.WeightedRandomSampler(torch.as_tensor(w), len(w), replacement=True)
        log.info("rare_oversample=%.2f: 重み中央値 %.2f / 最大 %.2f / 重み>1.5 の枚数 %d",
                 rare, float(np.median(w)), float(w.max()), int((w > 1.5).sum()))

    train_dl = DataLoader(TigerDataset(train_df, REPO, cfg, train=True),
                          shuffle=(sampler is None), sampler=sampler, drop_last=True, **dl_kwargs)
    val_dl = DataLoader(TigerDataset(val_df, REPO, cfg, train=False), shuffle=False, **dl_kwargs)

    class_weights = json.loads((REPO / cfg["paths"]["class_weights"]).read_text())
    module = SegModule(cfg, class_weights)

    loggers = [CSVLogger(save_dir=out_dir, name="csv_logs")]
    wandb_cfg = cfg.get("wandb", {})
    # dl2 には wandb が入っていない。WANDB_MODE=disabled か import 不可なら黙って CSV だけにする
    if (wandb_cfg.get("enabled") and not args.smoke
            and os.environ.get("WANDB_MODE") != "disabled"):
        try:
            loggers.append(WandbLogger(project=wandb_cfg["project"], name=cfg["experiment"]["name"],
                                       group=str(args.fold), save_dir=str(out_dir), config=cfg))
        except Exception as e:  # noqa: BLE001
            log.warning("wandb を無効化して続行: %s", e)

    ckpt_cb = ModelCheckpoint(dirpath=out_dir, filename="best", monitor="val/score", mode="max",
                              save_last=False, save_top_k=1)
    # Lightning の save_last は「best 保存時のコピー」でしかない。
    # 毎 epoch のレジューム地点は monitor=None の rolling checkpoint で別途持つ (CLAUDE.md)
    latest_cb = ModelCheckpoint(dirpath=out_dir, filename="latest", monitor=None,
                                save_top_k=1, every_n_epochs=1)
    trainer = pl.Trainer(
        logger=loggers, accelerator="gpu", devices=1,
        precision=cfg["train"]["precision"], max_epochs=cfg["train"]["epochs"],
        accumulate_grad_batches=cfg["train"]["accumulate_grad_batches"],
        callbacks=[ckpt_cb, latest_cb, LearningRateMonitor()]
        # 擬似ラベル用サンプラーのときだけ epoch 通知が要る
        # （希少クラス重点の WeightedRandomSampler には set_epoch が無い）
        + ([PseudoScheduleCallback(sampler)]
           if isinstance(sampler, PseudoScheduleSampler) else []),
        default_root_dir=out_dir, log_every_n_steps=10,
        limit_train_batches=args.limit_batches, limit_val_batches=args.limit_batches,
        num_sanity_val_steps=0, gradient_clip_val=cfg["train"].get("grad_clip"),
    )
    if resume_ckpt is not None:
        log.info("resuming from %s", resume_ckpt)
    trainer.fit(module, train_dl, val_dl, ckpt_path=str(resume_ckpt) if resume_ckpt else None)

    (out_dir / "training_log.json").write_text(json.dumps(module.history, indent=2))

    # スイープ集計用の 1 行サマリ
    hist = module.history
    best = max(hist, key=lambda r: r["val_score"]) if hist else {}
    tail = [r["val_score"] for r in hist[-5:]]
    summary = {
        "name": cfg["experiment"]["name"], "fold": args.fold,
        "best_score": best.get("val_score"), "best_epoch": best.get("epoch"),
        "tail5_mean": float(np.mean(tail)) if tail else None,
        "hd_fine": hist[-1].get("val_hd_fine") if hist else None,
        "hd_coarse": hist[-1].get("val_hd_coarse") if hist else None,
        "dice_fine": best.get("val_dice_fine"), "dice_coarse": best.get("val_dice_coarse"),
        "arch": cfg["model"].get("arch") or cfg["model"].get("hub_id"),
        "encoder": cfg["model"].get("encoder_name"), "loss": cfg["loss"]["name"],
        "finished": datetime.now().isoformat(timespec="seconds"),
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))

    # ckpt 衛生: ゲート run は fp16 の重みだけ残して 8GB/run の消費を止める
    best_ckpt = Path(ckpt_cb.best_model_path)
    if best_ckpt.exists():
        export_fp16(best_ckpt, out_dir / "best_fp16.pt")
    # 全データ学習は val が学習データで best の根拠が無い → 最終 epoch も fp16 で残す
    if cfg["cv"].get("use_all_data") and (out_dir / "latest.ckpt").exists():
        export_fp16(out_dir / "latest.ckpt", out_dir / "last_fp16.pt")
    if cfg["train"].get("prune_ckpt"):
        for f in (best_ckpt, out_dir / "latest.ckpt", out_dir / "last.ckpt"):
            if f and Path(f).exists():
                Path(f).unlink()
        log.info("prune_ckpt: best/latest ckpt を削除し best_fp16.pt のみ残した")
    log.info("done. best=%s (val/score=%.4f)", ckpt_cb.best_model_path,
             ckpt_cb.best_model_score or -1)


if __name__ == "__main__":
    main()
