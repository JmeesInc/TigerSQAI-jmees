"""expA23: スイープ用 config を生成する.

**基準 (BASE)** = expA19 l_384 = 現時点の fold0 最良 0.6821
  convnext_large.fb_in22k_ft_in1k_384 + Unet++ dual head / 1024x576 / 20ep /
  dice + f2c 0.25 / fold v2

1 本の config で **1 つの軸だけ**を BASE から動かす。比較が壊れるので複数軸を同時に
動かした config は作らない（組合せは fold0 ゲートを通ったもの同士でのみ後から作る）。

Usage:
    python3 make_configs.py            # configs/*.yaml を生成
    python3 make_configs.py --list     # 生成される arm の一覧だけ表示
"""

from __future__ import annotations

import argparse
import copy
from pathlib import Path

import yaml

HERE = Path(__file__).resolve().parent
CFG_DIR = HERE / "configs"

CONVNEXT_L384 = "tu-convnext_large.fb_in22k_ft_in1k_384"

BASE = {
    "experiment": {"name": "expA23_base", "seed": 42},
    "paths": {
        "images_dir": "workspace/data_proc/images_1024",
        "labels_fine_dir": "workspace/data_proc/labels_fine_1024",
        "labels_coarse_dir": "workspace/data_proc/labels_coarse_1024",
        "images_pseudo_dir": "workspace/data_proc/images_pseudo_1024",
        "labels_fine_pseudo_dir": "workspace/data_proc/labels_fine_pseudo_1024",
        "labels_coarse_pseudo_dir": "workspace/data_proc/labels_coarse_pseudo_1024",
        "class_weights": "workspace/data_proc/class_weights.json",
        "results_root": "workspace/expA23_sweep/results",
        "orig_images_dir": "data/images",
        "labelmap_csv": "data/labelmap.csv",
    },
    "cv": {"folds_csv": "workspace/fold/v2/folds.csv"},
    "wandb": {"enabled": True, "project": "tigersqai"},
    "model": {
        "source": "timm",
        "arch": "unetplusplus",
        "encoder_name": CONVNEXT_L384,
        "encoder_weights": "imagenet",
        "decoder_channels": [256, 128, 64, 32, 16],
        "num_classes_fine": 31,
        "num_classes_coarse": 16,
    },
    "data": {"img_h": 576, "img_w": 1024, "batch_size": 2, "num_workers": 8,
             "aug": "normal", "hflip": True},
    "loss": {"name": "dice", "task1_weight": 0.5, "task2_weight": 0.5,
             "fine2coarse_weight": 0.25},
    "val": {"hd_last_epochs": 3, "hd_subsample": 2},
    "train": {"epochs": 20, "lr": 2.0e-4, "weight_decay": 0.01, "warmup_epochs": 3,
              "accumulate_grad_batches": 4, "precision": "16-mixed", "prune_ckpt": False},
}


def hub(name: str, hub_id: str, lr: float) -> dict:
    """smp-hub 経路は arch/encoder_name を使わないので、紛らわしいので落としておく。"""
    cfg = arm(name, **{"model": {"source": "smp_hub", "hub_id": hub_id}, "train.lr": lr})
    for k in ("arch", "encoder_name", "encoder_weights", "decoder_channels"):
        cfg["model"].pop(k, None)
    return cfg


def arm(name: str, **patch) -> dict:
    """BASE を深いコピーして、ドット区切りのキーだけ差し替える。"""
    cfg = copy.deepcopy(BASE)
    cfg["experiment"]["name"] = f"expA23_{name}"
    for k, v in patch.items():
        node = cfg
        *parents, leaf = k.split(".")
        for p in parents:
            node = node.setdefault(p, {})
        if isinstance(v, dict) and isinstance(node.get(leaf), dict):
            node[leaf].update(v)
        else:
            node[leaf] = v
    return cfg


# ---------------------------------------------------------------- 軸 1: デコーダ
# encoder は BASE 固定。Unet++ 以外は smp.create_model 経路 (model.py の経路 2)
DECODER_ARMS = [
    arm("d_base"),                                   # 実装等価性の確認 (A19 l_384 = 0.6821 を再現するはず)
    arm("d_unetpp_scse", **{"model.attention": "scse"}),
    arm("d_upernet", **{"model.arch": "upernet"}),
    arm("d_fpn", **{"model.arch": "fpn"}),
    arm("d_manet", **{"model.arch": "manet"}),
    arm("d_deeplabv3p", **{"model.arch": "deeplabv3plus"}),
]

# ---------------------------------------------------------------- 軸 2: エンコーダ
# CNN / MetaFormer 系は Unet++ のまま。ViT 系はピラミッドを持たないので UPerNet と組む
ENCODER_ARMS = [
    # l_384 (384 事前学習タグ) が勝ったので、同じ 384 タグの一段大きい encoder も見る
    arm("e_convnext_xl_384", **{"model.encoder_name": "tu-convnext_xlarge.fb_in22k_ft_in1k_384"}),
    arm("e_caformer_m36", **{"model.encoder_name": "tu-caformer_m36.sail_in22k_ft_in1k_384"}),
    arm("e_convnextv2_base", **{"model.encoder_name": "tu-convnextv2_base.fcmae_ft_in22k_in1k_384"}),
    arm("e_seresnext101", **{"model.encoder_name": "tu-seresnextaa101d_32x8d.sw_in12k_ft_in1k"}),
    arm("e_swinv2_base", **{"model.arch": "upernet",
                            "model.encoder_name": "tu-swinv2_base_window12to24_192to384.ms_in22k_ft_in1k"}),
    arm("e_mitb5", **{"model.encoder_name": "mit_b5", "model.encoder_weights": "imagenet"}),
    arm("e_beitv2_large", **{"model.arch": "upernet",
                             "model.encoder_name": "tu-beitv2_large_patch16_224.in1k_ft_in22k_in1k"}),
]

# ---------------------------------------------------------------- 軸 3: smp-hub (decoder ごと事前学習)
HUB_ARMS = [
    hub("h_upernet_swin_l", "smp-hub/upernet-swin-large", 1.0e-4),
    hub("h_upernet_convnext_l", "smp-hub/upernet-convnext-large", 1.0e-4),
    hub("h_segformer_b5", "smp-hub/segformer-b5-640x640-ade-160k", 6.0e-5),
    hub("h_dpt_large", "smp-hub/dpt-large-ade20k", 6.0e-5),
]

# ---------------------------------------------------------------- 軸 4: loss
LOSS_ARMS = [
    arm("l_dicece", **{"loss.name": "dicece", "loss.params": {"extra_weight": 1.0}}),
    arm("l_dicefocal", **{"loss.name": "dicefocal", "loss.params": {"extra_weight": 1.0, "gamma": 2.0}}),
    arm("l_dicedet", **{"loss.name": "dicedet",
                        "loss.params": {"extra_weight": 1.0, "threshold": 0.5}}),
    arm("l_sizeweighted", **{"loss.name": "sizeweighted",
                             "loss.params": {"alpha": 0.5, "neg_weight": 1.0}}),
    arm("l_focaltversky", **{"loss.name": "focaltversky",
                             "loss.params": {"alpha": 0.3, "beta": 0.7, "gamma": 0.75}}),
    # HD 直撃枠。MONAI の HausdorffDTLoss は CPU 距離変換で 8s/step (20ep で 12 時間) と
    # 使い物にならなかったので、GT 距離マップを dataset 側で 1/4 解像度に前計算して
    # 「GT から遠い確率」を叩く boundary loss に置き換えた
    arm("l_boundary", **{"loss.name": "diceboundary", "loss.params": {"extra_weight": 1.0},
                         "data.dist_maps": {"enabled": True, "downsample": 4}}),
    arm("l_dicermi", **{"loss.name": "dicermi",
                        "loss.params": {"extra_weight": 0.5, "pool_stride": 4}}),
    arm("l_f2c05", **{"loss.fine2coarse_weight": 0.5}),
    arm("l_rules", **{"rules": {"enabled": True, "rules_dir": "workspace/anatomy_graph/out",
                                "adj_weight": 1.0, "excl_weight": 0.1, "start_epoch": 8,
                                "ramp_epochs": 5, "coarse_scale": 1.0, "presence_pool": 16}}),
]

# ---------------------------------------------------------------- 軸 5': 学習長 / lr
# 200M パラメータの encoder を 20ep / lr2e-4 で回しているが、この設定は
# convnext_base 時代に決めたもの。large でも最適かは測っていない
SCHED_ARMS = [
    arm("x_lr1e4", **{"train.lr": 1.0e-4}),
    arm("x_ep30", **{"train.epochs": 30}),
]

# ---------------------------------------------------------------- 組合せ（fold0 を通ったもの同士）
# h_upernet_swin_l が fold0 0.6921 (tail5 0.6891) で基準 0.6821 を明確に超えた。
# ただし HD は悪い (hd_fine 0.297 vs boundary 0.255) ので、HD 狙いの項と組ませる。
def hub_arm(name: str, hub_id: str, lr: float, **patch) -> dict:
    cfg = hub(name, hub_id, lr)
    for k, v in patch.items():
        node = cfg
        *parents, leaf = k.split(".")
        for p_ in parents:
            node = node.setdefault(p_, {})
        if isinstance(v, dict) and isinstance(node.get(leaf), dict):
            node[leaf].update(v)
        else:
            node[leaf] = v
    return cfg


COMBO_ARMS = [
    # --- loss 同士 / loss × aug の組合せ (2 fold ゲート) ---
    # これまで組合せは 3 戦 3 敗だが、いずれも「同じ失敗モードを直す組合せ」だった。
    # ここでは機序が独立しているものを選ぶ:
    #   dicedet = クラスの存在判定 / rules = 空間配置 / boundary = 遠方の誤検出 / toolpaste = 入力分布
    arm("k_dicedet_rules", **{"loss.name": "dicedet",
                              "loss.params": {"extra_weight": 1.0, "threshold": 0.5},
                              "rules": {"enabled": True, "rules_dir": "workspace/anatomy_graph/out",
                                        "adj_weight": 1.0, "excl_weight": 0.1, "start_epoch": 8,
                                        "ramp_epochs": 5, "coarse_scale": 1.0, "presence_pool": 16}}),
    arm("k_dicedet_toolpaste", **{"loss.name": "dicedet",
                                  "loss.params": {"extra_weight": 1.0, "threshold": 0.5},
                                  "toolpaste": {"enabled": True, "p": 0.5, "max_tools": 2,
                                                "protect_w3": 0.15}}),
    arm("k_rules_boundary", **{"loss.name": "diceboundary",
                               "loss.params": {"extra_weight": 1.0},
                               "data.dist_maps": {"enabled": True, "downsample": 4},
                               "rules": {"enabled": True, "rules_dir": "workspace/anatomy_graph/out",
                                         "adj_weight": 1.0, "excl_weight": 0.1, "start_epoch": 8,
                                         "ramp_epochs": 5, "coarse_scale": 1.0, "presence_pool": 16}}),
    arm("k_dicedet_boundary", **{"loss.name": "dicedet",
                                 "loss.params": {"extra_weight": 1.0, "threshold": 0.5},
                                 "data.dist_maps": {"enabled": True, "downsample": 4}}),
    arm("k_rules_toolpaste", **{"rules": {"enabled": True, "rules_dir": "workspace/anatomy_graph/out",
                                          "adj_weight": 1.0, "excl_weight": 0.1, "start_epoch": 8,
                                          "ramp_epochs": 5, "coarse_scale": 1.0, "presence_pool": 16},
                                "toolpaste": {"enabled": True, "p": 0.5, "max_tools": 2,
                                              "protect_w3": 0.15}}),
    # 単体最良の encoder (convnext_xlarge.384: 0.6868 / HD 0.239) × 2 fold で勝った loss
    arm("c_xl_dicedet", **{"model.encoder_name": "tu-convnext_xlarge.fb_in22k_ft_in1k_384",
                           "loss.name": "dicedet",
                           "loss.params": {"extra_weight": 1.0, "threshold": 0.5}}),
    arm("c_xl_rules", **{"model.encoder_name": "tu-convnext_xlarge.fb_in22k_ft_in1k_384",
                         "rules": {"enabled": True, "rules_dir": "workspace/anatomy_graph/out",
                                   "adj_weight": 1.0, "excl_weight": 0.1, "start_epoch": 8,
                                   "ramp_epochs": 5, "coarse_scale": 1.0, "presence_pool": 16}}),
    arm("c_xl_nohflip", **{"model.encoder_name": "tu-convnext_xlarge.fb_in22k_ft_in1k_384",
                           "data.hflip": False}),
    # 2 fold とも勝った 2 つ (loss 軸の dicedet + aug 軸の nohflip) は直交するので重ねる
    arm("c_dicedet_nohflip", **{"loss.name": "dicedet",
                                "loss.params": {"extra_weight": 1.0, "threshold": 0.5},
                                "data.hflip": False}),
    arm("c_xl_dicedet_nohflip", **{"model.encoder_name": "tu-convnext_xlarge.fb_in22k_ft_in1k_384",
                                   "loss.name": "dicedet",
                                   "loss.params": {"extra_weight": 1.0, "threshold": 0.5},
                                   "data.hflip": False}),
    # アンサンブル用の種違い (同一レシピの seed 違いは安価で確実な多様性)
    arm("s_xl_seed43", **{"model.encoder_name": "tu-convnext_xlarge.fb_in22k_ft_in1k_384",
                          "experiment.seed": 43}),
    arm("s_dicedet_seed43", **{"loss.name": "dicedet",
                               "loss.params": {"extra_weight": 1.0, "threshold": 0.5},
                               "experiment.seed": 43}),
    # 勝因は swin encoder 本体 (UPerNet+ADE20k decoder でも convnext_l は 0.6320 止まり)。
    # ならば「勝っている encoder × 勝っている decoder」を組む
    arm("c_swin_unetpp", **{"model.encoder_name": "tu-swin_large_patch4_window12_384.ms_in22k_ft_in1k",
                            "train.lr": 1.0e-4}),
    hub_arm("c_swin_boundary", "smp-hub/upernet-swin-large", 1.0e-4,
            **{"loss.name": "diceboundary", "loss.params": {"extra_weight": 1.0},
               "data.dist_maps": {"enabled": True, "downsample": 4}}),
    hub_arm("c_swin_dicedet", "smp-hub/upernet-swin-large", 1.0e-4,
            **{"loss.name": "dicedet", "loss.params": {"extra_weight": 1.0, "threshold": 0.5}}),
    hub_arm("c_swin_ep30", "smp-hub/upernet-swin-large", 1.0e-4, **{"train.epochs": 30}),
]

# ------------------------------------------------- 軸 7: タスク特化 fine-tune
# 現状は 1 つの encoder が coarse と fine の両方に奉仕している。
# 公式 Task1(coarse) と Task2(fine) では最適な構成が違うと分かっている
#   (coarse は 5 モデルで 25 モデルの ens5 に勝つ / fine はアンサンブルと α が効く)
# ので、学習済みモデルから **相手側の loss を切って**数 epoch 追学習し、
# タスク特化させると上がるかを測る。
# 注意: config の task1_weight は **fine** に、task2_weight は **coarse** に掛かる
#       (expA00 以来の歴史的な逆転命名)。ここでは公式番号で命名する。
def ft_fine(name: str, src: str, **patch) -> dict:
    """<src> の重みから coarse loss を切って fine 特化させる arm を作る。"""
    base = {"model.init_from": src,
            "loss.task1_weight": 1.0, "loss.task2_weight": 0.0,
            "loss.fine2coarse_weight": 0.25,
            "train.epochs": 8, "train.lr": 5.0e-5, "train.warmup_epochs": 1}
    base.update(patch)
    return arm(name, **base)


def alldata(name: str, seed: int) -> dict:
    """全データ学習 (held-out 無し)。**検証不可**なので提出用途に限る。
    val は学習データの一部なので training_log の val 値は参考にならない。"""
    return arm(name, **{"cv.use_all_data": True, "experiment.seed": seed,
                        "loss.name": "dicedet",
                        "loss.params": {"extra_weight": 1.0, "threshold": 0.5},
                        "rules": {"enabled": True, "rules_dir": "workspace/anatomy_graph/out",
                                  "adj_weight": 1.0, "excl_weight": 0.1, "start_epoch": 8,
                                  "ramp_epochs": 5, "coarse_scale": 1.0, "presence_pool": 16}})


ALLDATA_ARMS = [alldata(f"a_full_s{i}", 42 + i) for i in range(6)]

DATASIZE_ARMS = [
    # 「学習データ量 -> スコア」の傾きを測る。80%(通常) に対し 60% で学習した場合を比較し、
    # 80% -> 100% (全データ学習) の伸びを外挿する。
    # 目的: 提出を「5fold アンサンブル」にするか「全データ学習」にするかの定量判断
    arm("z_data60", **{"cv.train_fold_limit": 3,
                       "loss.name": "dicedet",
                       "loss.params": {"extra_weight": 1.0, "threshold": 0.5},
                       "rules": {"enabled": True, "rules_dir": "workspace/anatomy_graph/out",
                                 "adj_weight": 1.0, "excl_weight": 0.1, "start_epoch": 8,
                                 "ramp_epochs": 5, "coarse_scale": 1.0, "presence_pool": 16}}),
    arm("z_data40", **{"cv.train_fold_limit": 2,
                       "loss.name": "dicedet",
                       "loss.params": {"extra_weight": 1.0, "threshold": 0.5},
                       "rules": {"enabled": True, "rules_dir": "workspace/anatomy_graph/out",
                                 "adj_weight": 1.0, "excl_weight": 0.1, "start_epoch": 8,
                                 "ramp_epochs": 5, "coarse_scale": 1.0, "presence_pool": 16}}),
]

COMBO3_ARMS = [
    # dicedet x rules が超加算だったので 3 要素目 (boundary = 遠方の誤検出抑制) も重ねる
    arm("k3_dicedet_rules_boundary",
        **{"loss.name": "diceboundary", "loss.params": {"extra_weight": 1.0},
           "data.dist_maps": {"enabled": True, "downsample": 4},
           "rules": {"enabled": True, "rules_dir": "workspace/anatomy_graph/out",
                     "adj_weight": 1.0, "excl_weight": 0.1, "start_epoch": 8,
                     "ramp_epochs": 5, "coarse_scale": 1.0, "presence_pool": 16}}),
    # 最強レシピの seed 違い（アンサンブル用の安価な多様性）
    arm("s_kdr_seed43",
        **{"loss.name": "dicedet", "loss.params": {"extra_weight": 1.0, "threshold": 0.5},
           "experiment.seed": 43,
           "rules": {"enabled": True, "rules_dir": "workspace/anatomy_graph/out",
                     "adj_weight": 1.0, "excl_weight": 0.1, "start_epoch": 8,
                     "ramp_epochs": 5, "coarse_scale": 1.0, "presence_pool": 16}}),
]

FT_ARMS = [
    # fine 特化が **他のレシピにも転移するか**（転移すればアンサンブル全体に効く）
    ft_fine("ft_dicedet_fine", "expA23_l_dicedet",
            **{"loss.name": "dicedet", "loss.params": {"extra_weight": 1.0, "threshold": 0.5}}),
    ft_fine("ft_deeplab_fine", "expA23_d_deeplabv3p", **{"model.arch": "deeplabv3plus"}),
    # 公式 Task1 = coarse 特化: fine 側の loss と f2c を切る
    arm("ft_t1_coarse", **{"model.init_from": "expA23_k_dicedet_rules",
                           "loss.task1_weight": 0.0, "loss.task2_weight": 1.0,
                           "loss.fine2coarse_weight": 0.0,
                           "loss.name": "dicedet",
                           "loss.params": {"extra_weight": 1.0, "threshold": 0.5},
                           "rules": {"enabled": True, "rules_dir": "workspace/anatomy_graph/out",
                                     "adj_weight": 1.0, "excl_weight": 0.1, "start_epoch": 0,
                                     "ramp_epochs": 1, "coarse_scale": 1.0, "presence_pool": 16},
                           "train.epochs": 8, "train.lr": 5.0e-5, "train.warmup_epochs": 1}),
    # 公式 Task2 = fine 特化: coarse 側の loss を切る (f2c は fine の正則化として残す)
    arm("ft_t2_fine", **{"model.init_from": "expA23_k_dicedet_rules",
                         "loss.task1_weight": 1.0, "loss.task2_weight": 0.0,
                         "loss.fine2coarse_weight": 0.25,
                         "loss.name": "dicedet",
                         "loss.params": {"extra_weight": 1.0, "threshold": 0.5},
                         "rules": {"enabled": True, "rules_dir": "workspace/anatomy_graph/out",
                                   "adj_weight": 1.0, "excl_weight": 0.1, "start_epoch": 0,
                                   "ramp_epochs": 1, "coarse_scale": 1.0, "presence_pool": 16},
                         "train.epochs": 8, "train.lr": 5.0e-5, "train.warmup_epochs": 1}),
]

# ---------------------------------------------------------------- 軸 6: 擬似ラベル / aug 強度
# expS03 では A06 ベース(v1/60ep, fold0 0.6575)で 擬似単独 0.6515 / 強aug単独 0.6646 /
# 擬似+強aug+ft 0.6640 だった。土台が変わった今の base (0.6821) で測り直す
PSEUDO_ARMS = [
    arm("a_strongaug", **{"data.aug": "strong"}),
    # 別スレッドが expS05 で検証中の **同ドメイン器具貼り付け** を新ベースでも測る
    # (expA02 の異ドメイン cutout は w=3 の細長い構造を壊して失敗した。こちらは
    #  Tiger 自身の器具を切り出し、w=3 クラスを 15% 以上隠す配置を棄却する)
    arm("a_toolpaste", **{"toolpaste": {"enabled": True, "p": 0.5, "max_tools": 2,
                                        "protect_w3": 0.15}}),
    arm("p_mix", **{"data.aug": "strong",
                    "pseudo": {"enabled": True, "index_csv": "workspace/data_proc/pseudo_index.csv",
                               "fraction": 0.5, "loss_weight": 0.5, "min_coverage": 0.70,
                               "real_only_last_epochs": 3}}),
    arm("p_heavy", **{"data.aug": "strong",
                      "pseudo": {"enabled": True, "index_csv": "workspace/data_proc/pseudo_index.csv",
                                 "fraction": 0.7, "loss_weight": 0.5, "min_coverage": 0.70,
                                 "real_only_last_epochs": 6}}),
]

# ---------------------------------------------------------------- 軸 5: aug / seed
MISC_ARMS = [
    # L/R を持つクラスがあるのに水平反転して学習している (expA00 以来)。切ると上がるか
    arm("a_nohflip", **{"data.hflip": False}),
    # fold0 のノイズ幅を測る。これ無しに ±0.005 の差を読まない
    arm("z_seed43", **{"experiment.seed": 43}),
]

ALL = (DECODER_ARMS + ENCODER_ARMS + HUB_ARMS + LOSS_ARMS + SCHED_ARMS
       + COMBO_ARMS + COMBO3_ARMS + DATASIZE_ARMS + ALLDATA_ARMS + FT_ARMS + PSEUDO_ARMS + MISC_ARMS)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--list", action="store_true")
    args = ap.parse_args()
    CFG_DIR.mkdir(exist_ok=True)
    for cfg in ALL:
        name = cfg["experiment"]["name"]
        if not args.list:
            (CFG_DIR / f"{name}.yaml").write_text(
                yaml.safe_dump(cfg, sort_keys=False, allow_unicode=True))
        m = cfg["model"]
        print(f"{name:28s} arch={str(m.get('arch') or m.get('hub_id')):30s} "
              f"enc={str(m.get('encoder_name', '-')):55s} loss={cfg['loss']['name']}")
    print(f"\n{len(ALL)} arms -> {CFG_DIR}")


if __name__ == "__main__":
    main()
