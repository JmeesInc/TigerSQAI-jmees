"""純粋に解剖学 (3D の位置関係) から書き起こした隣接グラフ — データ統計を一切使わない.

expS01 の 3D アトラスは 14/30 クラスしか持たず、下肺靱帯・心膜・反回神経・胸管・気管支動脈・
胃管など weight=3 の中核が欠けている。ここでは食道切除の術野 (右胸腔アプローチ, 後縦隔) で
問題になる 23 の解剖クラスについて、教科書的な 3D 位置関係を手で符号化する:

  contact[a][b] ∈ {1, 0.5, 0}
     1   = 剥離前から直接接している / 同じ筋膜面で apposed (例: 気管–食道, 反回神経–気管食道溝)
     0.5 = 近接 (≤ ~2 cm, 剥離すると隣り合って見える。例: 上大静脈–気管, 胸管–食道)
     0   = 離れている / 間に別の構造が挟まる

  level[c] = (上端, 下端) [cm]: 胸郭入口 (T1 上縁) を 0 とした頭尾方向の存在範囲の目安。
     頭尾方向の隙間が ≥ EXCL_GAP_CM のペアは「同じ術野に同時に写らない」= 排他 (E=1)。

出力 (out_anat3d/):
  rules_fine_fold{0..4}.npz / rules_fine.npz   : W_adj (= 1 − contact), E_excl, K_max, names, ignore_classes
  rules_coarse_fold{0..4}.npz / rules_coarse.npz : fine→merged に集約 (どれか 1 ペアでも接すれば接する)
  knowledge_graph.json                          : 根拠付きの一覧 (可視化・write-up 用)
  validation.txt                                : GT 統計 (out/fine) と 3D アトラス距離との一致度

学習で使うときは config の rules.rules_dir を workspace/anatomy_graph/out_anat3d にするだけ
(train.py は rules_dir/rules_{task}_fold{k}.npz を読む。fold 依存が無いので全 fold 同一)。
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
OUT = HERE / "out_anat3d"

EXCL_GAP_CM = 5.0
NEAR_W = 0.5          # 近接ペアの隣接ペナルティ (0 = 罰しない, 1 = 完全禁止)

# ルール対象外 (覆い被さる / 偏在する / 非解剖) — GT 版 rules.py と同じ集合
IGNORE = ["Background", "Instrument", "Other", "Fatty tissue esophagus", "Pleura",
          "Fatty tissue", "Pool of blood", "Resection area"]

# 頭尾方向の存在範囲 [cm from thoracic inlet]。右胸腔からの食道切除の術野を想定。
LEVEL = {
    "Trachea": (0, 7),                       # 胸郭入口〜気管分岐部 (T4/5)
    "Right main bronchus": (6, 9),
    "Left main bronchus": (6, 10),
    "Esophagus": (0, 25),                    # 胸部食道全長
    "Right inferior pulmonary ligament": (12, 20),
    "Left inferior pulmonary ligament": (12, 20),
    "Pericardium": (8, 18),                  # 心膜は分岐部より下
    "Inferior pulmonary vein": (10, 14),
    "Right subclavian artery": (-1, 2),      # 腕頭動脈からの分岐、胸郭入口
    "Right vagal nerve": (0, 20),            # 気管に沿って下り食道神経叢へ
    "Aorta": (3, 25),                        # 弓部〜下行
    "Azygos vein": (4, 20),                  # 奇静脈弓 (T4/5) と上行部
    "Superior caval vein": (0, 7),
    "Lung": (0, 25),
    "Lymph node": (0, 20),
    "Left subclavian artery": (0, 3),        # 弓部から
    "Right bronchial artery": (5, 9),
    "Pulmonary artery": (6, 10),
    "Gastric conduit": (5, 25),              # 挙上後、食道床に沿う
    "Right recurrent laryngeal nerve": (0, 3),   # 右鎖骨下動脈下を反回
    "Left recurrent laryngeal nerve": (2, 7),    # 大動脈弓下を反回、気管食道溝を上行
    "Omentum": (12, 25),                     # 胃管に付随して挙上
    "Thoracic duct": (2, 25),                # 大動脈と奇静脈の間、食道後方
}

# 直接接触 (1) / 近接 (0.5)。書いていないペアは 0 (離れている)。対称化する。
CONTACT: dict[tuple[str, str], tuple[float, str]] = {}


def c(a, b, v, why):
    CONTACT[(a, b)] = (v, why)


# --- 気道
c("Trachea", "Right main bronchus", 1, "分岐部で連続")
c("Trachea", "Left main bronchus", 1, "分岐部で連続")
c("Trachea", "Esophagus", 1, "膜様部が食道前壁に接する (全長)")
c("Trachea", "Right vagal nerve", 1, "右迷走神経は気管右側面を下行")
c("Trachea", "Right recurrent laryngeal nerve", 1, "右気管食道溝を上行")
c("Trachea", "Left recurrent laryngeal nerve", 1, "左気管食道溝を上行")
c("Trachea", "Lymph node", 1, "傍気管リンパ節 (2R/2L/4R/4L)")
c("Trachea", "Aorta", 1, "大動脈弓が気管左前面に接する")
c("Trachea", "Right subclavian artery", 0.5, "腕頭動脈分岐部が気管右前に近接")
c("Trachea", "Left subclavian artery", 0.5, "弓部起始が気管左側に近接")
c("Trachea", "Superior caval vein", 0.5, "気管右前方、脂肪を挟んで近接")
c("Trachea", "Azygos vein", 0.5, "奇静脈弓は分岐部直下で右主気管支上を跨ぐ")
c("Trachea", "Lung", 0.5, "右上葉肺尖が胸膜越しに近接")
c("Trachea", "Right bronchial artery", 0.5, "気管支動脈は分岐部近傍を走る")
c("Trachea", "Pulmonary artery", 0.5, "肺動脈幹は分岐部直下")
c("Trachea", "Thoracic duct", 0.5, "上縦隔で食道左後方、気管とは食道を挟む")
c("Trachea", "Gastric conduit", 0.5, "挙上胃管が気管膜様部に接することがある")

c("Right main bronchus", "Esophagus", 1, "食道は右主気管支後面に接する")
c("Right main bronchus", "Azygos vein", 1, "奇静脈弓が右主気管支上を跨ぐ")
c("Right main bronchus", "Lymph node", 1, "気管支周囲 (10R) / 分岐下 (7)")
c("Right main bronchus", "Pulmonary artery", 1, "右肺動脈が右主気管支前面を走る")
c("Right main bronchus", "Right bronchial artery", 1, "気管支動脈が気管支に沿う")
c("Right main bronchus", "Lung", 1, "肺門で肺に入る")
c("Right main bronchus", "Right vagal nerve", 1, "迷走神経肺枝が気管支後面")
c("Right main bronchus", "Left main bronchus", 1, "分岐部で連続")
c("Right main bronchus", "Superior caval vein", 0.5, "上大静脈は右主気管支前上方")
c("Right main bronchus", "Pericardium", 0.5, "心膜上縁が肺門部に近接")
c("Right main bronchus", "Inferior pulmonary vein", 0.5, "下肺静脈は肺門下部、気管支の下方")
c("Right main bronchus", "Thoracic duct", 0.5, "胸管は分岐部高位で食道後方")
c("Right main bronchus", "Gastric conduit", 0.5, "胃管は食道床で気管支後面に接しうる")

c("Left main bronchus", "Esophagus", 1, "食道は左主気管支後面を横切る")
c("Left main bronchus", "Aorta", 1, "大動脈弓が左主気管支上を跨ぐ")
c("Left main bronchus", "Lymph node", 1, "分岐下 (7) / 気管支周囲 (10L)")
c("Left main bronchus", "Pulmonary artery", 1, "左肺動脈が左主気管支上を跨ぐ")
c("Left main bronchus", "Left recurrent laryngeal nerve", 1, "大動脈弓下で反回した直後に近接・接触")
c("Left main bronchus", "Lung", 1, "肺門で肺に入る")
c("Left main bronchus", "Right bronchial artery", 0.5, "気管支動脈 (右) は分岐部下で左側にも枝")
c("Left main bronchus", "Pericardium", 0.5, "心膜上縁が肺門部に近接")
c("Left main bronchus", "Thoracic duct", 0.5, "胸管は分岐部高位で食道後方、左主気管支の後ろ")
c("Left main bronchus", "Right vagal nerve", 0.5, "食道神経叢が分岐部下で交差")
c("Left main bronchus", "Gastric conduit", 0.5, "胃管は食道床で気管支後面に接しうる")
c("Left main bronchus", "Inferior pulmonary vein", 0.5, "下肺静脈は肺門下部")

# --- 食道と周囲
c("Esophagus", "Aorta", 1, "下行大動脈が食道左後方に接する")
c("Esophagus", "Azygos vein", 1, "奇静脈が食道右後方を上行")
c("Esophagus", "Thoracic duct", 1, "胸管は食道後方 (大動脈と奇静脈の間) を上行")
c("Esophagus", "Right vagal nerve", 1, "食道神経叢を形成")
c("Esophagus", "Right recurrent laryngeal nerve", 1, "気管食道溝")
c("Esophagus", "Left recurrent laryngeal nerve", 1, "気管食道溝")
c("Esophagus", "Lymph node", 1, "傍食道リンパ節 (8)")
c("Esophagus", "Pericardium", 1, "中下部食道は心膜後面に接する")
c("Esophagus", "Lung", 1, "縦隔胸膜越しに両肺に接する")
c("Esophagus", "Right inferior pulmonary ligament", 1, "下肺靱帯は食道側縦隔に付着")
c("Esophagus", "Left inferior pulmonary ligament", 1, "同左")
c("Esophagus", "Inferior pulmonary vein", 1, "下肺静脈は食道外側に隣接 (下肺靱帯上端)")
c("Esophagus", "Gastric conduit", 1, "胃管は食道と連続・同じ床")
c("Esophagus", "Right subclavian artery", 0.5, "胸郭入口で気管の右、食道から数 mm–1 cm")
c("Esophagus", "Left subclavian artery", 0.5, "上縦隔で食道左に近接")
c("Esophagus", "Superior caval vein", 0.5, "右前方、気管を挟む")
c("Esophagus", "Right bronchial artery", 1, "気管支動脈が食道前面を横切って気管支へ")
c("Esophagus", "Pulmonary artery", 0.5, "左肺動脈が食道前方に近接")
c("Esophagus", "Omentum", 0.5, "胃管に付随した大網が食道床に入る")

# --- 下縦隔 (心膜・下肺静脈・下肺靱帯)
c("Pericardium", "Inferior pulmonary vein", 1, "下肺静脈は心膜に入る")
c("Pericardium", "Lung", 1, "肺は心膜に接する")
c("Pericardium", "Right inferior pulmonary ligament", 1, "下肺靱帯は心膜外側に付着")
c("Pericardium", "Left inferior pulmonary ligament", 1, "同左")
c("Pericardium", "Lymph node", 1, "傍食道・下肺靱帯リンパ節")
c("Pericardium", "Right vagal nerve", 1, "迷走神経は心膜後面 (食道前面) を下る")
c("Pericardium", "Pulmonary artery", 1, "肺動脈は心膜内から出る")
c("Pericardium", "Aorta", 1, "上行大動脈は心膜内; 下行は食道を挟んで後方 (近接)")
c("Pericardium", "Superior caval vein", 1, "上大静脈は心膜に入る")
c("Pericardium", "Gastric conduit", 1, "挙上胃管は心膜後面に接する")
c("Pericardium", "Azygos vein", 0.5, "奇静脈は食道後方、心膜からは食道を挟む")
c("Pericardium", "Thoracic duct", 0.5, "胸管は食道後方")
c("Pericardium", "Omentum", 0.5, "大網が胃管とともに心膜後面")
c("Pericardium", "Right bronchial artery", 0.5, "分岐部下で近接")

c("Inferior pulmonary vein", "Lung", 1, "肺門から出る")
c("Inferior pulmonary vein", "Right inferior pulmonary ligament", 1, "下肺靱帯上端が下肺静脈")
c("Inferior pulmonary vein", "Left inferior pulmonary ligament", 1, "同左")
c("Inferior pulmonary vein", "Lymph node", 1, "下肺靱帯リンパ節 (9)")
c("Inferior pulmonary vein", "Pulmonary artery", 0.5, "肺門で近接")
c("Inferior pulmonary vein", "Right vagal nerve", 0.5, "食道神経叢が近接")
c("Inferior pulmonary vein", "Gastric conduit", 0.5, "胃管は食道床で近接")
c("Inferior pulmonary vein", "Aorta", 0.5, "左下肺静脈は下行大動脈に近接")
c("Inferior pulmonary vein", "Azygos vein", 0.5, "右側で食道を挟んで近接")

c("Right inferior pulmonary ligament", "Lung", 1, "下葉を縦隔に繋ぐ")
c("Right inferior pulmonary ligament", "Lymph node", 1, "下肺靱帯リンパ節 (9R)")
c("Right inferior pulmonary ligament", "Azygos vein", 0.5, "食道後方で近接")
c("Right inferior pulmonary ligament", "Right vagal nerve", 0.5, "食道神経叢")
c("Right inferior pulmonary ligament", "Gastric conduit", 0.5, "食道床")
c("Right inferior pulmonary ligament", "Thoracic duct", 0.5, "下部食道後方")
c("Left inferior pulmonary ligament", "Lung", 1, "下葉を縦隔に繋ぐ")
c("Left inferior pulmonary ligament", "Lymph node", 1, "下肺靱帯リンパ節 (9L)")
c("Left inferior pulmonary ligament", "Aorta", 1, "左下肺靱帯は下行大動脈に接する")
c("Left inferior pulmonary ligament", "Right vagal nerve", 0.5, "食道神経叢")
c("Left inferior pulmonary ligament", "Gastric conduit", 0.5, "食道床")
c("Left inferior pulmonary ligament", "Thoracic duct", 0.5, "下部食道後方")

# --- 大血管
c("Aorta", "Azygos vein", 0.5, "下行大動脈と奇静脈は椎体前で胸管を挟んで隣接")
c("Aorta", "Thoracic duct", 1, "胸管は大動脈右側 (大動脈と奇静脈の間) を上行")
c("Aorta", "Left subclavian artery", 1, "弓部から分岐")
c("Aorta", "Left recurrent laryngeal nerve", 1, "弓部下で反回")
c("Aorta", "Lymph node", 1, "大動脈弓下 (5) / 傍大動脈")
c("Aorta", "Lung", 1, "左肺が弓部・下行大動脈に接する")
c("Aorta", "Pulmonary artery", 1, "左肺動脈は弓部直下 (動脈管索)")
c("Aorta", "Right bronchial artery", 1, "気管支動脈は下行大動脈から起始")
c("Aorta", "Right subclavian artery", 0.5, "腕頭動脈経由で弓部に近接")
c("Aorta", "Right vagal nerve", 0.5, "右迷走神経は気管右側、弓部とは気管を挟む (左迷走神経は弓部を跨ぐ)")
c("Aorta", "Gastric conduit", 0.5, "胃管は食道床で下行大動脈に近接")
c("Aorta", "Superior caval vein", 0.5, "上行大動脈右側に上大静脈")
c("Aorta", "Right inferior pulmonary ligament", 0.5, "食道を挟んで対側")

c("Azygos vein", "Superior caval vein", 1, "奇静脈弓は上大静脈に流入")
c("Azygos vein", "Thoracic duct", 1, "胸管は奇静脈左側を上行")
c("Azygos vein", "Lymph node", 1, "奇静脈弓周囲 (4R/10R)")
c("Azygos vein", "Lung", 1, "右肺が奇静脈弓に接する")
c("Azygos vein", "Right vagal nerve", 1, "右迷走神経は奇静脈弓の内側を下行")
c("Azygos vein", "Right bronchial artery", 1, "奇静脈弓下で気管支動脈が走る")
c("Azygos vein", "Pulmonary artery", 0.5, "奇静脈弓下で右肺動脈に近接")
c("Azygos vein", "Gastric conduit", 0.5, "食道床")
c("Azygos vein", "Right recurrent laryngeal nerve", 0.5, "上縦隔で近接 (奇静脈は T4 以下)")
c("Azygos vein", "Right subclavian artery", 0.5, "上縦隔、間に脂肪")

c("Superior caval vein", "Right subclavian artery", 0.5, "腕頭静脈合流部で近接")
c("Superior caval vein", "Right vagal nerve", 1, "右迷走神経は上大静脈と気管の間を下行")
c("Superior caval vein", "Lymph node", 1, "傍気管 (4R) は上大静脈後方")
c("Superior caval vein", "Lung", 1, "右上葉が接する")
c("Superior caval vein", "Pulmonary artery", 1, "右肺動脈が上大静脈後方を走る")
c("Superior caval vein", "Right recurrent laryngeal nerve", 0.5, "上縦隔で近接")
c("Superior caval vein", "Pericardium", 1, "上大静脈は心膜に入る")

c("Right subclavian artery", "Right recurrent laryngeal nerve", 1, "右反回神経は右鎖骨下動脈下を反回")
c("Right subclavian artery", "Right vagal nerve", 1, "右迷走神経が右鎖骨下動脈前面を横切る")
c("Right subclavian artery", "Lymph node", 1, "右上縦隔・鎖骨下リンパ節 (2R/1R)")
c("Right subclavian artery", "Lung", 0.5, "肺尖が胸膜越し")
c("Right subclavian artery", "Left subclavian artery", 0.5, "両側、気管を挟む")

c("Left subclavian artery", "Left recurrent laryngeal nerve", 1, "弓部で近接 (左反回神経は動脈管索の外側で反回)")
c("Left subclavian artery", "Lymph node", 1, "左上縦隔 (2L/4L)")
c("Left subclavian artery", "Thoracic duct", 1, "胸管は左鎖骨下動脈近傍で静脈角へ")
c("Left subclavian artery", "Lung", 0.5, "左肺尖")

c("Right vagal nerve", "Lymph node", 1, "傍気管・傍食道リンパ節に接する")
c("Right vagal nerve", "Right recurrent laryngeal nerve", 1, "右迷走神経から分岐")
c("Right vagal nerve", "Lung", 1, "肺門後方を通る")
c("Right vagal nerve", "Right bronchial artery", 0.5, "肺門後方で近接")
c("Right vagal nerve", "Pulmonary artery", 0.5, "肺門で近接")
c("Right vagal nerve", "Gastric conduit", 0.5, "食道床")
c("Right vagal nerve", "Thoracic duct", 0.5, "食道を挟む")
c("Right vagal nerve", "Left recurrent laryngeal nerve", 0.5, "対側だが上縦隔で近接")

c("Right recurrent laryngeal nerve", "Lymph node", 1, "反回神経リンパ節 (106rec R)")
c("Right recurrent laryngeal nerve", "Left recurrent laryngeal nerve", 0.5, "気管を挟んで両側の溝")
c("Right recurrent laryngeal nerve", "Thoracic duct", 0.5, "上縦隔")
c("Left recurrent laryngeal nerve", "Lymph node", 1, "反回神経リンパ節 (106rec L)")
c("Left recurrent laryngeal nerve", "Thoracic duct", 1, "胸管は上縦隔で食道左後方、左反回神経に近接")
c("Left recurrent laryngeal nerve", "Pulmonary artery", 0.5, "動脈管索で肺動脈に近接")
c("Left recurrent laryngeal nerve", "Lung", 0.5, "左肺")
c("Left recurrent laryngeal nerve", "Gastric conduit", 0.5, "胃管が気管食道溝近くまで上がる")

c("Lymph node", "Lung", 1, "肺門リンパ節")
c("Lymph node", "Right bronchial artery", 1, "気管支動脈周囲")
c("Lymph node", "Pulmonary artery", 1, "肺門・気管支周囲")
c("Lymph node", "Thoracic duct", 1, "傍食道・後縦隔")
c("Lymph node", "Gastric conduit", 0.5, "胃管に付随する所属リンパ節")
c("Lymph node", "Omentum", 0.5, "大網に含まれる")
c("Lymph node", "Right inferior pulmonary ligament", 1, "9R")
c("Lymph node", "Left inferior pulmonary ligament", 1, "9L")

c("Lung", "Pulmonary artery", 1, "肺門")
c("Lung", "Right bronchial artery", 0.5, "肺門後方")
c("Lung", "Gastric conduit", 1, "挙上胃管は右肺に接する")
c("Lung", "Omentum", 0.5, "胃管とともに")
c("Lung", "Thoracic duct", 0.5, "後縦隔、胸膜越し")

c("Right bronchial artery", "Pulmonary artery", 1, "気管支動脈は肺動脈に沿う")
c("Right bronchial artery", "Thoracic duct", 0.5, "後縦隔")
c("Pulmonary artery", "Gastric conduit", 0.5, "分岐部下で近接")
c("Gastric conduit", "Omentum", 1, "大網は胃管に付着")
c("Gastric conduit", "Thoracic duct", 0.5, "食道床")
# 大網は胃管再建時に胃管とともに下縦隔へ挙上され、食道床の構造に接する
c("Omentum", "Thoracic duct", 0.5, "挙上大網が食道床で胸管に近接")
c("Omentum", "Inferior pulmonary vein", 0.5, "挙上大網が下肺静脈に近接")
c("Omentum", "Aorta", 0.5, "挙上大網が下行大動脈に接する")
c("Omentum", "Right inferior pulmonary ligament", 0.5, "挙上大網が下肺靱帯に近接")
c("Omentum", "Left inferior pulmonary ligament", 0.5, "同左")
c("Omentum", "Azygos vein", 0.5, "食道床で近接")
c("Omentum", "Right vagal nerve", 0.5, "食道床で近接")


def build() -> None:
    OUT.mkdir(exist_ok=True)
    lm = pd.read_csv(REPO / "data/labelmap.csv").drop_duplicates("fine_id").sort_values("fine_id")
    names = lm.fine_name.tolist()
    C = len(names)
    idx = {n: i for i, n in enumerate(names)}
    ign = [idx[n] for n in IGNORE]
    active = [n for n in names if n not in IGNORE]
    for n in active:
        assert n in LEVEL, f"level 未定義: {n}"
    for (a, b) in CONTACT:
        assert a in idx and b in idx, (a, b)
        assert a not in IGNORE and b not in IGNORE, (a, b)
    contact = np.zeros((C, C), np.float32)
    why = {}
    for (a, b), (v, w) in CONTACT.items():
        i, j = idx[a], idx[b]
        assert contact[i, j] == 0 or contact[i, j] == v, f"重複定義 {a}-{b}"
        contact[i, j] = contact[j, i] = v
        why[(min(i, j), max(i, j))] = w
    # 隣接ペナルティ: 離れている = 1, 近接 = NEAR_W, 接触 = 0。除外クラスは 0
    W = np.zeros((C, C), np.float32)
    E = np.zeros((C, C), np.float32)
    act = [idx[n] for n in active]
    for i in act:
        for j in act:
            if i == j:
                continue
            W[i, j] = 1.0 if contact[i, j] == 0 else (NEAR_W if contact[i, j] == 0.5 else 0.0)
            lo = max(LEVEL[names[i]][0], LEVEL[names[j]][0])
            hi = min(LEVEL[names[i]][1], LEVEL[names[j]][1])
            if lo - hi >= EXCL_GAP_CM:
                E[i, j] = 1.0
                W[i, j] = 1.0
    # 個数上限は GT 版から流用 (幾何からは決められない)
    gt = np.load(HERE / "out/rules_fine.npz", allow_pickle=True)
    K = gt["K_max"]
    for k in list(range(5)) + [None]:
        fn = f"rules_fine_fold{k}.npz" if k is not None else "rules_fine.npz"
        np.savez(OUT / fn, W_adj=W, E_excl=E, K_max=K, names=np.array(names),
                 ignore_classes=np.array(ign))
    # coarse: fine→merged へ集約
    cm = pd.read_csv(REPO / "data/labelmap.csv").drop_duplicates("merged_id").sort_values("merged_id")
    cnames = cm.merged_name.tolist()
    f2c = dict(zip(lm.fine_id, lm.merged_id))
    Cc = len(cnames)
    gtc = np.load(HERE / "out/rules_coarse.npz", allow_pickle=True)
    ignc = [int(i) for i in gtc["ignore_classes"]]
    best = np.full((Cc, Cc), 0.0, np.float32)      # 最大 contact
    seen = np.zeros((Cc, Cc), bool)
    for i in act:
        for j in act:
            if i == j:
                continue
            a, b = f2c[i], f2c[j]
            if a == b or a in ignc or b in ignc:
                continue
            best[a, b] = max(best[a, b], contact[i, j])
            seen[a, b] = True
    Wc = np.zeros((Cc, Cc), np.float32)
    Ec = np.zeros((Cc, Cc), np.float32)
    for a in range(Cc):
        for b in range(Cc):
            if not seen[a, b]:
                continue
            Wc[a, b] = 1.0 if best[a, b] == 0 else (NEAR_W if best[a, b] == 0.5 else 0.0)
            # 排他: 構成 fine クラスのどのペアも排他なら
            fa = [i for i in act if f2c[i] == a]
            fb = [j for j in act if f2c[j] == b]
            if fa and fb and all(E[i, j] == 1 for i in fa for j in fb):
                Ec[a, b] = 1.0
                Wc[a, b] = 1.0
    for k in list(range(5)) + [None]:
        fn = f"rules_coarse_fold{k}.npz" if k is not None else "rules_coarse.npz"
        np.savez(OUT / fn, W_adj=Wc, E_excl=Ec, K_max=gtc["K_max"], names=np.array(cnames),
                 ignore_classes=np.array(ignc))

    # --- 可視化 / 根拠
    kg = {"level_cm": LEVEL, "ignore": IGNORE, "near_w": NEAR_W, "excl_gap_cm": EXCL_GAP_CM,
          "pairs": [{"a": names[i], "b": names[j], "contact": float(contact[i, j]),
                     "W": float(W[i, j]), "E": int(E[i, j]), "why": why.get((i, j), "離れている")}
                    for i in act for j in act if i < j]}
    (OUT / "knowledge_graph.json").write_text(json.dumps(kg, ensure_ascii=False, indent=1))

    # --- 検証: GT 統計 / 3D アトラス距離との一致
    from scipy.stats import spearmanr
    adj = np.load(HERE / "out/fine/adj_img.npy"); cooc = np.load(HERE / "out/fine/cooc.npy")
    p = (adj + 0.5) / (cooc + 1.0)
    Wgt = gt["W_adj"]; Egt = gt["E_excl"]
    lines = []
    xs, ys, ws, wg = [], [], [], []
    for i in act:
        for j in act:
            if i < j and cooc[i, j] >= 10:
                xs.append(contact[i, j]); ys.append(p[i, j]); ws.append(W[i, j]); wg.append(Wgt[i, j])
    xs, ys, ws, wg = map(np.array, (xs, ys, ws, wg))
    lines.append(f"GT で共起 ≥10 枚のペア: {len(xs)}")
    lines.append(f"  knowledge contact(1/0.5/0) vs GT p_adj: Spearman {spearmanr(xs, ys)[0]:.3f}")
    lines.append(f"  knowledge W vs GT W_adj: Spearman {spearmanr(ws, wg)[0]:.3f}")
    for lab, m in (("contact=1", xs == 1), ("contact=0.5", xs == 0.5), ("contact=0", xs == 0)):
        if m.sum():
            lines.append(f"  {lab:12s}: n={int(m.sum()):3d}  GT p_adj 中央値 {np.median(ys[m]):.3f}  "
                         f"GT 禁止(p≤.02) {int((ys[m] <= .02).sum())}  GT 頻繁(p≥.2) {int((ys[m] >= .2).sum())}")
    # 3D アトラス距離
    md = np.load(HERE / "out/atlas3d/class_mindist_mm.npy")
    ds, cs = [], []
    for i in act:
        for j in act:
            if i < j and np.isfinite(md[i, j]):
                ds.append(md[i, j]); cs.append(contact[i, j])
    ds, cs = np.array(ds), np.array(cs)
    lines.append(f"3D アトラス収録ペア: {len(ds)}  contact vs 表面距離 Spearman {spearmanr(cs, ds)[0]:.3f}")
    for lab, m in (("contact=1", cs == 1), ("contact=0.5", cs == 0.5), ("contact=0", cs == 0)):
        if m.sum():
            lines.append(f"  {lab:12s}: n={int(m.sum()):3d}  3D 距離中央値 {np.median(ds[m]):.1f} mm  ≤5mm {int((ds[m] <= 5).sum())}  >20mm {int((ds[m] > 20).sum())}")
    # 排他の一致
    ex_k = {(i, j) for i in act for j in act if i < j and E[i, j] == 1}
    ex_g = {(i, j) for i in act for j in act if i < j and Egt[i, j] == 1}
    lines.append(f"排他ペア: knowledge {len(ex_k)} / GT {len(ex_g)} / 一致 {len(ex_k & ex_g)}")
    lines.append("  knowledge のみ: " + "; ".join(f"{names[i]}–{names[j]}" for i, j in sorted(ex_k - ex_g)))
    lines.append("  GT のみ: " + "; ".join(f"{names[i]}–{names[j]}" for i, j in sorted(ex_g - ex_k)))
    # 不一致の大きいペア
    bad = [(names[i], names[j], float(contact[i, j]), float(p[i, j]), int(cooc[i, j]))
           for i in act for j in act if i < j and cooc[i, j] >= 10 and
           ((contact[i, j] == 0 and p[i, j] >= .2) or (contact[i, j] == 1 and p[i, j] <= .02))]
    lines.append(f"不一致 (knowledge 離れている×GT 頻繁接触, または knowledge 接触×GT 禁止): {len(bad)}")
    for r in sorted(bad, key=lambda r: -r[4]):
        lines.append(f"  {r[0]} – {r[1]}: contact {r[2]}  GT p_adj {r[3]:.3f} (cooc {r[4]})")
    n_pen = int((W[np.ix_(act, act)] > 0).sum() // 2); n_full = int((W[np.ix_(act, act)] >= 1).sum() // 2)
    lines.insert(0, f"knowledge graph: {len(act)} クラス, 隣接ペナルティ {n_pen} ペア (W=1: {n_full}), 排他 {len(ex_k)} ペア; "
                    f"coarse: ペナルティ {int((Wc > 0).sum() // 2)} ペア, 排他 {int(Ec.sum() // 2)}")
    (OUT / "validation.txt").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    build()
