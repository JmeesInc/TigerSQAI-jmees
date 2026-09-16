"""TigerSQAI Task1+2+3 推論コンテナ (v007 = v006 の実行時間対策版).

v006 が主催者環境で **139 フレームに約 9 時間**かかり、公式の予算
（1 分 / フレーム / タスク → 3 タスクなら 139x3 = 417 分）を超過した。
v006 の構造的な問題は 3 つで、すべて公式 Docker Instructions の
「Practical consequences」に明記されている推奨の逆をやっていた:

  1. **メンバー外側 / フレーム内側**のループだったため、1 枚の PNG を
     32 回 (seg 17 + Task3 encoder 15) デコードしていた。
     → 公式: "loop over frames on the outside ... Reading and decoding each
       frame once instead of three times is free performance."
  2. **全フレームの確率を CPU に貯めていた**。576x1024 の fine 31ch + coarse 16ch を
     float32 で持つと 1 枚 110MB、139 枚で **15.3GB**。評価機の RAM 次第で
     スワップに入り、これだけで桁違いに遅くなる。
     → 公式: "Do not buffer all frames in memory. Read, predict, write, release."
  3. **全部終わってから書き出していた**ので、時間切れで kill されると
     出力がゼロになり、全タスク失格になる。
     → 公式: "Write results incrementally ... A crash on frame 130 should not
       throw away the first 129 results."

v007 の設計:

  * **チャンク単位 (既定 12 枚) のフレーム外側ループ**。チャンク内で PNG を 1 回だけ
    デコードし、メンバー解像度ごとの入力テンソルを使い回す。
  * 確率の累積は **GPU 上・チャンク分だけ**。常駐メモリはチャンク幅に比例するので
    フレーム数が増えても破綻しない。
  * チャンクを処理し終えるたびに **PNG と task3.csv を即書き出して flush**。
    途中で止められても、そこまでのフレームは有効な提出物として残る。
  * **実測ベースの時間予算制御**。1 チャンク処理するたびにメンバー 1 本あたりの
    実コストを測り直し、「残り時間 / 残りフレーム」に収まるメンバー数 K を決める。
    GPU が出ない環境に落ちても、K が自動的に 1 まで縮んで完走する。
    メンバーは `model/priority.json` の順（= 5-fold CV の強い順）に使うので、
    K が小さくなっても残るのは上位メンバーになる。

出力: /output/task1/<name>.png (merged 16 色), /output/task2/<name>.png (fine 31 色),
      /output/task3.csv

公式定義: **Task1 = merged/coarse, Task2 = fine**。
"""

from __future__ import annotations

import csv
import json
import os
import sys
import time

import cv2
import numpy as np
import pandas as pd
import torch

from model_def import build_model
from t3_features import mask_features, all_features, grid_features, SUB

INPUT_DIR = os.environ.get("INPUT_DIR", "/input")
OUTPUT_DIR = os.environ.get("OUTPUT_DIR", "/output")
BASE = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.environ.get("MODEL_DIR", os.path.join(BASE, "model"))
MODEL_T3_DIR = os.environ.get("MODEL_T3_DIR", os.path.join(BASE, "model_t3"))
MODEL_T3ENC_DIR = os.environ.get("MODEL_T3ENC_DIR", os.path.join(BASE, "model_t3enc"))
LABELMAP = os.path.join(BASE, "labelmap.csv")

MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)
STATIONS = ["6L", "6R", "7L", "7R", "8", "9", "10L", "10R", "11L", "11R", "12L", "12R", "13L", "13R"]
N_FINE, N_COARSE = 31, 16

# --- 実行時間の制御 -------------------------------------------------------
# 公式予算: 1 分 / フレーム / タスク。このイメージは 3 タスクを書くので 3 分 / フレーム。
SEC_PER_FRAME_PER_TASK = float(os.environ.get("SEC_PER_FRAME_PER_TASK", "60"))
N_TASKS = 3
SAFETY = float(os.environ.get("BUDGET_SAFETY", "0.60"))   # 主催者環境は自環境より遅い前提
BUDGET_MIN = float(os.environ.get("BUDGET_MIN", "0"))     # >0 なら総予算[分]を直接指定
CHUNK = int(os.environ.get("CHUNK", "12"))                # 同時に確率を持つフレーム数
BATCH = int(os.environ.get("BATCH", "4"))                 # forward のミニバッチ
MAX_MEMBERS = int(os.environ.get("MAX_MEMBERS", "0"))     # >0 で上限を固定（デバッグ用）
MAX_T3ENC = int(os.environ.get("MAX_T3ENC", "0"))
ISLAND_PPM = float(os.environ.get("ISLAND_PPM", "4000"))
_ppm_json = os.path.join(MODEL_DIR, "island_ppm.json")   # クラス別しきい値 {"fine": {id: ppm}, "coarse": {id: ppm}}
ISLAND_PPM_MAP = json.load(open(_ppm_json)) if os.path.exists(_ppm_json) else None  # 画像面積の 0.4% 未満の連結成分を削除

T0 = time.time()


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')} +{time.time() - T0:7.1f}s] {msg}", flush=True)


# ---------------------------------------------------------------- モデル一覧
def list_members(root: str = MODEL_DIR) -> list[dict]:
    """model/<member>/config.json を走査し、**読み込まずに仕様だけ**返す。

    priority.json があればその順に並べる（前から順に予算の許す分だけ使う）。
    """
    specs = []
    if not os.path.isdir(root):
        return []
    for name in sorted(os.listdir(root)):
        d = os.path.join(root, name)
        cfg_path = os.path.join(d, "config.json")
        if not os.path.isdir(d) or not os.path.exists(cfg_path):
            continue
        cfg = json.load(open(cfg_path))
        for fn in sorted(f for f in os.listdir(d) if f.startswith("fold") and f.endswith(".pt")):
            specs.append({"member": name, "fold": int(fn[4:-3]), "cfg": cfg,
                          "path": os.path.join(d, fn), "img": tuple(cfg["img_size"]),
                          "fine_only": bool(cfg.get("fine_only", False)),
                          "weight": float(cfg.get("weight", 1.0)),
                          # 同じレシピのシード違いは同じ group に入れ、**group 内で平均してから**
                          # group 間で平均する。シードが何本あってもレシピの重みは 1 のまま。
                          "group": str(cfg.get("group", name)),
                          # 水平反転 TTA（config.json の "tta": true）。反転なしで学習した
                          # レシピ (r_nohflip) には付けない。forward が 2 回になる
                          "tta": bool(cfg.get("tta", False))})
    pri_path = os.path.join(root, "priority.json")
    if os.path.exists(pri_path):
        order = {n: i for i, n in enumerate(json.load(open(pri_path)))}
        specs.sort(key=lambda s: (order.get(s["member"], 10_000), s["member"], s["fold"]))
        log(f"{root}: priority.json に従って並べ替え（先頭 = {specs[0]['member']}）")
    log(f"{root}: {len(specs)} models across {len({s['member'] for s in specs})} members "
        f"/ {len({s['group'] for s in specs})} groups (recipes)")
    return specs


# 75 メンバーのうち **構造は 7 通りしかない**（大半が deeplabv3plus + ConvNeXt-L @576x1024）。
# build_model は ConvNeXt-Large を 2 本組むので 5〜17 秒かかる。構造をキャッシュして
# state_dict だけ差し替えれば、2 本目以降のロードは重みの転送だけで済む。
_ARCH_CACHE: "dict[str, tuple]" = {}
ARCH_CACHE_MAX = int(os.environ.get("ARCH_CACHE_MAX", "3"))


# 重み本体も RAM に載せておく。チャンクをまたぐたびに 1 本 0.5-1GB を読み直すと、
# 評価機のストレージが遅い場合そこだけで予算を食い潰す（自ホストの HDD 実測 30MB/s では
# 1 本 27 秒かかった）。空きメモリの一定割合までを上限にキャッシュする。
_SD_CACHE: "dict[str, dict]" = {}
_SD_BYTES = [0]


def _mem_available() -> int:
    try:
        for line in open("/proc/meminfo"):
            if line.startswith("MemAvailable:"):
                return int(line.split()[1]) * 1024
    except Exception:
        pass
    return 0


# 上限は「空きメモリの 30%」と 12GB の小さい方。v006 が全フレームの確率で 15GB 使って
# 破綻した反省から、ここは必ず頭打ちにする。
_SD_CACHE_LIMIT = (int(float(os.environ.get("WEIGHT_CACHE_GB", "0")) * 1e9)
                   or min(int(_mem_available() * 0.30), 12_000_000_000))


def _cache_sd(path: str, sd: dict) -> None:
    n = sum(v.numel() * v.element_size() for v in sd.values())
    if _SD_BYTES[0] + n > _SD_CACHE_LIMIT:
        return
    _SD_CACHE[path] = sd
    _SD_BYTES[0] += n


def _arch_key(spec: dict) -> str:
    return json.dumps([spec["cfg"]["model"], list(spec["img"])], sort_keys=True, default=str)


def load_one(spec: dict, device: str):
    key = _arch_key(spec)
    m = _ARCH_CACHE.pop(key, None)
    if m is None:
        m = build_model(spec["cfg"]["model"], spec["img"])
        m = (m.half() if device == "cuda" else m.float()).to(device).eval()
        while len(_ARCH_CACHE) >= max(ARCH_CACHE_MAX, 1):
            _ARCH_CACHE.pop(next(iter(_ARCH_CACHE)))
            if device == "cuda":
                torch.cuda.empty_cache()
    _ARCH_CACHE[key] = m          # 末尾に入れ直して LRU にする
    sd = _SD_CACHE.get(spec["path"])
    if sd is None:
        # mmap は cold read がページフォールト単位のランダム I/O になり、
        # 遅いストレージで極端に落ちる（自ホストで 15MB/s まで低下）。逐次読みにする。
        sd = torch.load(spec["path"], map_location="cpu")
        _cache_sd(spec["path"], sd)
    # load_state_dict に dict を渡すと GPU 上に一時コピーがもう 1 セット出来るので、
    # 既存のパラメータ領域へ直接 copy_ する（転送 1 回・追加確保なし）。
    dst = m.state_dict()
    assert set(dst) == set(sd), f"state_dict のキーが一致しない: {spec['path']}"
    with torch.no_grad():
        for k, v in dst.items():
            v.copy_(sd[k], non_blocking=True)
    return m


def encoder_of(m):
    core = m.core
    return core.encoder if hasattr(core, "encoder") else core.m_fine.encoder


def load_alpha(device: str):
    p = os.path.join(MODEL_DIR, "alpha.json")
    if not os.path.exists(p):
        return None
    a = json.load(open(p))
    out = {}
    for task, n in (("fine", N_FINE), ("coarse", N_COARSE)):
        v = torch.ones(n, device=device)
        for k, val in a.get(task, {}).items():
            if int(k) < n:
                v[int(k)] = float(val)
        out[task] = v.view(1, -1, 1, 1)
    log(f"alpha loaded: fine {out['fine'].numel()} / coarse {out['coarse'].numel()} クラス")
    return out


def load_t3(device: str) -> dict:
    meta_path = os.path.join(MODEL_T3_DIR, "meta.json")
    if not os.path.exists(meta_path):
        return {}
    import lightgbm as lgb
    meta = json.load(open(meta_path))
    mlps = []
    for fn in sorted(f for f in os.listdir(MODEL_T3_DIR)
                     if f.startswith("mlp_") and f.endswith(".pt")):
        o = torch.load(os.path.join(MODEL_T3_DIR, fn), map_location="cpu", weights_only=False)
        sd = o["state_dict"]
        d_in = sd["0.weight"].shape[1]
        net = torch.nn.Sequential(
            torch.nn.Linear(d_in, sd["0.weight"].shape[0]), torch.nn.ReLU(inplace=True),
            torch.nn.Dropout(0.3), torch.nn.Linear(sd["3.weight"].shape[1], sd["3.weight"].shape[0]))
        net.load_state_dict(sd, strict=True)
        mlps.append({"net": net.to(device).float().eval(),
                     "mu": torch.tensor(o["mu"], device=device),
                     "sd": torch.tensor(o["sd"], device=device),
                     "encoder": o["encoder"], "fold": o["fold"]})
    boosters, consts = {}, {}
    for fn in sorted(os.listdir(MODEL_T3_DIR)):
        if fn.startswith("lgb_") and fn.endswith(".txt"):
            boosters[fn[4:-4]] = lgb.Booster(model_file=os.path.join(MODEL_T3_DIR, fn))
        elif fn.startswith("lgb_") and fn.endswith(".const"):
            consts[fn[4:-6]] = float(open(os.path.join(MODEL_T3_DIR, fn)).read())
    # v12: GBDT 枝は XGBoost vector-leaf（xgb_s<seed>_fold<k>.json）。LGBM ファイルが無ければこちらだけ使う
    xgbs = []
    xgb_files = sorted(f for f in os.listdir(MODEL_T3_DIR) if f.startswith("xgb_") and f.endswith(".json"))
    if xgb_files:
        import xgboost as xgb
        for fn in xgb_files:
            m = xgb.XGBClassifier()
            m.load_model(os.path.join(MODEL_T3_DIR, fn))
            m.set_params(device="cpu")      # 1 行ずつの推論なので CPU で十分・GPU 有無に依存しない
            xgbs.append(m)
    meta["mlps"], meta["boosters"], meta["consts"], meta["xgbs"] = mlps, boosters, consts, xgbs
    log(f"task3: MLP {len(mlps)} 本 / LGBM {len(boosters)} 本 / XGB {len(xgbs)} 本 / "
        f"閾値 {len(meta.get('thresholds', {}))} / 特徴 GBDT {len(meta['features'])} "
        f"MLP {len(meta.get('features_mlp', meta['features']))}")
    return meta


def calibrate(p: float, t: float) -> float:
    """t が 0.5 に来る区分線形写像（単調なので AUROC は不変）。"""
    t = min(max(t, 1e-3), 1 - 1e-3)
    return float(min(max(0.5 * p / t if p < t else 0.5 + 0.5 * (p - t) / (1 - t), 0.0), 1.0))


def remove_islands(lab: np.ndarray, ppm: float, per_class: dict | None = None) -> np.ndarray:
    """各クラスの小さい連結成分を「周囲の多数決クラス」で埋める.

    公式規約では GT に無いクラスを 1 画素でも出すとそのクラスが 0 点になるため、
    孤立した小さい偽陽性の除去が大きく効く（OOF: fine +0.032 / coarse +0.027、
    leave-one-center-out でも 6/6 改善）。しきい値は画像面積比で持つ。
    """
    if ppm <= 0 and not per_class:
        return lab
    out = lab.copy()
    for c in np.unique(lab):
        if c == 0:
            continue
        thr = int((per_class.get(str(int(c)), ppm) if per_class else ppm) * lab.size / 1_000_000)
        if thr <= 1:
            continue
        m = (lab == c).astype(np.uint8)
        n, cc, stats, _ = cv2.connectedComponentsWithStats(m, 8)
        if n <= 1:
            continue
        areas = stats[1:, cv2.CC_STAT_AREA]
        for i in np.where(areas < thr)[0]:
            x, y, w_, h_, _ = stats[i + 1]
            sl = (slice(max(y - 2, 0), y + h_ + 2), slice(max(x - 2, 0), x + w_ + 2))
            comp = cc[sl] == (i + 1)
            ring = cv2.dilate(comp.astype(np.uint8), np.ones((5, 5), np.uint8)) > 0
            ring &= ~comp
            vals = out[sl][ring]
            vals = vals[vals != c]
            out[sl][comp] = np.bincount(vals).argmax() if len(vals) else 0
    return out


def build_id2rgb(lm: pd.DataFrame):
    fine = np.zeros((256, 3), dtype=np.uint8)
    coarse = np.zeros((256, 3), dtype=np.uint8)
    for r in lm.itertuples():
        fine[int(r.fine_id)] = (r.fine_r, r.fine_g, r.fine_b)
        coarse[int(r.merged_id)] = (r.merged_r, r.merged_g, r.merged_b)
    return fine, coarse


def to_tensor(bgr: np.ndarray, ih: int, iw: int, device: str) -> torch.Tensor:
    rgb = cv2.cvtColor(cv2.resize(bgr, (iw, ih), interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2RGB)
    x = (rgb.astype(np.float32) / 255.0 - MEAN) / STD
    t = torch.from_numpy(x.transpose(2, 0, 1)).to(device)
    return t.half() if device == "cuda" else t


# ---------------------------------------------------------------- 予算配分
class Budget:
    """実測から「このチャンクで何本のメンバーを回せるか」を決める。"""

    def __init__(self, n_frames: int, n_members: int, n_t3enc: int):
        total = BUDGET_MIN * 60 if BUDGET_MIN > 0 else n_frames * N_TASKS * SEC_PER_FRAME_PER_TASK
        self.deadline = T0 + total * SAFETY
        self.n_frames = n_frames
        self.n_members, self.n_t3enc = n_members, n_t3enc
        # 初回チャンク用の保守的な初期値（GPU 想定より遅めに置く）
        self.seg_pf = 0.45      # メンバー 1 本 x フレーム 1 枚
        self.enc_pf = 0.45      # Task3 encoder 1 本 x フレーム 1 枚
        self.load = 8.0         # モデル 1 本のロード
        self.fixed_pf = 2.0     # デコード + 原寸 argmax + island + PNG 書き + 特徴量
        self.measured = False
        self.first_max = int(os.environ.get("FIRST_CHUNK_MAX", "6"))
        # Task3 は AUROC が全行を横断して計算されるので、フレームによって
        # 使う encoder が変わると確率のスケールが揃わない。**最初に 1 度だけ決めて固定**する。
        self.kt_fixed: "int | None" = None
        log(f"予算: 全体 {total / 60:.0f} 分 x 安全率 {SAFETY} → 締切 +{total * SAFETY / 60:.0f} 分")
        log(f"重み RAM キャッシュ上限 {_SD_CACHE_LIMIT / 1e9:.1f} GB (MemAvailable {_mem_available() / 1e9:.0f} GB)")

    def plan(self, n_done: int, n_this: int) -> tuple[int, int]:
        """残りフレーム数と残り時間から (seg メンバー数 K, Task3 encoder 数 Kt) を出す。"""
        rem_frames = self.n_frames - n_done
        rem = self.deadline - time.time()
        if rem_frames <= 0:
            return self.n_members, self.n_t3enc
        per_frame = rem / rem_frames                       # 1 枚に使ってよい秒数
        avail = per_frame - self.fixed_pf
        enc_cost = self.enc_pf + self.load / max(n_this, 1)
        if self.kt_fixed is None:
            # Task3 encoder には多くとも 50% を割く（LGBM 枝だけでも Task3 は成立するが、
            # 検証した構成は 15 本全部なので予算があれば全部使う）
            kt = int(max(0.0, avail * 0.50) // enc_cost) if self.n_t3enc else 0
            kt = max(0, min(kt, self.n_t3enc))
            if MAX_T3ENC:
                kt = min(kt, MAX_T3ENC)
            self.kt_fixed = kt
            log(f"Task3 encoder は全チャンク共通で {kt}/{self.n_t3enc} 本に固定")
        kt = self.kt_fixed
        seg_cost = self.seg_pf + self.load / max(n_this, 1)
        k = int(max(0.0, avail - kt * enc_cost) // seg_cost)
        k = max(1, min(k, self.n_members))
        if MAX_MEMBERS:
            k = min(k, MAX_MEMBERS)
        return k, kt

    def update(self, k: int, kt: int, n: int, t_seg: float, t_enc: float, t_fix: float,
               t_load_seg: float, t_load_enc: float):
        nl = max(k + kt, 1)
        new_load = (t_load_seg + t_load_enc) / nl
        self.load = new_load if not self.measured else 0.5 * self.load + 0.5 * new_load
        if k:
            v = t_seg / (k * n)
            self.seg_pf = v if not self.measured else 0.5 * self.seg_pf + 0.5 * v
        if kt:
            v = t_enc / (kt * n)
            self.enc_pf = v if not self.measured else 0.5 * self.enc_pf + 0.5 * v
        v = t_fix / n
        self.fixed_pf = v if not self.measured else 0.5 * self.fixed_pf + 0.5 * v
        self.measured = True
        log(f"実測: seg {self.seg_pf * 1000:.0f} ms/本/枚 | enc {self.enc_pf * 1000:.0f} ms/本/枚 "
            f"| load {self.load:.2f} s/本 | 後処理 {self.fixed_pf:.2f} s/枚")


# ---------------------------------------------------------------- 本体
@torch.no_grad()
def run_members(specs, frames, raw, device, acc_f, acc_c, gaps, for_t3: bool):
    """メンバーを 1 本ずつ載せ、チャンク内の全フレームをまとめて forward する。"""
    t_load = t_fwd = 0.0
    w_f, w_c = {}, {}                 # group -> その group で実際に回したメンバーの重み和
    for spec in specs:
        t = time.time()
        m = load_one(spec, device)
        t_load += time.time() - t
        ih, iw = spec["img"]
        t = time.time()
        for i in range(0, len(frames), BATCH):
            grp = frames[i:i + BATCH]
            x = torch.stack([to_tensor(raw[n], ih, iw, device) for n in grp])
            if for_t3:
                feats = encoder_of(m)(x)
                f = (feats[0] if isinstance(feats, tuple) else feats)[-1]
                g = f.float().mean((2, 3)).cpu()
                for j, n in enumerate(grp):
                    gaps[n][(spec["member"], spec["fold"])] = g[j]
                del feats, f
                continue
            lf, lc = m(x)
            w = spec["weight"]
            pf = lf.float().softmax(1)
            pc = lc.float().softmax(1)
            if spec["tta"]:
                lf2, lc2 = m(torch.flip(x, dims=[3]))
                pf = 0.5 * (pf + torch.flip(lf2.float().softmax(1), dims=[3]))
                pc = 0.5 * (pc + torch.flip(lc2.float().softmax(1), dims=[3]))
                del lf2, lc2
            pf = pf * w
            pc = pc * w
            g = spec["group"]
            for j, n in enumerate(grp):
                a = pf[j]
                d = acc_f.setdefault(n, {})
                ref = next(iter(d.values()), None)
                if ref is not None and a.shape[-2:] != ref.shape[-2:]:
                    a = torch.nn.functional.interpolate(a[None], size=ref.shape[-2:],
                                                        mode="bilinear", align_corners=False)[0]
                d[g] = a.clone() if g not in d else d[g] + a
                if spec["fine_only"]:
                    continue
                b = pc[j]
                d = acc_c.setdefault(n, {})
                ref = next(iter(d.values()), None)
                if ref is not None and b.shape[-2:] != ref.shape[-2:]:
                    b = torch.nn.functional.interpolate(b[None], size=ref.shape[-2:],
                                                        mode="bilinear", align_corners=False)[0]
                d[g] = b.clone() if g not in d else d[g] + b
            del lf, lc, pf, pc
        if not for_t3:
            w_f[spec["group"]] = w_f.get(spec["group"], 0.0) + spec["weight"]
            if not spec["fine_only"]:
                w_c[spec["group"]] = w_c.get(spec["group"], 0.0) + spec["weight"]
        t_fwd += time.time() - t
    return t_load, t_fwd, w_f, w_c


@torch.no_grad()
def calibrate_speed(bud, specs, t3_specs, names, device, alpha, t3) -> None:
    """先頭数枚で 1 本ずつ実測し、予算モデルの初期値を実機の値に置き換える.

    これをやらずに推定値で 1 チャンク目を組むと、遅い環境では最初のチャンクだけで
    予算を食い潰す。ここで測ってから Task3 encoder 本数を確定させる。
    """
    probe = names[: min(3, len(names))]
    t = time.time()
    raw = {n: cv2.imread(os.path.join(INPUT_DIR, n), cv2.IMREAD_COLOR) for n in probe}
    t_dec = (time.time() - t) / len(probe)

    t = time.time(); m = load_one(specs[0], device); t_load = time.time() - t
    ih, iw = specs[0]["img"]
    # 初回 forward は CUDA コンテキスト生成と cuDNN のアルゴリズム探索を含み
    # 10 倍近く遅い。これを計ると予算制御が過剰に保守的になるので先に捨て打ちする。
    warm = torch.stack([to_tensor(raw[probe[0]], ih, iw, device)])
    for _ in range(2):
        m(warm)
    if device == "cuda":
        torch.cuda.synchronize()
    del warm
    t = time.time()
    acc_f, acc_c = {}, {}
    _, _, cw_f, cw_c = run_members(specs[:1], probe, raw, device, acc_f, acc_c, {n: {} for n in probe}, False)
    if device == "cuda":
        torch.cuda.synchronize()
    bud.seg_pf = (time.time() - t) / len(probe)

    t_enc = 0.0
    if t3_specs:
        # encoder 側も初回は cuDNN 探索で 10 倍以上遅い。1 回捨て打ちしてから計る
        # （ここを計り損ねると Task3 encoder が 7/15 本に絞られ、検証した 45 ヘッド構成と変わる）
        run_members(t3_specs[:1], probe[:1], raw, device, {}, {}, {n: {} for n in probe}, True)
        if device == "cuda":
            torch.cuda.synchronize()
        t = time.time()
        run_members(t3_specs[:1], probe, raw, device, {}, {}, {n: {} for n in probe}, True)
        if device == "cuda":
            torch.cuda.synchronize()
        t_enc = (time.time() - t) / len(probe)
        bud.enc_pf = t_enc

    # 後段（原寸へ拡大 → island → PNG エンコード → Task3 特徴量）を 1 枚分だけ計る
    n = probe[0]
    oh, ow = raw[n].shape[:2]
    t = time.time()
    pf = group_mean(acc_f[n], cw_f)[None]
    pc = group_mean(acc_c[n], cw_c)[None]
    if alpha is not None:
        pf, pc = pf * alpha["fine"], pc * alpha["coarse"]
    ids_f = torch.nn.functional.interpolate(pf, size=(oh, ow), mode="bilinear",
                                            align_corners=False).argmax(1)[0].to(torch.uint8).cpu().numpy()
    ids_c = torch.nn.functional.interpolate(pc, size=(oh, ow), mode="bilinear",
                                            align_corners=False).argmax(1)[0].to(torch.uint8).cpu().numpy()
    ids_f = remove_islands(ids_f, ISLAND_PPM, ISLAND_PPM_MAP["fine"] if ISLAND_PPM_MAP else None)
    ids_c = remove_islands(ids_c, ISLAND_PPM, ISLAND_PPM_MAP["coarse"] if ISLAND_PPM_MAP else None)
    cv2.imencode(".png", np.zeros((oh, ow, 3), np.uint8))
    cv2.imencode(".png", np.zeros((oh, ow, 3), np.uint8))
    if t3:
        task3_features(ids_f, ids_c, set(t3["features"]))
    bud.fixed_pf = (time.time() - t) + t_dec
    bud.load = t_load
    bud.measured = True
    del acc_f, acc_c, pf, pc, raw, m
    if device == "cuda":
        torch.cuda.empty_cache()
    log(f"実機較正: seg {bud.seg_pf * 1000:.0f} ms/本/枚 | enc {bud.enc_pf * 1000:.0f} ms/本/枚 "
        f"| load {bud.load:.1f} s/本 | 後処理 {bud.fixed_pf:.2f} s/枚")


def group_mean(d: dict, gw: dict) -> torch.Tensor:
    """{group: 重み付き確率和} と {group: 重み和} から、group 内で正規化してから group 間で平均する。"""
    gs = [g for g in d if gw.get(g, 0) > 0]
    return torch.stack([d[g] / gw[g] for g in gs]).mean(0)


@torch.no_grad()
def main() -> int:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    log(f"device={device}" + ("" if device == "cuda" else "  ★GPU が見えていません。予算制御でメンバー数を絞ります"))
    if device == "cuda":
        log(f"gpu={torch.cuda.get_device_name(0)}")
    os.makedirs(os.path.join(OUTPUT_DIR, "task1"), exist_ok=True)
    os.makedirs(os.path.join(OUTPUT_DIR, "task2"), exist_ok=True)
    id2rgb_fine, id2rgb_coarse = build_id2rgb(pd.read_csv(LABELMAP))

    specs = list_members()
    assert specs, f"no models under {MODEL_DIR}"
    t3_specs = list_members(MODEL_T3ENC_DIR)
    alpha = load_alpha(device)
    t3 = load_t3(device)
    if not t3:
        t3_specs = []

    names = sorted(n for n in os.listdir(INPUT_DIR) if n.lower().endswith(".png"))
    assert names, f"no PNG inputs in {INPUT_DIR}"
    log(f"{len(names)} input frames")

    bud = Budget(len(names), len(specs), len(t3_specs))
    csv_f = writer = None
    if t3:
        csv_f = open(os.path.join(OUTPUT_DIR, "task3.csv"), "w", newline="")
        writer = csv.writer(csv_f)
        writer.writerow(["case_id"] + STATIONS)
        csv_f.flush()

    calibrate_speed(bud, specs, t3_specs, names, device, alpha, t3)
    done = 0
    used_hist = []
    for c0 in range(0, len(names), CHUNK):
        frames = names[c0:c0 + CHUNK]
        t = time.time()
        raw = {n: cv2.imread(os.path.join(INPUT_DIR, n), cv2.IMREAD_COLOR) for n in frames}
        for n in frames:
            assert raw[n] is not None, f"failed to read {n}"
        t_decode = time.time() - t

        k, kt = bud.plan(done, len(frames))
        # coarse ヘッドを持つメンバーが 1 本も入らないと Task1 が作れない
        while k < len(specs) and not any(not sp["fine_only"] for sp in specs[:k]):
            k += 1
        used_hist.append((k, kt))
        acc_f, acc_c = {}, {}
        gaps = {n: {} for n in frames}
        tl1, tf1, w_f, w_c = run_members(specs[:k], frames, raw, device, acc_f, acc_c, gaps, False)
        tl2, tf2, _, _ = run_members(t3_specs[:kt], frames, raw, device, acc_f, acc_c, gaps, True)
        assert sum(w_f.values()) > 0 and sum(w_c.values()) > 0, f"no usable members (w_f={w_f}, w_c={w_c})"

        t = time.time()
        for name in frames:
            stem = name.rsplit(".", 1)[0]
            oh, ow = raw[name].shape[:2]
            pf = group_mean(acc_f.pop(name), w_f)[None]
            pc = group_mean(acc_c.pop(name), w_c)[None]
            if alpha is not None:
                pf = pf * alpha["fine"]
                pc = pc * alpha["coarse"]
            ids_f = torch.nn.functional.interpolate(
                pf, size=(oh, ow), mode="bilinear", align_corners=False).argmax(1)[0].to(torch.uint8).cpu().numpy()
            ids_c = torch.nn.functional.interpolate(
                pc, size=(oh, ow), mode="bilinear", align_corners=False).argmax(1)[0].to(torch.uint8).cpu().numpy()
            del pf, pc
            ids_f = remove_islands(ids_f, ISLAND_PPM, ISLAND_PPM_MAP["fine"] if ISLAND_PPM_MAP else None)
            ids_c = remove_islands(ids_c, ISLAND_PPM, ISLAND_PPM_MAP["coarse"] if ISLAND_PPM_MAP else None)
            for ids, id2rgb, sub in [(ids_c, id2rgb_coarse, "task1"), (ids_f, id2rgb_fine, "task2")]:
                ok = cv2.imwrite(os.path.join(OUTPUT_DIR, sub, name),
                                 cv2.cvtColor(id2rgb[ids], cv2.COLOR_RGB2BGR))
                assert ok, f"failed to write {sub}/{name}"
            if t3:
                writer.writerow([stem] + [f"{v:.6f}" for v in
                                          task3_row(ids_f, ids_c, gaps[name], t3, device)])
                csv_f.flush()
            raw.pop(name)
        t_fix = time.time() - t + t_decode
        if device == "cuda":
            torch.cuda.empty_cache()
        done += len(frames)
        bud.update(k, kt, len(frames), tf1, tf2, t_fix, tl1, tl2)
        rem = bud.deadline - time.time()
        log(f"重みキャッシュ {_SD_BYTES[0] / 1e9:.1f} GB / {len(_SD_CACHE)} 本")
        log(f"[{done}/{len(names)}] seg {k}/{len(specs)} 本・t3enc {kt}/{len(t3_specs)} 本で処理 "
            f"| このチャンク {tl1 + tf1 + tl2 + tf2 + t_fix:.1f}s | 締切まで残り {rem / 60:.1f} 分")

    if csv_f is not None:
        csv_f.close()
        df = pd.read_csv(os.path.join(OUTPUT_DIR, "task3.csv"))
        assert list(df.columns) == ["case_id"] + STATIONS, df.columns
        assert len(df) == len(names), f"task3.csv {len(df)} != {len(names)}"
        assert ((df[STATIONS] >= 0) & (df[STATIONS] <= 1)).all().all()
    for sub in ("task1", "task2"):
        n_out = len([n for n in os.listdir(os.path.join(OUTPUT_DIR, sub)) if n.endswith(".png")])
        assert n_out == len(names), f"{sub}: {n_out} != {len(names)}"
    ks = [h[0] for h in used_hist]
    log(f"使用メンバー数: seg min {min(ks)} / max {max(ks)} / 平均 {sum(ks) / len(ks):.1f}")
    log(f"all outputs written and verified in {(time.time() - T0) / 60:.1f} min "
        f"(task1=coarse / task2=fine / task3.csv)")
    return 0


GRID_H, GRID_W = 6, 12


def task3_features(ids_f, ids_c, need: "set[str]") -> dict:
    """meta.json が要求する列だけを作る。

    在庫 315 次元だけのモデルと、解剖文脈・粗グリッドまで入った 1741 次元のモデルの
    どちらでも同じコードで動くようにする（学習側 3 スクリプトとの同値は
    verify_features2.py / parity チェックで最大差 0 を確認済み）。
    """
    f4, c4 = ids_f[::SUB, ::SUB], ids_c[::SUB, ::SUB]
    if any(k.startswith(("ln_", "fat_", "p_", "c_ct_")) for k in need):
        feat = all_features(f4, c4)
    else:
        feat = {}
        feat.update(mask_features(f4, N_FINE, "f"))
        feat.update(mask_features(c4, N_COARSE, "c"))
    if any(k.startswith("c_g") for k in need):
        cg = cv2.resize(ids_c, (GRID_W * 8, GRID_H * 8), interpolation=cv2.INTER_NEAREST)
        feat.update(grid_features(cg, N_COARSE, GRID_H, GRID_W, "c"))
    return feat


def task3_row(ids_f, ids_c, gap, t3, device) -> list[float]:
    # v12: GBDT 枝と MLP 枝で特徴列が違う（GBDT 1741 / MLP 315）。features_mlp が無ければ共通
    cols_g = t3["features"]
    cols_m = t3.get("features_mlp", cols_g)
    feat = task3_features(ids_f, ids_c, set(cols_g) | set(cols_m))
    missing = [c for c in cols_g + cols_m if c not in feat]
    assert not missing, f"meta.json の特徴が作れない: {missing[:5]} ({len(missing)} 個)"
    inv_g = np.array([feat[c] for c in cols_g], dtype=np.float32)
    inv_m = np.array([feat[c] for c in cols_m], dtype=np.float32)
    inv_t = torch.tensor(inv_m, device=device)
    mlp_probs = []
    for h in t3["mlps"]:
        key = (h["encoder"].replace("expA23_", ""), h["fold"])
        if key not in gap:
            continue
        x = torch.cat([gap[key].to(device), inv_t])
        x = (x - h["mu"]) / h["sd"]
        mlp_probs.append(torch.sigmoid(h["net"](x[None]))[0])
    gbdt_probs = {st: [] for st in STATIONS}
    if t3.get("xgbs"):
        # vector-leaf: 1 モデルで 14 station の確率が同時に出る
        Xg = inv_g[None]
        for m in t3["xgbs"]:
            pv = np.asarray(m.predict_proba(Xg), dtype=np.float64)[0]
            for k, st in enumerate(STATIONS):
                gbdt_probs[st].append(float(pv[k]))
    else:
        X = inv_g[None].astype(np.float64)
        for key, bst in t3["boosters"].items():
            st = key.rsplit("_", 1)[1]
            if st in gbdt_probs:
                gbdt_probs[st].append(float(bst.predict(X)[0]))
        for key, v in t3["consts"].items():
            st = key.rsplit("_", 1)[1]
            if st in gbdt_probs:
                gbdt_probs[st].append(v)
    w = float(t3.get("w_gbdt", t3.get("w_lgb", 0.4)))
    if not mlp_probs:
        w = 1.0            # ★予算で encoder を落とした場合は GBDT 枝だけで正規化する
        mlp_mean = np.zeros(len(STATIONS))
    else:
        mlp_mean = torch.stack(mlp_probs).mean(0).cpu().numpy()
    out = []
    for k, st in enumerate(STATIONS):
        pg = float(np.mean(gbdt_probs[st])) if gbdt_probs[st] else 0.0
        p = w * pg + (1 - w) * float(mlp_mean[k])
        out.append(calibrate(p, t3.get("thresholds", {}).get(st, 0.5)))
    return out


if __name__ == "__main__":
    sys.exit(main())
