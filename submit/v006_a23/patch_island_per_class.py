"""process.py にクラス別の島除去しきい値（model/island_ppm.json = {"fine": {id: ppm}, "coarse": {id: ppm}}）を入れる。
json が無ければ従来どおり ISLAND_PPM 一律。適用:  python3 patch_island_per_class.py  (冪等)"""
import re, pathlib
p = pathlib.Path(__file__).with_name("process.py"); s = p.read_text()
if "ISLAND_PPM_MAP" in s:
    print("already patched"); raise SystemExit
s = s.replace('ISLAND_PPM = float(os.environ.get("ISLAND_PPM", "4000"))',
 'ISLAND_PPM = float(os.environ.get("ISLAND_PPM", "4000"))\n'
 '_ppm_json = os.path.join(MODEL_DIR, "island_ppm.json")   # クラス別しきい値 {"fine": {id: ppm}, "coarse": {id: ppm}}\n'
 'ISLAND_PPM_MAP = json.load(open(_ppm_json)) if os.path.exists(_ppm_json) else None', 1)
s = s.replace('def remove_islands(lab: np.ndarray, ppm: float) -> np.ndarray:', 'def remove_islands(lab: np.ndarray, ppm: float, per_class: dict | None = None) -> np.ndarray:', 1)
old = '''    if ppm <= 0:
        return lab
    thr = int(ppm * lab.size / 1_000_000)
    if thr <= 1:
        return lab
    out = lab.copy()
    for c in np.unique(lab):
        if c == 0:
            continue
'''
new = '''    if ppm <= 0 and not per_class:
        return lab
    out = lab.copy()
    for c in np.unique(lab):
        if c == 0:
            continue
        thr = int((per_class.get(str(int(c)), ppm) if per_class else ppm) * lab.size / 1_000_000)
        if thr <= 1:
            continue
'''
assert old in s, "remove_islands body changed"; s = s.replace(old, new, 1)
n = s.count('ids_f = remove_islands(ids_f, ISLAND_PPM)')
s = s.replace('ids_f = remove_islands(ids_f, ISLAND_PPM)', 'ids_f = remove_islands(ids_f, ISLAND_PPM, ISLAND_PPM_MAP["fine"] if ISLAND_PPM_MAP else None)')
s = s.replace('ids_c = remove_islands(ids_c, ISLAND_PPM)', 'ids_c = remove_islands(ids_c, ISLAND_PPM, ISLAND_PPM_MAP["coarse"] if ISLAND_PPM_MAP else None)')
p.write_text(s); print(f"patched ({n} call sites x2)")
