"""Host-side automated Blender integration check of inverse-square point lighting."""
import argparse
from pathlib import Path
import subprocess
import numpy as np
from .rgb import read_rgb
from .util import atomic_json


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--blender',default='blender');p.add_argument('--output',type=Path,required=True);a=p.parse_args();a.output=a.output.resolve()
    if a.output.exists() and any(a.output.iterdir()):p.error('Output must be empty')
    a.output.mkdir(parents=True,exist_ok=True)
    with open(a.output/'blender.log','w') as log:
        subprocess.run([a.blender,'-b','--factory-startup','--python-exit-code','1','--python',str(Path(__file__).with_name('verify_tier1_lighting.py')),'--',str(a.output)],stdout=log,stderr=subprocess.STDOUT,check=True)
    near=float(read_rgb(a.output/'50.exr')[30:34,30:34].mean());far=float(read_rgb(a.output/'100.exr')[30:34,30:34].mean());ratio=near/far
    passed=bool(np.isfinite(ratio) and abs(ratio-4)<.04);atomic_json(a.output/'verification.json',{'near_mm':50,'far_mm':100,'linear_rgb_ratio':ratio,'expected':4.,'passed':passed})
    if not passed:raise SystemExit(f'Inverse-square integration failure: {ratio}')
    print(f'PASS: 50mm/100mm linear RGB ratio={ratio:.6f}')

if __name__=='__main__':main()
