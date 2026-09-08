"""Generate independent patient-port layouts via separate seeds, then index the successful batches."""
import argparse
from pathlib import Path
import subprocess
import sys
import yaml
from .bank import build


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--frames',type=int,default=2048);p.add_argument('--batches',type=int,default=8);p.add_argument('--output',type=Path,required=True);p.add_argument('--blender',default='blender');p.add_argument('--workers',type=int,default=2);p.add_argument('--device',default='CPU',choices=['CPU','CUDA','OPTIX']);p.add_argument('--seed',type=int,default=20260908);p.add_argument('--config',type=Path,default=Path('configs/registration.yaml'));p.add_argument('--camera-prior',default='configs/camera_prior.yaml');p.add_argument('--dissection',default='configs/dissection.yaml');p.add_argument('--atlas-config',default='configs/atlas.yaml');p.add_argument('--allow-provisional',action='store_true');p.add_argument('--disable-provisional-structures',action='store_true');p.add_argument('--calibration-report',type=Path,required=True);a=p.parse_args()
    from .calibration_gate import require_report
    require_report(a.calibration_report)
    if not 1<=a.batches<=a.frames:p.error('Require 1 <= batches <= frames')
    if a.output.exists() and any(a.output.iterdir()):p.error('Output must be empty')
    cfg=yaml.safe_load(a.config.read_text())
    camera=yaml.safe_load(Path(a.camera_prior).read_text());camera['resolution']=cfg['resolution']
    atlas=yaml.safe_load(Path(a.atlas_config).read_text())
    if a.disable_provisional_structures:atlas.setdefault('provisional',{})['enabled']=False
    require_report(a.calibration_report,{'camera':camera,'dissection':yaml.safe_load(Path(a.dissection).read_text()),'atlas':atlas})
    a.output.mkdir(parents=True,exist_ok=True);batches=[]
    for i in range(a.batches):
        batch=a.output/f'batch_{i:03d}';batches.append(batch);n=a.frames//a.batches+(i<a.frames%a.batches)
        cmd=[sys.executable,'-m','render.generate','--frames',str(n),'--output',str(batch),'--blender',a.blender,'--workers',str(a.workers),'--device',a.device,'--seed',str(a.seed+i*104729),'--resolution',*map(str,cfg['resolution']),'--camera-prior',a.camera_prior,'--dissection',a.dissection,'--atlas-config',a.atlas_config]
        if a.allow_provisional:cmd.append('--allow-provisional')
        if a.disable_provisional_structures:cmd.append('--disable-provisional-structures')
        subprocess.run(cmd,check=True)
    build(batches,a.output/'bank',cfg,a.calibration_report)


if __name__=='__main__':main()
