"""Replay saved synthetic poses, never search/resample for a new accepted frame."""
import argparse
import copy
import json
import shutil
from pathlib import Path
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor
import numpy as np
import yaml
import OpenEXR
from PIL import Image
from .util import atomic_json,save_meshes,sha256
from .volume import dissect
from .output import read_passes,remap,write_outputs
from .rgb import write_rgb

ROOT=Path(__file__).resolve().parents[1]


def saved_grid(pose,size,iterations=18):
    w,h=size;k=np.array(pose['K']);d=pose['distortion'];rk=np.array(d['render_K']);sw,sh=d['render_size'];yy,xx=np.mgrid[:h,:w]
    xd=(xx-k[0,2])/k[0,0];yd=(yy-k[1,2])/k[1,1];rd=np.hypot(xd,yd);ru=rd.copy()
    for _ in range(iterations):ru-=(ru*(1+d['k1']*ru**2+d['k2']*ru**4)-rd)/(1+3*d['k1']*ru**2+5*d['k2']*ru**4)
    ratio=np.divide(ru,rd,out=np.ones_like(rd),where=rd>0);sx=np.rint(xd*ratio*k[0,0]+rk[0,2]).astype(int);sy=np.rint(yd*ratio*k[1,1]+rk[1,2]).astype(int)
    valid=(sx>=0)&(sx<sw)&(sy>=0)&(sy<sh)
    return sx.clip(0,sw-1),sy.clip(0,sh-1),valid


def legacy_geometry(source,meta,cfg,provenance,mode):
    if meta['instrument_count']!=0:raise ValueError('Legacy instruments require saved geometry snapshots; cannot infer consumed RNG safely')
    path=source/'work/envelope.npz'
    if not path.exists():raise ValueError('Missing work/envelope.npz: restore batch work assets; seed alone is insufficient')
    with np.load(path) as f:volume={k:f[k].copy() for k in f.files}
    seed=meta['seed'];number=int(meta['frame_id']);rng=np.random.default_rng(np.random.SeedSequence([seed,number,838]));prior=cfg['dissection']['progress'];t=meta['dissection']['progress']
    if mode=='sampled':
        for _ in range(10000):
            sampled=float(rng.beta(prior['alpha'],prior['beta']))
            if prior.get('min',0)<=sampled<=prior.get('max',1):break
        if abs(sampled-t)>1e-12:raise ValueError('Saved progress is not sampled from stored prior; try --legacy-progress-mode fixed if original --progress was used')
    geometry,description=dissect(volume,provenance['landmarks_mm'],cfg['dissection'],t,rng)
    if description!=meta['dissection']:raise ValueError('Dissection metadata differs; restore original renderer version/geometry')
    return geometry


def worker(index,files,a,cfg,provenance):
    work=a.output/'work'/str(index);work.mkdir(parents=True);config=copy.deepcopy(cfg['camera']);config.update(emit_rgb=True,materials=yaml.safe_load(a.materials.read_text()))
    if a.engine=='cycles':config['materials']['engine']='CYCLES'
    config['device']='CPU' if a.engine=='eevee' else a.device
    atomic_json(work/'config.json',config)
    log=open(a.output/'logs'/f'{index}.log','w');process=subprocess.Popen([a.blender,'-b','--factory-startup','--python-exit-code','1','--python',str(ROOT/'render/blender_worker.py'),'--',str(work/'config.json'),str(a.source/'work/base.npz')],stdin=subprocess.PIPE,stdout=log,stderr=subprocess.STDOUT,text=True)
    rows=[]
    try:
        for f in files:
            start=time.perf_counter();m=json.loads(f.read_text());frame=f.stem;geometry=a.source/'geometry'/f'{frame}.npz'
            if geometry.exists():
                if m.get('geometry_sha256') and sha256(geometry)!=m['geometry_sha256']:raise ValueError('Geometry snapshot SHA mismatch')
            else:
                geometry=work/'geometry.npz';save_meshes(geometry,legacy_geometry(a.source,m,cfg,provenance,a.legacy_progress_mode))
            response=work/'response.json';raw=work/'raw.exr';response.unlink(missing_ok=True)
            job={'geometry':str(geometry),'geometry_key':frame,'camera':m['camera'],'response':str(response),'raw_exr':str(raw)}
            process.stdin.write(json.dumps(job)+'\n');process.stdin.flush();deadline=time.monotonic()+config['render_timeout_seconds']
            while not response.exists():
                if process.poll() is not None or time.monotonic()>deadline:raise RuntimeError(f'Worker failed/timeout: {log.name}')
                time.sleep(.05)
            r=json.loads(response.read_text())
            if not r['accepted_geometry']:raise ValueError(r)
            index_pass,depth=read_passes(raw);grid=saved_grid(m['camera'],m['resolution'],config['distortion']['inverse_iterations']);label,z=remap(index_pass,depth,m['camera'],grid)
            old=np.array(Image.open(a.source/'label'/f'{frame}.png'));mismatch=int((old!=label).sum())
            with OpenEXR.File(str(a.source/'depth'/f'{frame}.exr'),separate_channels=True) as old_exr:old_z=old_exr.channels()['Z'].pixels.copy()
            depth_error=float(np.max(np.abs(old_z-z)));depth_match=bool(np.allclose(old_z,z,rtol=1e-5,atol=.001))
            row={'depth_max_abs_error_mm':depth_error,'depth_match_within_tolerance':depth_match,'frame_id':frame,'label_mismatch_pixels':mismatch,'pixels':int(label.size),'render_seconds':r['render_seconds'],'source_label_sha256':sha256(a.source/'label'/f'{frame}.png'),'engine':r['engine']}
            if (mismatch or not depth_match) and not a.rerender_triplet:
                atomic_json(a.output/'mismatch'/f'{frame}.json',row);raise ValueError(f'{frame}: {mismatch} label pixels differ; depth max error={depth_error:.6g} mm; no RGB written. Use an explicitly approved engine or --rerender-triplet with the new controls.')
            write_outputs(a.output,frame,label,z);write_rgb(raw,a.output/'rgb'/f'{frame}.png',grid,z,config['materials'])
            shutil.copyfile(geometry,a.output/'geometry'/f'{frame}.npz')
            m.update(geometry_sha256=sha256(geometry),source_provenance_sha256=m.get('provenance_sha256'),source_config_sha256=m.get('config_sha256'),provenance_sha256=sha256(a.output/'provenance.json'),config_sha256=sha256(a.output/'resolved_configs.json'))
            m.update(source_meta_sha256=sha256(f),tier1=row,rgb={'same_render_as_label_depth':True,'shared_nearest_distortion_grid':True,'materials':config['materials']});atomic_json(a.output/'meta'/f'{frame}.json',m)
            row['wall_seconds']=time.perf_counter()-start;rows.append(row);print(frame,row,flush=True)
    finally:
        if process.poll() is None:
            try:process.stdin.write('{"stop":true}\n');process.stdin.flush();process.wait(timeout=15)
            except (BrokenPipeError,subprocess.TimeoutExpired):process.terminate();process.wait(timeout=10)
        log.close()
    return rows


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('source',type=Path);p.add_argument('--output',type=Path,required=True);p.add_argument('--blender',default='blender');p.add_argument('--materials',type=Path,default=Path('configs/materials.yaml'));p.add_argument('--workers',type=int,default=1);p.add_argument('--limit',type=int);p.add_argument('--engine',choices=['eevee','cycles'],default='eevee');p.add_argument('--device',choices=['CPU','CUDA','OPTIX'],default='CPU');p.add_argument('--rerender-triplet',action='store_true',help='Explicitly use newly rendered labels/depth instead of existing controls');p.add_argument('--legacy-progress-mode',choices=['sampled','fixed'],default='sampled');a=p.parse_args();a.source=a.source.resolve();a.output=a.output.resolve()
    if a.workers<1:p.error('workers >=1 required')
    if a.output.exists() and any(a.output.iterdir()):p.error('Output must be empty; source is never overwritten')
    if not (a.source/'work/base.npz').exists():p.error('Restore original batch work/base.npz; no geometry guessing')
    cfg=json.loads((a.source/'resolved_configs.json').read_text());provenance=json.loads((a.source/'provenance.json').read_text());files=sorted((a.source/'meta').glob('*.json'));files=[f for f in files if not f.name.endswith('.rejected.json')]
    if a.limit:files=files[:a.limit]
    if not files:p.error('No metadata')
    for name in ['rgb','label','depth','meta','logs','work','mismatch','geometry']:(a.output/name).mkdir(parents=True,exist_ok=True)
    shutil.copyfile(a.source/'work/base.npz',a.output/'work/base.npz')
    render_cfg=copy.deepcopy(cfg);render_cfg['camera'].update(emit_rgb=True,materials=yaml.safe_load(a.materials.read_text()))
    if a.engine=='cycles':render_cfg['camera']['materials']['engine']='CYCLES'
    atomic_json(a.output/'resolved_configs.json',render_cfg)
    atomic_json(a.output/'provenance.json',provenance|{'replay_source':str(a.source),'source_provenance_sha256':sha256(a.source/'provenance.json')})
    start=time.perf_counter()
    with ThreadPoolExecutor(max_workers=a.workers) as pool:
        tasks=[pool.submit(worker,i,files[i::a.workers],a,cfg,provenance) for i in range(min(a.workers,len(files)))];rows=[r for task in tasks for r in task.result()]
    elapsed=time.perf_counter()-start;atomic_json(a.output/'metrics.json',{'frames':len(rows),'total_seconds':elapsed,'amortized_seconds_per_frame':elapsed/len(rows),'rows':rows});print(f'{len(rows)} frames in {elapsed:.2f}s')

if __name__=='__main__':main()
