"""python -m render.generate --frames 100 --output outputs/tier0 --device CUDA"""
import argparse
from concurrent.futures import ThreadPoolExecutor
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import subprocess
import tempfile
import time
import numpy as np
import trimesh
import yaml
from .camera import RejectPose,sample_ports,sample_camera,distortion_grid,choice,unit
from .volume import build_volume,dissect
from .util import sha256,save_meshes,atomic_json
from .output import read_passes,remap,reject_reason,write_outputs

ROOT=Path(__file__).resolve().parents[1]


def load_yaml(path):
    with open(path) as stream:return yaml.safe_load(stream)


def validate(camera,dissection,atlas):
    w,h=camera['resolution']
    if min(w,h)<8 or max(w,h)>16384:raise ValueError('Unsupported resolution')
    if camera['filter_size']!=0.01:raise ValueError('Tier 0 requires filter_size=0.01')
    if len(camera['ports']['count']['values'])==0:raise ValueError('No port counts')
    if not set(camera['ports']['count']['values'])<={3,4}:raise ValueError('Use 3 or 4 ports')
    if not set(camera['ports']['interspaces']['values'])<=set(range(3,10)):raise ValueError('Use interspaces 3..9')
    end=0.
    for phase in dissection['phases']:
        start,new_end=phase['interval']
        if start!=end or new_end<=start:raise ValueError('Dissection intervals must partition [0,1]')
        for window in phase['windows']:
            w0,w1=window.get('interval',[start,new_end])
            if not start<=w0<w1<=new_end:raise ValueError('Window interval must be inside its phase')
        end=new_end
    if end!=1:raise ValueError('Dissection must end at t=1')
    if dissection['fat_thickness_mm']<=0 or dissection['pleura_thickness_mm']<=0 or dissection['pleura_surface_thickness_mm']<=0:
        raise ValueError('Fat and pleura may not be disabled')
    ids={int(k) for k in atlas['objects']}
    if not ids<=set(range(1,31)):raise ValueError('Invalid fine IDs')
    if atlas['context_class_id'] not in [0,2]:raise ValueError('Context ribs require explicit background/Other convention')


def prepare(args,camera,dissection,atlas,output):
    blend=Path(args.atlas or atlas['blend']).resolve()
    digest=sha256(blend)
    if digest!=atlas['sha256']:raise ValueError('Atlas SHA mismatch: update reviewed atlas manifest for this file')
    if atlas['status'].startswith('provisional') and not args.allow_provisional:
        raise ValueError('Task A identities/provenance are provisional. For exploratory output explicitly use --allow-provisional; reviewed training requires a reviewed manifest.')
    work=output/'work';work.mkdir()
    atlas_json=work/'atlas_config.json';atomic_json(atlas_json,atlas)
    exported=work/'atlas.npz'
    cmd=[args.blender,'-b',str(blend),'--disable-autoexec','--python-exit-code','1',
         '--python',str(ROOT/'render/export_atlas.py'),'--',str(atlas_json),str(exported)]
    with open(output/'logs'/'export.log','w') as log:
        subprocess.run(cmd,stdout=log,stderr=subprocess.STDOUT,check=True,timeout=args.startup_timeout)
    exported_meta=json.loads(exported.with_suffix('.json').read_text())
    objects=[]
    with np.load(exported,allow_pickle=False) as archive:
        for record in exported_meta['objects']:
            objects.append(record|{'v':archive[record['key']+'_v'].copy(),'f':archive[record['key']+'_f'].copy()})
    ribs={k:np.asarray(v) for k,v in exported_meta['ribs'].items()}
    landmarks=exported_meta['landmarks_mm']
    registration=None;patient=None
    if args.patient:
        from .registration import apply_patient
        patient=load_yaml(args.patient)
        if patient.get('status')!='reviewed':raise ValueError('CT patient manifest must be reviewed, with real paired landmarks')
        if not patient.get('license') or not patient.get('source_url'):raise ValueError('Patient provenance missing')
        objects,ribs,landmarks,registration=apply_patient(objects,ribs,landmarks,patient)
        atlas['roi_mm']=patient['roi_mm']
        camera['patient_id']=patient['patient_id']
        camera['ports']['thorax_center_xy_mm']=patient['thorax_center_xy_mm']
    collapse=atlas['right_lung_collapse']
    if collapse['enabled']:
        for o in objects:
            if o['fine_id']==18:
                # CT may combine both lungs: classify vertices by patient RAS side.
                right=o['v'][:,0]>camera['ports']['thorax_center_xy_mm'][0]
                center=o['v'][right].mean(axis=0) if right.any() else np.zeros(3)
                if patient:
                    center[0]=float(patient['right_lung_lateral_pivot_mm'])
                else:center[0]=collapse['lateral_pivot_mm']
                o['v'][right]=center+(o['v'][right]-center)*np.asarray(collapse['scale'])
    # Limit rendered anatomy to the declared thoracic ROI; never read challenge data.
    lo,hi=np.asarray(atlas['roi_mm'])
    for o in objects:
        centers=o['v'][o['f']].mean(axis=1)
        o['f']=o['f'][np.all((centers>=lo)&(centers<=hi),axis=1)]
    objects=[o for o in objects if len(o['f'])]
    # Validate every declared class retains at least one surface after ROI clipping.
    missing={int(k) for k in atlas['objects']}-{o['fine_id'] for o in objects}
    if missing:raise ValueError(f'ROI discarded declared classes: {missing}')
    print('Building union fat volume and pleural shell...',flush=True)
    volume=build_volume(objects,atlas,dissection)
    save_meshes(work/'base.npz',objects)
    np.savez_compressed(work/'envelope.npz',**volume)
    atomic_json(work/'camera_config.json',camera)
    atomic_json(output/'resolved_configs.json',{'camera':camera,'dissection':dissection,'atlas':atlas})
    rng=np.random.default_rng(np.random.SeedSequence([camera['seed'],991]))
    for _ in range(camera['ports']['max_layout_attempts']):
        ports=sample_ports(rng,ribs,camera)
        positions=np.array([p['position_mm'] for p in ports])
        targets=np.array([landmarks[n] for n in camera['targets']['names']])
        distances=np.linalg.norm(positions[:,None]-targets[None],axis=2)
        low,high=camera['ports']['port_to_target_mm']
        if np.all(np.any((distances>=low)&(distances<=high),axis=0)):break
    else:raise ValueError('No port layout reaches all nominated targets under distance limits')
    provenance={'atlas_sha256':digest,'atlas_source':atlas['source_url'],'atlas_license':atlas['license'],
                'atlas_attribution':atlas['attribution'],'manifest_status':atlas['status'],
                'registration':registration,'patient_manifest':patient,'ports':ports,
                'landmarks_mm':landmarks,'right_lung_collapse':collapse,
                'unavailable_class_ids':sorted(set(atlas['unavailable_class_ids'])-{o['fine_id'] for o in objects}),
                'config_sha256':hashlib.sha256(json.dumps([camera,dissection,atlas],sort_keys=True).encode()).hexdigest()}
    atomic_json(output/'provenance.json',provenance)
    return objects,volume,landmarks,ports,provenance


def instruments(rng,ports,camera,cfg):
    count=int(choice(rng,cfg['count']))
    allowed=[i for i in range(len(ports)) if i!=camera['scope_port_index']]
    rng.shuffle(allowed)
    result=[]
    for i in allowed[:count]:
        port=np.array(ports[i]['position_mm'])
        target=np.array(camera['target_mm'])+rng.normal(0,cfg['target_jitter_sd_mm'],3)
        # Stop short of the anatomical focus; collision validation follows in Blender.
        tip=target-unit(target-port)*12.
        shaft=trimesh.creation.cylinder(radius=cfg['shaft_radius_mm'],segment=[port,tip],sections=10)
        axis=unit(tip-port);side=unit(np.cross(axis,[0.,0.,1.]))
        meshes=[shaft]
        for sign in [-1,1]:
            jaw=tip+cfg['jaw_length_mm']*(np.cos(np.deg2rad(cfg['jaw_angle_deg']))*axis+sign*np.sin(np.deg2rad(cfg['jaw_angle_deg']))*side)
            meshes.append(trimesh.creation.cylinder(radius=cfg['shaft_radius_mm']*0.65,segment=[tip,jaw],sections=8))
        mesh=trimesh.util.concatenate(meshes)
        result.append({'name':f'instrument_port_{i}','fine_id':1,'v':mesh.vertices,'f':mesh.faces,
                       'port_mm':port.tolist(),'tip_mm':tip.tolist()})
    return result


def worker_loop(worker_id,frame_indices,args,camera,dissection,volume,landmarks,ports,provenance,output):
    work=output/'work'/f'worker_{worker_id}';work.mkdir()
    cmd=[args.blender,'-b','--factory-startup','--disable-autoexec','--python-exit-code','1',
         '--python',str(ROOT/'render/blender_worker.py'),'--',str(output/'work/camera_config.json'),str(output/'work/base.npz')]
    results=[]
    with open(output/'logs'/f'worker_{worker_id}.log','w') as log:
        process=subprocess.Popen(cmd,stdin=subprocess.PIPE,stdout=log,stderr=subprocess.STDOUT,text=True)
        try:
            for number in frame_indices:
                start=time.perf_counter()
                frame_id=f'{number:08d}'
                rng_t=np.random.default_rng(np.random.SeedSequence([camera['seed'],number,838]))
                t=float(args.progress if args.progress is not None else rng_t.beta(dissection['progress']['alpha'],dissection['progress']['beta']))
                # Patient noise is fixed; t controls a cumulative surgical window sequence.
                extra,dissection_meta=dissect(volume,landmarks,dissection,t,rng_t)
                rejections=Counter();done=False
                for attempt in range(camera['max_attempts_per_frame']):
                    rng=np.random.default_rng(np.random.SeedSequence([camera['seed'],number,attempt]))
                    try:
                        pose=sample_camera(rng,ports,landmarks,camera)
                        grid=distortion_grid(rng,pose,camera)
                    except RejectPose as error:
                        rejections[str(error)]+=1;continue
                    items=extra+instruments(rng,ports,pose,dissection['instruments'])
                    geometry=work/'geometry.npz';save_meshes(geometry,items)
                    response=work/'response.json';raw=work/'raw.exr'
                    if response.exists():response.unlink()
                    job={'geometry':str(geometry),'geometry_key':frame_id,'camera':pose,'response':str(response),'raw_exr':str(raw)}
                    if process.poll() is not None:raise RuntimeError(f'Blender worker exited: {output}/logs/worker_{worker_id}.log')
                    process.stdin.write(json.dumps(job)+'\n');process.stdin.flush()
                    deadline=time.monotonic()+camera['render_timeout_seconds']
                    while not response.exists():
                        if process.poll() is not None:raise RuntimeError(f'Blender exited ({process.returncode}); see worker log')
                        if time.monotonic()>deadline:raise TimeoutError('Blender render timeout; see worker log')
                        time.sleep(0.05)
                    result=json.loads(response.read_text())
                    if not result['accepted_geometry']:
                        if 'traceback' in result and result['reason']!='instrument crosses anatomy':
                            raise RuntimeError(result['traceback'])
                        rejections[result['reason']]+=1;continue
                    index,depth=read_passes(raw)
                    label,axial=remap(index,depth,pose,grid)
                    reason=reject_reason(label,grid[2],camera,t)
                    if reason:
                        rejections[reason]+=1;continue
                    write_outputs(output,frame_id,label,axial)
                    counts=np.bincount(label.ravel(),minlength=31)
                    elapsed=time.perf_counter()-start
                    meta={'frame_id':frame_id,'patient_id':camera['patient_id'],'seed':camera['seed'],
                          'attempt':attempt,'resolution':camera['resolution'],'camera':pose,'ports':ports,
                          'dissection':dissection_meta,'visible_class_ids':np.flatnonzero(counts[1:]).astype(int).__add__(1).tolist(),
                          'class_pixel_counts':{str(i):int(v) for i,v in enumerate(counts)},
                          'visible_stations':None,'station_nomenclature':'要確認',
                          'station_status':'not_implemented_task_D','depth':{'channel':'Z','unit':'mm',
                          'meaning':'linear camera axial Z from Cycles Depth.Z','invalid_value':0.,
                          'distorted':bool(camera['distortion']['enabled']),'interpolation':'nearest'},'label_encoding':'PNG uint8 grayscale; Object Index pass',
                          'missing_anatomy_class_ids':provenance['unavailable_class_ids'],
                          'provenance_sha256':sha256(output/'provenance.json'),'config_sha256':provenance['config_sha256'],
                          'geometry/render':result,'rejection_counts':dict(rejections),'frame_wall_seconds':elapsed,
                          'valid_optics_fraction':float(grid[2].mean()),'instrument_count':sum(o['fine_id']==1 for o in items),
                          'procedural_instance_counts':dict(Counter(str(o['fine_id']) for o in items))}
                    atomic_json(output/'meta'/f'{frame_id}.json',meta)
                    if args.keep_raw:raw.replace(output/'raw'/f'{frame_id}.exr')
                    print(f'Frame {frame_id}: t={t:.3f}, attempt={attempt}, {elapsed:.2f}s, classes={meta["visible_class_ids"]}',flush=True)
                    results.append({'frame_id':frame_id,'seconds':elapsed,'render_seconds':result['render_seconds'],'rejections':dict(rejections)})
                    done=True;break
                if not done:
                    atomic_json(output/'meta'/f'{frame_id}.rejected.json',{'progress':t,'rejections':dict(rejections)})
                    raise RuntimeError(f'{frame_id}: exhausted pose budget: {dict(rejections)}; inspect priors, do not silently relax physics')
        finally:
            if process.poll() is None:
                try:process.stdin.write('{"stop":true}\n');process.stdin.flush();process.wait(timeout=15)
                except (BrokenPipeError,subprocess.TimeoutExpired):process.terminate();process.wait(timeout=10)
    return results


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--frames',type=int,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--blender',default='blender');p.add_argument('--atlas');p.add_argument('--atlas-config',default='configs/atlas.yaml')
    p.add_argument('--camera-prior',default='configs/camera_prior.yaml');p.add_argument('--dissection',default='configs/dissection.yaml')
    p.add_argument('--workers',type=int,default=1);p.add_argument('--device',choices=['CPU','CUDA','OPTIX'])
    p.add_argument('--threads-per-worker',type=int,default=2)
    p.add_argument('--seed',type=int);p.add_argument('--resolution',nargs=2,type=int);p.add_argument('--progress',type=float)
    p.add_argument('--patient',help='Optional reviewed public CT masks + paired-landmark YAML')
    p.add_argument('--allow-provisional',action='store_true');p.add_argument('--keep-raw',action='store_true')
    p.add_argument('--startup-timeout',type=float,default=240)
    args=p.parse_args()
    if args.frames<1 or args.workers<1:p.error('frames and workers must be positive')
    if args.progress is not None and not 0<=args.progress<=1:p.error('progress must be in [0,1]')
    camera=load_yaml(args.camera_prior);dissection=load_yaml(args.dissection);atlas=load_yaml(args.atlas_config)
    if args.device:camera['device']=args.device
    if args.seed is not None:camera['seed']=args.seed
    if args.resolution:camera['resolution']=args.resolution
    camera['threads_per_worker']=args.threads_per_worker
    validate(camera,dissection,atlas)
    output=args.output.resolve()
    if output.exists() and any(output.iterdir()):p.error('Output must be absent or empty (no overwrite)')
    output.mkdir(parents=True,exist_ok=True)
    for name in ['label','depth','meta','logs','raw']: (output/name).mkdir()
    atomic_json(output/'resolved_configs.json',{'camera':camera,'dissection':dissection,'atlas':atlas})
    started=time.perf_counter()
    _,volume,landmarks,ports,provenance=prepare(args,camera,dissection,atlas,output)
    prepared=time.perf_counter()
    workers=min(args.workers,args.frames)
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures=[pool.submit(worker_loop,i,list(range(i,args.frames,workers)),args,camera,dissection,volume,landmarks,ports,provenance,output) for i in range(workers)]
        results=[r for f in futures for r in f.result()]
    elapsed=time.perf_counter()-started
    atomic_json(output/'metrics.json',{'frames':len(results),'workers':workers,'device':camera['device'],
        'preparation_seconds':prepared-started,'total_seconds':elapsed,'amortized_seconds_per_frame':elapsed/len(results),
        'median_render_seconds':float(np.median([r['render_seconds'] for r in results])),'frames_detail':results})
    print(f'Completed {len(results)} frames in {elapsed:.1f}s: {output}',flush=True)


if __name__=='__main__':main()
