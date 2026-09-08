"""Build a local search bank from one or more complete render.generate batches."""
import argparse
import json
from pathlib import Path
import numpy as np
import yaml
from .features import describe,read_label
from render.util import atomic_json,sha256


def build(batches,output,cfg,calibration_report=None):
    from .calibration_gate import require_report
    gate=require_report(calibration_report)
    if output.exists() and any(output.iterdir()):raise ValueError('Bank output must be empty')
    output.mkdir(parents=True,exist_ok=True);(output/'labels').mkdir()
    descriptors=[];records=[];supported=None;provisional=set();provenances=[];scene_signatures=set()
    for batch in batches:
        provenance=json.loads((batch/'provenance.json').read_text());provenances.append(provenance)
        # Capabilities derive from manifest, never from class occurrence in a small bank.
        cap=set(range(1,31))-set(provenance['unavailable_class_ids'])
        prov={int(r['fine_id']) for r in provenance.get('provisional_structures',[])};provisional|=prov
        cap-=prov-set(cfg['allow_provisional_class_ids']);supported=cap if supported is None else supported&cap
        resolved=json.loads((batch/'resolved_configs.json').read_text())
        require_report(calibration_report,resolved)
        atlas=resolved['atlas'];scene_signatures.add(json.dumps([provenance['atlas_sha256'],provenance['registration'],atlas.get('roi_mm'),atlas.get('right_lung_collapse'),atlas.get('provisional')],sort_keys=True))
        files=sorted((batch/'label').glob('*.png'))
        if (batch/'metrics.json').exists():
            if len(files)!=json.loads((batch/'metrics.json').read_text())['frames']:raise ValueError('Incomplete batch')
        else:raise ValueError('Bank requires complete successful batch metrics')
        for f in files:
            m=json.loads((batch/'meta'/f'{f.stem}.json').read_text())
            a,size=read_label(f,cfg['resolution'],cfg['aspect_ratio_tolerance']);descriptors.append(describe(a,cfg['minimum_pixels']))
            name=f'{len(records):08d}';from PIL import Image
            Image.fromarray(a).save(output/'labels'/f'{name}.png')
            records.append({'bank_id':name,'source_frame_id':f.stem,'source_batch':str(batch.resolve()),'source_label_sha256':sha256(f),'resolution':list(size),'meta':m})
    if not records:raise ValueError('Empty bank')
    if len(scene_signatures)!=1:raise ValueError('Use separate banks for different anatomy/TPS/collapse/proxy scenes; poses require a consistent atlas coordinate frame')
    ids=sorted(supported-set(cfg['exclude_class_ids']))
    if not ids:raise ValueError('No comparable class capabilities')
    np.savez_compressed(output/'descriptors.npz',**{k:np.stack([d[k] for d in descriptors]) for k in descriptors[0]})
    presence=np.stack([d['present'] for d in descriptors]).mean(0)
    cameras=[r['meta']['camera'] for r in records]
    coverage={'frames':len(records),'eligible_class_presence_pct':{str(i):float(presence[i]*100) for i in ids},
              'capable_but_unseen_class_ids':[i for i in ids if presence[i]==0],
              'target_counts':{name:sum(p['target_name']==name for p in cameras) for name in sorted({p['target_name'] for p in cameras})},
              'pose_bounds':{key:[float(min(p[key] for p in cameras)),float(max(p[key] for p in cameras))] for key in ['sensor_roll_deg','working_distance_mm','horizontal_fov_deg']},
              'warning':'Bank coverage is empirical, not proof of anatomical identifiability. Unseen-capable classes remain penalized; expand bank or explicitly revise exclusion config.'}
    atomic_json(output/'coverage.json',coverage)
    if any(presence[i]==0 for i in gate['required_bank_class_ids']):
        raise ValueError('Actual bank still lacks required IDs 12/8/9; inspect coverage.json; refusing to publish its manifest')
    atomic_json(output/'manifest.json',{'schema_version':1,'resolution':cfg['resolution'],'config':cfg,'eligible_class_ids':ids,'provisional_class_ids':sorted(provisional),'records':records,'provenances':provenances,'descriptor_version':1})
    print(f'{len(records)} bank frames; eligible IDs {ids}')


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('batches',nargs='+',type=Path);p.add_argument('--output',type=Path,required=True);p.add_argument('--config',type=Path,default=Path('configs/registration.yaml'));p.add_argument('--calibration-report',type=Path,required=True);a=p.parse_args();build(a.batches,a.output,yaml.safe_load(a.config.read_text()),a.calibration_report)


if __name__=='__main__':main()
