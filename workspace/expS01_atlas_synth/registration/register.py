"""Stage 1: descriptor kNN -> weighted IoU. Estimates only; no real masks leave the host."""
import argparse
from copy import deepcopy
import json
from pathlib import Path
import numpy as np
import yaml
from .features import read_label,describe,distances,weighted_iou
from render.util import atomic_json,sha256


def pose_distance(a,b):
    x=np.array(a['camera_to_world_cv_mm']);y=np.array(b['camera_to_world_cv_mm'])
    angle=float(np.rad2deg(np.arccos(np.clip((np.trace(x[:3,:3].T@y[:3,:3])-1)/2,-1,1))))
    return float(np.linalg.norm(x[:3,3]-y[:3,3])),angle


def resized_camera(meta,size):
    camera=deepcopy(meta['camera']);w,h=meta['resolution'];sx,sy=size[0]/w,size[1]/h
    for key in ['K']:
        k=np.array(camera[key],float);k[0,0]*=sx;k[1,1]*=sy;k[0,2]=(k[0,2]+.5)*sx-.5;k[1,2]=(k[1,2]+.5)*sy-.5;camera[key]=k.tolist()
    # Overscan renderer calibration belongs to the original synthetic candidate only.
    camera['source_render_distortion']=camera.pop('distortion',None)
    d=camera['source_render_distortion'] or {}
    camera['distortion']={k:d[k] for k in ['model','k1','k2','p1','p2'] if k in d}
    camera['estimate_status']='retrieved_atlas_pose_not_patient_ground_truth'
    return camera


def confidence(candidates,cfg):
    best=candidates[0];q=cfg['confidence'];score=best['weighted_iou'];reasons=[]
    comparable=len(best['common_visible_classes'])
    if score<q['minimum_weighted_iou']:reasons.append('low weighted IoU')
    if best['descriptor_distance']>q['maximum_descriptor_distance']:reasons.append('descriptor out of support')
    if comparable<q['minimum_comparable_classes']:reasons.append('too few comparable classes')
    if best['comparable_pixel_fraction']<q['minimum_comparable_pixel_fraction']:reasons.append('insufficient anatomical pixel coverage')
    near=[v for v in candidates if score-v['weighted_iou']<=q['ambiguity_iou_tolerance']]
    translation=max(v['translation_from_best_mm'] for v in near);angle=max(v['angle_from_best_deg'] for v in near)
    if translation>q['maximum_ambiguous_translation_mm'] or angle>q['maximum_ambiguous_angle_deg']:reasons.append('ambiguous spatial hypotheses')
    support=min(1.,comparable/max(q['minimum_comparable_classes'],1))
    spatial=1/(1+translation/q['maximum_ambiguous_translation_mm']+angle/q['maximum_ambiguous_angle_deg'])
    value=float(score*np.exp(-best['descriptor_distance'])*support*spatial)
    if value<q['minimum_confidence']:reasons.append('low confidence')
    return value,reasons,{'near_equal_candidate_count':len(near),'translation_spread_mm':translation,'angle_spread_deg':angle,'top_score_margin':float(score-candidates[1]['weighted_iou']) if len(candidates)>1 else None}


def run(bank_path,masks,output,cfg,mapping):
    if output.exists() and any(output.iterdir()):raise ValueError('Registration output must be empty')
    output.mkdir(parents=True,exist_ok=True);(output/'poses').mkdir()
    manifest=json.loads((bank_path/'manifest.json').read_text());size=manifest['resolution']
    bank_hash=sha256(bank_path/'manifest.json')
    if size!=cfg['resolution'] or cfg['minimum_pixels']!=manifest['config']['minimum_pixels']:raise ValueError('Rebuild bank for changed descriptor resolution/minimum_pixels')
    ids=sorted(set(manifest['eligible_class_ids'])-set(cfg['exclude_class_ids'])-(set(manifest['provisional_class_ids'])-set(cfg['allow_provisional_class_ids'])))
    if not ids:raise ValueError('No eligible class intersection')
    with np.load(bank_path/'descriptors.npz',allow_pickle=False) as data:bank={k:data[k] for k in data.files}
    records=manifest['records'];weights={c['fine_id']:c['weight'] for c in mapping['classes']}
    files=sorted(masks.glob('*.png'));summary=[]
    if not files:raise ValueError('No local PNG masks')
    for path in files:
        try:
            original,_=read_label(path)
            observed=np.unique(original).tolist()
            real,original_size=read_label(path,size,cfg['aspect_ratio_tolerance']);desc=describe(real,cfg['minimum_pixels']);d=distances(desc,bank,ids,cfg['weights'])
            nearest=np.argsort(d,kind='stable')[:cfg['knn_k']];candidates=[]
            for index in nearest:
                rec=records[int(index)];synth,_=read_label(bank_path/'labels'/f"{rec['bank_id']}.png")
                match=weighted_iou(real,synth,ids,weights,cfg['minimum_pixels'],cfg.get('real_background_policy','ignore'))
                candidates.append({'bank_id':rec['bank_id'],'descriptor_distance':float(d[index]),**match,'source_meta':rec['meta']})
            candidates.sort(key=lambda x:(-x['weighted_iou'],x['descriptor_distance'],x['bank_id']))
            best=candidates[0]
            for item in candidates:
                t,r=pose_distance(best['source_meta']['camera'],item['source_meta']['camera']);item.update(translation_from_best_mm=t,angle_from_best_deg=r)
            value,reasons,diagnostics=confidence(candidates,cfg)
            if len(records)<cfg['output_top_k']:reasons.append('bank smaller than requested hypothesis count')
            top=[]
            for item in candidates[:cfg['output_top_k']]:
                m=item['source_meta'];top.append({k:v for k,v in item.items() if k!='source_meta'}|{'camera':resized_camera(m,original_size),'progress':m['dissection']['progress'],'instrument_count':m['instrument_count']})
            result={'frame_id':path.stem,'accepted':not reasons,'status':'estimated' if not reasons else 'rejected','confidence':value,'confidence_is_calibrated_probability':False,'failure_reasons':reasons,'diagnostics':diagnostics,'resolution':list(original_size),'camera':top[0]['camera'],'progress':top[0]['progress'],'top_k':top,'eligible_class_ids':ids,'excluded_class_ids':sorted(set(range(31))-set(ids)),'provisional_class_ids':manifest['provisional_class_ids'],'bank_manifest_sha256':bank_hash,'visible_stations':None,'station_nomenclature':'要確認','observed_presence_ids':observed,'alignment_policy':'nearest resize only; no image-space roll/scale optimization; roll/FOV from physically rendered candidates','coordinate_frame':'source atlas RAS mm; not actual patient coordinates'}
        except ValueError as e:
            result={'frame_id':path.stem,'accepted':False,'status':'input_rejected','failure_reasons':[str(e)],'confidence':0.}
        atomic_json(output/'poses'/f'{path.stem}.json',result)
        summary.append({k:result[k] for k in ['frame_id','accepted','confidence','failure_reasons']})
    atomic_json(output/'summary.json',{'frames':len(summary),'accepted':sum(r['accepted'] for r in summary),'frames_detail':summary,'config':cfg,'bank_manifest_sha256':bank_hash})
    print(f"Accepted {sum(r['accepted'] for r in summary)}/{len(summary)} (Stage 1 heuristic confidence)")


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--bank',type=Path,required=True);p.add_argument('--masks',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--config',type=Path,default=Path('configs/registration.yaml'));p.add_argument('--mapping',type=Path,default=Path('assets/class_mapping.yaml'));a=p.parse_args();run(a.bank,a.masks,a.output,yaml.safe_load(a.config.read_text()),yaml.safe_load(a.mapping.read_text()))


if __name__=='__main__':main()
