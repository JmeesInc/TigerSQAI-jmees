"""Round 4: compare anatomy IDs 3..30, per-frame anatomy-normalized area."""
import argparse
from pathlib import Path
import json
import numpy as np
import yaml
from PIL import Image
from .util import atomic_json
from .acceptance import evaluate


def aggregate(files):
    counts=[]
    for f in files:
        with Image.open(f) as im:
            if im.mode!='L':raise ValueError(f'{f}: require L label IDs')
            a=np.asarray(im)
        if a.max()>30:raise ValueError('ID outside 0..30')
        counts.append(np.bincount(a.ravel(),minlength=31))
    if not counts:raise ValueError('No labels')
    c=np.array(counts);den=c[:,3:].sum(1);area=np.zeros_like(c,dtype=float)
    np.divide(c[:,3:]*100,den[:,None],out=area[:,3:],where=den[:,None]>0)
    bg=100*c[:,0]/c.sum(1)
    return {'frames':len(c),'presence_pct':((c>0).mean(0)*100).tolist(),'mean_area_pct':area.mean(0).tolist(),'empty_anatomy_frames':int((den==0).sum()),'visible_class_count':dict(zip(['p10','median','p90'],np.percentile((c[:,3:]>0).sum(1),[10,50,90]).tolist())),'background_summary':{'median_pct':float(np.median(bg)),'mean_pct':float(bg.mean())}}


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('batch',type=Path);p.add_argument('--reference',type=Path,default=Path('assets/reference_anatomy_only.yaml'));p.add_argument('--acceptance',type=Path,default=Path('configs/acceptance.yaml'));p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    ref=yaml.safe_load(a.reference.read_text());s=aggregate(sorted((a.batch/'label').glob('*.png')));gate=evaluate(s,ref,yaml.safe_load(a.acceptance.read_text()))
    metrics=a.batch/'metrics.json';gate['batch_complete']=bool(metrics.exists() and json.loads(metrics.read_text()).get('frames')==s['frames'] and not any((a.batch/'meta').glob('*.rejected.json')))
    if not gate['batch_complete']:gate.update(acceptance_passed=False,bank_generation_allowed=False,status='incomplete_batch')
    md=['# Round 4 anatomy-only calibration','',f"N={s['frames']}; IDs 0/1/2: 対象外。面積はフレームごとにID 3–30画素で割り、全フレームで平均。背景率のみ別途全画面で測定。",'','|ID|Class|出現 実%|合成%|差 pp|解剖面積 実%|合成%|差 pp|','|---:|---|---:|---:|---:|---:|---:|---:|']
    rows=[]
    for r in ref['classes']:
        i=r['fine_id'];pr=s['presence_pct'][i];ar=s['mean_area_pct'][i];rows.append({'fine_id':i,'presence_pct':pr,'mean_area_pct':ar})
        md.append(f"|{i}|{r['name']}|{r['presence_pct']:.2f}|{pr:.2f}|{pr-r['presence_pct']:+.2f}|{r['mean_area_pct']:.2f}|{ar:.2f}|{ar-r['mean_area_pct']:+.2f}|")
    md+=['',f"解剖クラス数: {s['visible_class_count']}; 目標: {ref['visible_class_count']}",f"空の解剖フレーム: {s['empty_anatomy_frames']}",'',f"Status: **{gate['status']}**; bank allowed: **{gate['bank_generation_allowed']}**",'','|Metric|Baseline|Target|Synthetic|Pass|','|---|---:|---|---:|---|']
    for k,v in gate['actual'].items():md.append(f"|{k}|{gate['baseline'][k]:.2f}|{gate['targets'][k]}|{v:.2f}|{gate['improved_toward_target'][k]}|")
    md+=['',f"Required IDs absent: {gate['missing_required_bank_class_ids']}"]
    resolved=a.batch/'resolved_configs.json';result={'schema_version':2,'synthetic':s,'reference':ref,'classes':rows,'acceptance':gate,'render_configuration':json.loads(resolved.read_text()) if resolved.exists() else None}
    a.output.mkdir(parents=True,exist_ok=True);atomic_json(a.output/'comparison.json',result);(a.output/'comparison.md').write_text('\n'.join(md)+'\n');print(json.dumps(gate,indent=2))

if __name__=='__main__':main()
