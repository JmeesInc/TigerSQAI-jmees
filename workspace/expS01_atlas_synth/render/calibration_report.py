"""Compare a synthetic label batch with shared aggregate statistics only."""
import argparse
from pathlib import Path
import json
import numpy as np
import yaml
from PIL import Image
from .util import atomic_json


def aggregate(files):
    ratios=[]
    for f in files:
        with Image.open(f) as im:
            if im.mode!='L':raise ValueError(f'{f}: expected grayscale uint8 labels')
            a=np.asarray(im)
        if a.max()>30:raise ValueError(f'{f}: ID >30')
        ratios.append(np.bincount(a.ravel(),minlength=31)/a.size)
    if not ratios:raise ValueError('No label PNGs')
    a=np.array(ratios)
    return {'frames':len(a),'presence_pct':(a>0).mean(0)*100,'mean_area_pct':a.mean(0)*100,
            'conditional_median_area_pct':[float(np.median(v[v>0])*100) if (v>0).any() else None for v in a.T],
            'visible_class_count':dict(zip(['p10','median','p90'],np.percentile((a[:,1:]>0).sum(1),[10,50,90]).tolist())),
            'background_summary':dict(zip(['mean_pct','median_pct','p90_pct'],[float(a[:,0].mean()*100),*np.percentile(a[:,0]*100,[50,90]).tolist()]))}


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('batch',type=Path);p.add_argument('--reference',type=Path,default=Path('assets/real_mask_statistics.yaml'));p.add_argument('--output',type=Path,required=True)
    p.add_argument('--acceptance',type=Path,default=Path('configs/acceptance.yaml'))
    a=p.parse_args();ref=yaml.safe_load(a.reference.read_text());s=aggregate(sorted((a.batch/'label').glob('*.png')))
    a.output.mkdir(parents=True,exist_ok=True)
    rows=[];md=['# 合成／実集計の差分（単位：%、差はpercentage points）','',f"合成 {s['frames']}枚 / 参照 {ref['frames']}枚。可視=1画素以上、面積平均は不出現を含む。",'', '|ID|Class|出現 実|合成|差|面積平均 実|合成|差|出現時中央値 実|合成|差|','|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
    for r in ref['classes']:
        i=r['fine_id'];row={'fine_id':i,'name':r['name']}
        cells=[str(i),r['name']]
        for key in ['presence_pct','mean_area_pct','conditional_median_area_pct']:
            real=r[key];syn=s[key][i];delta=float(syn-real) if syn is not None and real is not None else None
            if i==0 and key=='conditional_median_area_pct':delta=None
            row[key]={'reference':real,'synthetic':float(syn) if syn is not None else None,'delta_pp':delta}
            cells.extend(['—' if v is None else f'{v:.2f}' for v in [real,syn,delta]])
        rows.append(row);md.append('|'+ '|'.join(cells)+'|')
    bg=s['background_summary']
    md+=['','背景の全フレーム中央値はユーザー追認の18.0%を採用。表の出現時中央値33.96%は比較しない。',
         f"背景中央値（全フレーム）: 実 {ref['background_summary']['median_pct']:.2f}% / 合成 {bg['median_pct']:.2f}% / 差 {bg['median_pct']-ref['background_summary']['median_pct']:+.2f} pp。", '', '|可視クラス数（背景除外）|実|合成|差|','|---|---:|---:|---:|']
    for key,v in s['visible_class_count'].items():md.append(f"|{key}|{ref['visible_class_count'][key]}|{v:.2f}|{v-ref['visible_class_count'][key]:+.2f}|")
    mean_sum=sum(r['mean_area_pct'] for r in ref['classes'])
    md+=['',f'参照の平均面積合計={mean_sum:.2f}%。丸め／集計定義を要確認。正規化による書き換えはしない。','', '出現率は配置確率ではない。小バッチの差分だけで改善・収束を断定しない。']
    from .acceptance import evaluate
    gate=evaluate(s,ref,yaml.safe_load(a.acceptance.read_text()))
    metrics_path=a.batch/'metrics.json'
    complete=metrics_path.exists() and json.loads(metrics_path.read_text()).get('frames')==s['frames'] and not any((a.batch/'meta').glob('*.rejected.json'))
    gate['batch_complete']=bool(complete)
    if not complete:
        gate.update(acceptance_passed=False,bank_generation_allowed=False,status='incomplete_batch')
    md+=['','## Round 3 acceptance','',f"Status: **{gate['status']}**; bank generation allowed: **{gate['bank_generation_allowed']}**",'','|指標|baseline|target|synthetic|targetへ改善|','|---|---:|---:|---:|---|']
    for k,v in gate['actual'].items():md.append(f"|{k}|{gate['baseline'][k]:.2f}|{gate['targets'][k]:.2f}|{v:.2f}|{gate['improved_toward_target'][k]}|")
    md+=['',f"バンク必須IDで未出現: {gate['missing_required_bank_class_ids']}。判定には{gate['minimum_frames']}枚以上が必要。"]
    (a.output/'comparison.md').write_text('\n'.join(md)+'\n')
    result={'synthetic':{k:(v.tolist() if isinstance(v,np.ndarray) else v) for k,v in s.items()},'reference':ref,'classes':rows,'reference_mean_sum_pct':mean_sum}
    result['acceptance']=gate
    resolved=a.batch/'resolved_configs.json'
    result['render_configuration']=json.loads(resolved.read_text()) if resolved.exists() else None
    atomic_json(a.output/'comparison.json',result);print(a.output/'comparison.md')


if __name__=='__main__':main()
