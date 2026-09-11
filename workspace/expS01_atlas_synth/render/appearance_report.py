"""Rendered synthetic chroma diagnostics against shared aggregate references."""
import argparse,json
from pathlib import Path
import numpy as np
import yaml
from PIL import Image


def report(batch,reference):
    samples={};frames={}
    for file in sorted((batch/'rgb').glob('*.png')):
        rgb=np.array(Image.open(file).convert('RGB'),float)/255;label=np.array(Image.open(batch/'label'/file.name));total=rgb.sum(2);high=rgb.max(2);low=rgb.min(2)
        chroma=np.divide(rgb,total[...,None],out=np.zeros_like(rgb),where=total[...,None]>0);sat=np.divide(high-low,high,out=np.zeros_like(high),where=high>0)
        for i in np.unique(label):
            mask=(label==i)&(total>0)
            if mask.any():samples.setdefault(int(i),[]).append(np.r_[np.median(chroma[mask],axis=0),np.median(sat[mask])]);frames[int(i)]=frames.get(int(i),0)+1
    result=[]
    for row in reference:
        i=row['fine_id'];target=np.array([row['chroma_r'],row['chroma_g'],row['chroma_b'],row['saturation']])
        actual=np.median(samples[i],axis=0) if i in samples else None
        result.append({'fine_id':i,'frames':frames.get(i,0),'target':target.tolist(),'rendered':actual.tolist() if actual is not None else None,'delta':(actual-target).tolist() if actual is not None else None})
    return {'definition':'Median of per-frame per-class pixel medians in saved sRGB; not guaranteed identical to reference extractor aggregation','columns':['r','g','b','HSV_saturation'],'classes':result}


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('batch',type=Path);p.add_argument('--reference',type=Path,default=Path('assets/real_class_chroma.yaml'));p.add_argument('--output',type=Path);a=p.parse_args();ref=yaml.safe_load(a.reference.read_text())['classes'];r=report(a.batch,ref);(a.output or a.batch/'appearance.json').write_text(json.dumps(r,indent=2))
    for row in r['classes']:
        if row['frames']:print(row)

if __name__=='__main__':main()
