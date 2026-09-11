"""Label/RGB boundary evidence, not a proof that every tissue boundary must be bright."""
import argparse
from pathlib import Path
import json
import numpy as np
from PIL import Image


def boundary_metrics(label,rgb):
    if rgb.shape!=(*label.shape,3):raise ValueError('RGB/label dimension mismatch')
    x=rgb.astype(float)/255
    differences=np.concatenate([np.linalg.norm(x[:,1:]-x[:,:-1],axis=2).ravel(),np.linalg.norm(x[1:]-x[:-1],axis=2).ravel()])
    boundary=np.concatenate([(label[:,1:]!=label[:,:-1]).ravel(),(label[1:]!=label[:-1]).ravel()])
    if not boundary.any():return {'assessable':False,'reason':'no class boundary'}
    edge=float(differences[boundary].mean());inside=float(differences[~boundary].mean()) if (~boundary).any() else 0.
    return {'assessable':True,'boundary_contrast':edge,'interior_contrast':inside,'enrichment':edge/max(inside,1e-8),'boundary_pairs':int(boundary.sum()),'edge_evidence_pass':bool(edge>.002 and edge>inside*1.05)}


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('batch',type=Path);p.add_argument('--output',type=Path);p.add_argument('--require-edge-evidence',action='store_true');a=p.parse_args();rows=[]
    for f in sorted((a.batch/'label').glob('*.png')):
        with Image.open(f) as im:
            if im.mode!='L':raise ValueError('Expected label L')
            label=np.array(im)
        with Image.open(a.batch/'rgb'/f.name) as im:
            if im.mode!='RGB':raise ValueError('Expected RGB')
            rgb=np.array(im)
        row=boundary_metrics(label,rgb);row['frame_id']=f.stem;rows.append(row)
    if not rows:raise ValueError('No frames')
    result={'frames':len(rows),'edge_evidence_pass_frames':sum(r.get('edge_evidence_pass',False) for r in rows),'rows':rows,'limitation':'Shadows, similar materials and bump texture can hide or add edges. Same-pass scalar ID and shared pixel remap establish correspondence; this checks visual boundary evidence.'}
    text=json.dumps(result,indent=2);(a.output or a.batch/'alignment.json').write_text(text);print(text)
    if a.require_edge_evidence and result['edge_evidence_pass_frames']!=len(rows):raise SystemExit('Boundary evidence insufficient; inspect lighting/materials/alignment')

if __name__=='__main__':main()
