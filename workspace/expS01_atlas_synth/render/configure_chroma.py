"""Convert shared aggregate JSON into explicit initial albedo/vascular controls."""
import argparse,json
from pathlib import Path
import numpy as np
import yaml
from .util import sha256


def apply_chroma(cfg,rows):
    for r in rows:
        i=int(r['fine_id']);p=cfg['classes'][i];raw=np.array([r['chroma_r'],r['chroma_g'],r['chroma_b']],float)
        if np.any(raw<0) or not .99<raw.sum()<1.01:raise ValueError('Invalid RGB chroma')
        p['base_chroma_rgb']=(raw/raw.sum()).tolist();p['chroma_status']='measured_shared_aggregate_normalized_for_rounding';p['color_space_assumption']='sRGB-encoded measured RGB; verify extractor transfer function'
        p['target_saturation']=r['saturation'];p['target_vessel_pattern']=r['vessel_pattern'];p['chroma_n_frames']=r['n_frames']
        p['vascular_strength']=float(np.clip((r['vessel_pattern']-.02)/(.077-.02),0,1)*.55)
        p['vascular_status']='monotonic_initial_mapping_not_fitted_rendered_statistic';p['vessel_chroma_delta_r']=.08
        if i in [0,1,2,18,24]:p['vascular_strength']=0.
    return cfg


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--input',type=Path,required=True);p.add_argument('--materials',type=Path,default=Path('configs/materials.yaml'));p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    cfg=apply_chroma(yaml.safe_load(a.materials.read_text()),json.loads(a.input.read_text()));cfg['chroma_source_sha256']=sha256(a.input)
    a.output.write_text(yaml.safe_dump(cfg,sort_keys=False))

if __name__=='__main__':main()
