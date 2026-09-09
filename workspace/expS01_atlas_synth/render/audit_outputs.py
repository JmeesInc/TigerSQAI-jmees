"""Validate saved PNG/EXR/JSON triples without challenge data."""
import argparse
import json
from pathlib import Path
import numpy as np
import OpenEXR
from PIL import Image


def audit(output):
    files=sorted((output/'label').glob('*.png'))
    if not files:raise ValueError('No output labels')
    resolved=output/'resolved_configs.json'
    cfg=json.loads(resolved.read_text()) if resolved.exists() else {}
    no_tools=cfg.get('dissection',{}).get('instruments',{}).get('enabled') is False
    union=set();stages={};instrument_counts=set()
    for file in files:
        meta=json.loads((output/'meta'/f'{file.stem}.json').read_text())
        image=Image.open(file);assert image.mode=='L'
        labels=np.array(image);assert labels.dtype==np.uint8 and labels.max()<=30
        assert list(image.size)==meta['resolution']
        counts=np.bincount(labels.ravel(),minlength=31)
        if no_tools:assert counts[1]==0 and meta['instrument_count']==0, 'Disabled instruments appeared'
        assert {str(i):int(c) for i,c in enumerate(counts)}==meta['class_pixel_counts']
        assert np.flatnonzero(counts[1:]).__add__(1).tolist()==meta['visible_class_ids']
        with OpenEXR.File(str(output/'depth'/f'{file.stem}.exr'),separate_channels=True) as exr:
            assert list(exr.channels())==['Z']
            z=exr.channels()['Z'].pixels
        assert z.shape==labels.shape and z.dtype==np.float32
        assert np.isfinite(z).all() and np.all(z>=0)
        assert np.all(z[labels>0]>0)
        c=meta['camera'];s=np.array(c['shaft_axis_ras']);v=np.array(c['optical_axis_ras'])
        np.testing.assert_allclose(s@v,np.cos(np.deg2rad(c['oblique_angle_deg'])),atol=1e-6)
        np.testing.assert_allclose(np.array(c['port_mm'])+c['insertion_mm']*s,c['tip_mm'],atol=1e-5)
        np.testing.assert_allclose(np.array(c['tip_mm'])+c['working_distance_mm']*v,c['target_mm'],atol=1e-5)
        np.testing.assert_allclose(np.array(c['world_to_camera_cv_mm'])@np.array(c['camera_to_world_cv_mm']),np.eye(4),atol=1e-6)
        if c.get('window_coupling'):
            coupled=c['window_coupling'];w,h=meta['resolution']
            occupancy=2*coupled['window']['effective_radius_mm']/(c['working_distance_mm']*min(w/c['K'][0][0],h/c['K'][1][1]))
            np.testing.assert_allclose(occupancy,coupled['requested_short_side_occupancy'],atol=1e-6)
        assert meta['visible_stations'] is None and meta['station_nomenclature']=='要確認'
        union.update(meta['visible_class_ids']);phase=meta['dissection']['phase']
        stages[phase]=stages.get(phase,0)+1;instrument_counts.add(meta['instrument_count'])
    return {'frames':len(files),'visible_classes':sorted(union),'phase_counts':stages,
            'instrument_counts':sorted(instrument_counts),'checks':'PNG L/uint8, EXR Z/float32/mm, metadata counts, rigid scope geometry'}


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('output',type=Path)
    args=parser.parse_args();print(json.dumps(audit(args.output),indent=2))


if __name__=='__main__':main()
