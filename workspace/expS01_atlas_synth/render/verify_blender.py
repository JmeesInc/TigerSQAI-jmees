"""Real Blender pass test: known planar distances, overlapping IDs, no AA labels."""
import argparse
import json
from pathlib import Path
import subprocess
import tempfile
import time
import numpy as np
import yaml
from .util import save_meshes,atomic_json
from .output import read_passes,remap,write_outputs
from .camera import distortion_grid


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--blender',default='blender');args=p.parse_args()
    root=Path(__file__).resolve().parents[1]
    cfg=yaml.safe_load((root/'configs/camera_prior.yaml').read_text())
    cfg['resolution']=[128,96];cfg['device']='CPU';cfg['distortion']['enabled']=False
    cfg['scope']['enforce_shaft_collision']=True
    pose={'K':[[100,0,73.5],[0,80,42.5],[0,0,1]],'port_mm':[0,0,5],'tip_mm':[0,0,0],
          'target_mm':[0,0,-50],'camera_to_world_blender_mm':np.eye(4).tolist()}
    grid=distortion_grid(np.random.default_rng(1),pose,cfg)
    def plane(name,index,size,z):
        return {'name':name,'fine_id':index,'v':np.array([[-size,-size,z],[size,-size,z],[size,size,z],[-size,size,z]]),
                'f':np.array([[0,1,2],[0,2,3]])}
    with tempfile.TemporaryDirectory(prefix='tier0_pass_test_') as directory:
        work=Path(directory);atomic_json(work/'config.json',cfg)
        save_meshes(work/'base.npz',[plane('rear',3,100,-50),plane('front',19,8,-40)])
        save_meshes(work/'extra.npz',[])
        job={'geometry':str(work/'extra.npz'),'camera':pose,'response':str(work/'response.json'),'raw_exr':str(work/'raw.exr')}
        cmd=[args.blender,'-b','--factory-startup','--disable-autoexec','--python-exit-code','1','--python',str(root/'render/blender_worker.py'),'--',str(work/'config.json'),str(work/'base.npz')]
        result=subprocess.run(cmd,input=json.dumps(job)+'\n'+json.dumps({'stop':True})+'\n',text=True,capture_output=True,timeout=180)
        if result.returncode:raise RuntimeError(result.stdout+result.stderr)
        response=json.loads((work/'response.json').read_text())
        if not response['accepted_geometry']:raise RuntimeError(response)
        index,distance=read_passes(work/'raw.exr')
        assert set(np.unique(index))=={3,19},np.unique(index)
        labels,z=remap(index,distance,pose,grid)
        assert labels[48,64]==19 and labels[5,5]==3
        # Ignore boundary pixels; depth and Object Index must agree and axial Z be flat.
        np.testing.assert_allclose(z[10:20,10:20],50,atol=.08)
        np.testing.assert_allclose(z[38:46,70:78],40,atol=.08)
        yy,xx=np.where(labels==19)
        assert abs(xx.mean()-73.5)<.6 and abs(yy.mean()-42.5)<.6
        assert abs((xx.max()-xx.min()+1)-40)<=1 and abs((yy.max()-yy.min()+1)-32)<=1
        (work/'label').mkdir();(work/'depth').mkdir()
        write_outputs(work,'fixture',labels,z)
        from PIL import Image
        image=Image.open(work/'label/fixture.png')
        assert image.mode=='L' and np.array(image).dtype==np.uint8
        print('PASS: actual Blender Object Index -> uint8 L PNG; overlap IDs; metric axial depth',response)


if __name__=='__main__':main()
