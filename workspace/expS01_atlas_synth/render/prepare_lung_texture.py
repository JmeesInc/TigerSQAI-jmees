"""Prepare licensed donor textures for unchanged atlas lung geometry."""
import argparse,json,subprocess,zipfile
from pathlib import Path
import numpy as np,yaml
from PIL import Image
from .util import sha256

def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--archive',type=Path,required=True);p.add_argument('--atlas-mesh',type=Path,required=True);p.add_argument('--config',type=Path,default=Path('configs/lung_texture_transfer.yaml'));p.add_argument('--materials',type=Path,default=Path('configs/materials.yaml'));p.add_argument('--blender',default='blender');p.add_argument('--output',type=Path,required=True);a=p.parse_args()
 cfg=yaml.safe_load(a.config.read_text())
 if sha256(a.archive)!=cfg['archive_sha256']:raise ValueError('Donor archive SHA mismatch')
 out=a.output.resolve()
 if out.exists() and any(out.iterdir()):raise ValueError('Output must be empty')
 out.mkdir(parents=True,exist_ok=True);asset=out/'asset';asset.mkdir()
 with zipfile.ZipFile(a.archive) as z:
  for entry in z.infolist():
   if not (asset/entry.filename).resolve().is_relative_to(asset):raise ValueError('Unsafe archive path')
   z.extract(entry,asset)
 color=np.array(Image.open(asset/cfg['base_color']).convert('RGB'),dtype=np.float32)/255
 height=np.array(Image.open(asset/cfg['height']).convert('L'),dtype=np.float32)/255
 np.savez(out/'source_pixels.npz',color=color,height=height)
 (out/'job.json').write_text(json.dumps({'output':str(out),'asset':str(asset),'atlas_mesh':str(a.atlas_mesh.resolve()),'transfer':cfg}))
 with (out/'bake.log').open('w') as log:
  subprocess.run([a.blender,'-b','--factory-startup','--python-exit-code','1','--python',str(Path(__file__).with_name('bake_lung_texture.py').resolve()),'--',str(out/'job.json')],stdout=log,stderr=subprocess.STDOUT,check=True)
 manifest=json.loads((out/'manifest.json').read_text());digests={}
 for o in manifest['objects']:
  pixels=np.load(out/(o['key']+'_pixels.npy'))[::-1]
  for suffix,pix in [('color',pixels[:,:,:3]),('height',pixels[:,:,3])]:
   name=o['key']+'_'+suffix+'.png';Image.fromarray(pix).save(out/name);digests[name]=sha256(out/name)
 manifest.update(atlas_mesh_sha256=sha256(a.atlas_mesh),texture_sha256=digests)
 (out/'manifest.json').write_text(json.dumps(manifest,indent=2))
 m=yaml.safe_load(a.materials.read_text());m['lung_texture']={'enabled':True,'directory':str(out),'sha256':{n:sha256(out/n) for n in ['mapping.npz','manifest.json']},'source_url':cfg['source_url'],'license':cfg['license'],'author':cfg['author'],'status':cfg['status']}
 (out/'materials.yaml').write_text(yaml.safe_dump(m,sort_keys=False));print(out/'materials.yaml')
if __name__=='__main__':main()
