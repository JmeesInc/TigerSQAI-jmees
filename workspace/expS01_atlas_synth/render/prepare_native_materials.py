"""Build an optional Z-Anatomy appearance library without changing geometry."""
import argparse,json,subprocess
from pathlib import Path
import yaml
from .util import sha256

def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--atlas-config',type=Path,default=Path('configs/atlas.yaml'));p.add_argument('--atlas',type=Path);p.add_argument('--blender',default='blender');p.add_argument('--output',type=Path,required=True);p.add_argument('--materials',type=Path,default=Path('configs/materials.yaml'));p.add_argument('--output-config',type=Path,required=True);a=p.parse_args()
 cfg=yaml.safe_load(a.atlas_config.read_text());blend=(a.atlas or Path(cfg['blend'])).resolve()
 if sha256(blend)!=cfg['sha256']:raise ValueError('Atlas SHA mismatch')
 out=a.output.resolve()
 if out.exists() and any(out.iterdir()):raise ValueError('Appearance output must be empty')
 out.mkdir(parents=True,exist_ok=True);(out/'atlas_config.json').write_text(json.dumps(cfg))
 with (out/'export.log').open('w') as log:
  subprocess.run([a.blender,'-b',str(blend),'--disable-autoexec','--python-exit-code','1','--python',str(Path(__file__).with_name('export_native_materials.py').resolve()),'--',str(out/'atlas_config.json'),str(out)],stdout=log,stderr=subprocess.STDOUT,check=True)
 m=yaml.safe_load(a.materials.read_text());m['native_atlas']={'enabled':True,'directory':str(out),'sha256':{name:sha256(out/name) for name in ['surfaces.blend','mapping.npz','manifest.json']},'generated_surface_policy':'existing procedural materials; no native counterpart'}
 a.output_config.parent.mkdir(parents=True,exist_ok=True);a.output_config.write_text(yaml.safe_dump(m,sort_keys=False));print(a.output_config)
if __name__=='__main__':main()
