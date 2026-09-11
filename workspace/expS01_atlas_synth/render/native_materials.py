"""Assign native atlas materials using exact vertex-index triangle identity."""
import json
from pathlib import Path
import numpy as np
from .util import sha256


def triangle_rows(source_faces, target_faces):
    lookup={tuple(f):i for i,f in enumerate(source_faces)}
    if len(lookup)!=len(source_faces):raise ValueError('Ambiguous duplicate atlas triangles')
    try:return np.array([lookup[tuple(f)] for f in target_faces],dtype=np.int64)
    except KeyError as e:raise ValueError('Topology changed; cannot transfer native atlas materials') from e


class NativeMaterials:
    def __init__(self,config):
        import bpy
        folder=Path(config['directory'])
        for name,digest in config['sha256'].items():
            if sha256(folder/name)!=digest:raise ValueError('Native material library SHA mismatch: '+name)
        self.manifest=json.loads((folder/'manifest.json').read_text())
        self.records={r['name']:r for r in self.manifest['objects']}
        with np.load(folder/'mapping.npz',allow_pickle=False) as z:self.arrays={k:z[k].copy() for k in z.files}
        with bpy.data.libraries.load(str(folder/'surfaces.blend'),link=False) as (src,dst):
            dst.materials=[name for name in src.materials if name.startswith('AtlasSurface::')]
        self.materials={m.name:m for m in dst.materials}
        self.assigned=[]

    def apply(self,obj,item):
        r=self.records.get(item['name'])
        if r is None:return False
        if len(item['v'])!=r['vertex_count']:raise ValueError('Vertex count changed: '+item['name'])
        key=r['key'];rows=triangle_rows(self.arrays[key+'_f'],item['f'])
        obj.data.materials.clear()
        for name in r['materials']:
            if name is None:raise ValueError('Empty native material slot')
            obj.data.materials.append(self.materials['AtlasSurface::'+name])
        slots=self.arrays[key+'_slots'][rows];obj.data.polygons.foreach_set('material_index',slots)
        for j,name in enumerate(r['uv_layers']):
            uv=obj.data.uv_layers.new(name=name)
            uv.data.foreach_set('uv',self.arrays[key+f'_uv{j}'][rows].ravel())
        # Preserve the current renderer's normal interpolation for controlled A/B.
        self.assigned.append({'object':item['name'],'faces':len(rows),'slots_used':np.unique(slots).tolist()})
        return True
