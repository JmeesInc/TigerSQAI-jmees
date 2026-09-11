"""Baked external appearance on original atlas triangles, scalar ID unchanged."""
import json
from pathlib import Path
import numpy as np
from .util import sha256
from .native_materials import triangle_rows

class LungMaterials:
 def __init__(self,config):
  import bpy
  root=Path(config['directory'])
  for name,digest in config['sha256'].items():
   if sha256(root/name)!=digest:raise ValueError('Lung transfer SHA mismatch: '+name)
  manifest=json.loads((root/'manifest.json').read_text());cfg=manifest['transfer']
  for name,digest in manifest['texture_sha256'].items():
   if sha256(root/name)!=digest:raise ValueError('Lung texture SHA mismatch: '+name)
  self.records={r['name']:r for r in manifest['objects']};self.mats={}
  with np.load(root/'mapping.npz') as z:self.arrays={k:z[k].copy() for k in z.files}
  for name,r in self.records.items():
   mat=bpy.data.materials.new('Donor appearance: '+name);mat.use_nodes=True;n=mat.node_tree.nodes;l=mat.node_tree.links;n.clear()
   out=n.new('ShaderNodeOutputMaterial');bs=n.new('ShaderNodeBsdfPrincipled');bs.inputs['Roughness'].default_value=cfg['roughness'];bs.inputs['Specular IOR Level'].default_value=cfg['specular'];bs.inputs['Metallic'].default_value=0.;bs.inputs['Alpha'].default_value=1.
   uv=n.new('ShaderNodeUVMap');uv.uv_map='DonorBakedUV'
   for suffix,space in [('color','sRGB'),('height','Non-Color')]:
    tex=n.new('ShaderNodeTexImage');tex.image=bpy.data.images.load(str(root/(r['key']+'_'+suffix+'.png')),check_existing=True);tex.image.colorspace_settings.name=space;tex.extension='EXTEND';l.new(uv.outputs['UV'],tex.inputs['Vector'])
    if suffix=='color':l.new(tex.outputs['Color'],bs.inputs['Base Color'])
    else:
     bump=n.new('ShaderNodeBump');bump.inputs['Strength'].default_value=cfg['bump_strength'];bump.inputs['Distance'].default_value=cfg['bump_distance_mm'];l.new(tex.outputs['Color'],bump.inputs['Height']);l.new(bump.outputs['Normal'],bs.inputs['Normal'])
   l.new(bs.outputs['BSDF'],out.inputs['Surface']);info=n.new('ShaderNodeObjectInfo');aov=n.new('ShaderNodeOutputAOV');aov.aov_name='FineID';l.new(info.outputs['Object Index'],aov.inputs['Value']);self.mats[name]=mat
 def apply(self,obj,item):
  if item['fine_id']!=18:return False
  if item['name'] not in self.records:raise ValueError('Lung transfer has no record: '+item['name'])
  r=self.records[item['name']];key=r['key']
  if len(item['v'])!=r['vertices']:raise ValueError('Lung vertex count mismatch')
  rows=triangle_rows(self.arrays[key+'_f'],item['f']);uv=obj.data.uv_layers.new(name='DonorBakedUV');uv.data.foreach_set('uv',self.arrays[key+'_uv'][rows].ravel())
  obj.data.materials.clear();obj.data.materials.append(self.mats[item['name']]);obj.data.polygons.foreach_set('material_index',np.zeros(len(item['f']),dtype=np.int32));return True
