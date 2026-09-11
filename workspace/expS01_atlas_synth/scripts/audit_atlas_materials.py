import bpy,json,sys
from pathlib import Path
names=['Inferior lobe of right lung','Oesophagus','Trachea','Azygos vein','Thoracic aorta','Right inferior pulmonary vein']
r={'counts':{k:len(getattr(bpy.data,k)) for k in ['materials','images','textures']},'objects':[]}
for name in names:
 o=bpy.data.objects.get(name)
 if o is None:continue
 mats=[]
 for slot in o.material_slots:
  m=slot.material
  if not m:continue
  mats.append({'name':m.name,'diffuse_color':list(m.diffuse_color),'use_nodes':m.use_nodes,'nodes':[n.bl_idname for n in m.node_tree.nodes] if m.node_tree else [],'image_nodes':[{'image':n.image.name,'packed':bool(n.image.packed_file)} for n in m.node_tree.nodes if n.bl_idname=='ShaderNodeTexImage' and n.image] if m.node_tree else []})
 r['objects'].append({'name':name,'uv_layers':list(o.data.uv_layers.keys()) if o.type=='MESH' else [],'materials':mats})
Path(sys.argv[sys.argv.index('--')+1]).write_text(json.dumps(r,indent=2));print(json.dumps(r))
