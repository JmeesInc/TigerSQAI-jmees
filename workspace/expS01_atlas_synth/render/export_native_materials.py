"""bpy: export opaque atlas surface materials plus exact triangle/slot mapping.
Run through python -m render.prepare_native_materials. Never saves the source blend.
"""
import json, sys
from pathlib import Path
import bpy
import numpy as np


def flatten(material, holder, area):
    holder.data.materials.clear(); holder.data.materials.append(material)
    bpy.context.view_layer.objects.active=holder
    area.type='NODE_EDITOR'; area.ui_type='ShaderNodeTree'
    area.spaces.active.shader_type='OBJECT'
    with bpy.context.temp_override(area=area):
        for _ in range(256):
            groups=[n for n in material.node_tree.nodes if n.type=='GROUP']
            if not groups: return
            for n in material.node_tree.nodes: n.select=False
            groups[0].select=True; material.node_tree.nodes.active=groups[0]
            if bpy.ops.node.group_ungroup()!={'FINISHED'}: raise RuntimeError('Ungroup failed')
    raise ValueError('Too many nested material groups')


def main():
    config_path, output = sys.argv[sys.argv.index('--')+1:]
    cfg=json.loads(Path(config_path).read_text()); output=Path(output)
    output.mkdir(parents=True,exist_ok=True)
    deps=bpy.context.evaluated_depsgraph_get()
    arrays={}; records=[]; sources={}
    # Geometry is evaluated before any material edits. Only manifest anatomy.
    for i,name in enumerate(n for names in cfg['objects'].values() for n in names):
        obj=bpy.data.objects[name]; ev=obj.evaluated_get(deps); mesh=ev.to_mesh()
        mesh.calc_loop_triangles(); key=f'o{i}'
        arrays[key+'_f']=np.array([t.vertices[:] for t in mesh.loop_triangles],np.int32)
        arrays[key+'_slots']=np.array([t.material_index for t in mesh.loop_triangles],np.int32)
        arrays[key+'_local']=np.array([v.co[:] for v in mesh.vertices],np.float32)
        mats=[m.name if m else None for m in mesh.materials]
        for m in mesh.materials:
            if m: sources[m.name]=m
        records.append({'name':name,'key':key,'materials':mats,'vertex_count':len(mesh.vertices),
                        'uv_layers':list(mesh.uv_layers.keys())})
        for j,uv in enumerate(mesh.uv_layers):
            arrays[key+f'_uv{j}']=np.array([[uv.data[loop].uv[:] for loop in t.loops] for t in mesh.loop_triangles],np.float32)
        ev.to_mesh_clear()
    holder=bpy.data.objects.new('MaterialExportHolder',bpy.data.meshes.new('MaterialExportHolder'))
    bpy.context.scene.collection.objects.link(holder); area=bpy.context.screen.areas[0]
    exported=set(); audit=[]
    for name, original in sources.items():
        mat=original.copy(); mat.name='AtlasSurface::'+name
        if not mat.use_nodes: raise ValueError('Non-node material requires explicit conversion: '+name)
        flatten(mat,holder,area); nodes=mat.node_tree.nodes; links=mat.node_tree.links
        before=[n.bl_idname for n in nodes]
        bsdfs=[n for n in nodes if n.type=='BSDF_PRINCIPLED']
        if len(bsdfs)!=1: raise ValueError(f'{name}: expected one opaque Principled surface, got {len(bsdfs)}')
        bs=bsdfs[0]
        # Opaque physical surface only: atlas viewer cutaway/transparency/comic
        # wrappers would change label/depth visibility or defeat scope lighting.
        for key,value in [('Alpha',1.),('Transmission Weight',0.),('Subsurface Weight',0.),('Emission Strength',0.)]:
            for link in list(bs.inputs[key].links): links.remove(link)
            bs.inputs[key].default_value=value
        for n in list(nodes):
            if n.type=='OUTPUT_MATERIAL':nodes.remove(n)
        out=nodes.new('ShaderNodeOutputMaterial');links.new(bs.outputs['BSDF'],out.inputs['Surface'])
        # Keep exactly the upstream appearance network feeding this BSDF.
        live={out}; pending=[out]
        while pending:
            node=pending.pop()
            for inp in node.inputs:
                for link in inp.links:
                    if link.from_node not in live:live.add(link.from_node);pending.append(link.from_node)
        for n in list(nodes):
            if n not in live:nodes.remove(n)
        for n in list(nodes):
            if n.type=='ATTRIBUTE':
                if n.attribute_name not in ['key_color','comic_shader']:raise ValueError('Unmapped appearance attribute '+n.attribute_name)
                # Select anatomy's base appearance, not atlas UI highlighting.
                for socket in n.outputs:
                    for link in list(socket.links):
                        target=link.to_socket; links.remove(link)
                        if hasattr(target,'default_value'):
                            try:target.default_value=0.
                            except (TypeError,ValueError):target.default_value=(0.,0.,0.,1.) if target.type=='RGBA' else (0.,0.,0.)
                nodes.remove(n)
                continue
            if n.type in ['TEX_IMAGE','TEX_NOISE','TEX_VORONOI','TEX_WAVE','TEX_COORD','NEW_GEOMETRY','BUMP']:
                raise ValueError(f'New spatial material node {n.type} in {name}; implement coordinate transfer before adopting')
        info=nodes.new('ShaderNodeObjectInfo'); aov=nodes.new('ShaderNodeOutputAOV');aov.aov_name='FineID'
        links.new(info.outputs['Object Index'],aov.inputs['Value'])
        mat.surface_render_method='DITHERED'
        mat.use_backface_culling=False;mat.use_transparency_overlap=False
        mat.use_fake_user=True;exported.add(mat)
        audit.append({'source_material':name,'export_material':mat.name,'flattened_node_types':before,
                      'surface_node_types':[n.bl_idname for n in nodes],
                      'adaptations':['opaque Principled branch','cutaway/comic/key-color UI bypassed','no displacement or emission','pass_index VALUE AOV added']})
    bpy.data.libraries.write(str(output/'surfaces.blend'),exported,fake_user=True,compress=True)
    np.savez_compressed(output/'mapping.npz',**arrays)
    (output/'manifest.json').write_text(json.dumps({'schema_version':1,'objects':records,'materials':audit,
      'source_sha256':cfg['sha256'],'source_url':cfg['source_url'],'license':cfg['license'],
      'attribution':cfg['attribution'],'blender_version':bpy.app.version_string},indent=2))
    print(f'Exported {len(records)} objects / {len(exported)} native opaque materials')

if __name__=='__main__':main()
