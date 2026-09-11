"""bpy procedural materials; scalar pass_index AOV; bump never alters silhouette."""
import colorsys
import bpy
import numpy as np
from render.chroma import material_linear_color
from render.tissue_patterns import vascular_tile

VASCULAR_IMAGE=None


def make_material(class_id,cfg):
    p=cfg['classes'].get(str(class_id),cfg['classes'].get(class_id))
    if p is None:raise ValueError(f'Missing class material {class_id}')
    mat=bpy.data.materials.new(f'Tier1_{class_id}');mat.use_nodes=True;mat['smooth_shading']=p.get('smooth_shading',False)
    n=mat.node_tree.nodes;l=mat.node_tree.links;n.clear()
    out=n.new('ShaderNodeOutputMaterial');bs=n.new('ShaderNodeBsdfPrincipled')
    bs.inputs['Roughness'].default_value=p['roughness'];bs.inputs['Specular IOR Level'].default_value=p['specular']
    bs.inputs['Metallic'].default_value=0.;bs.inputs['Alpha'].default_value=1.
    geo=n.new('ShaderNodeNewGeometry');noise=n.new('ShaderNodeTexNoise');noise.noise_dimensions='3D'
    noise.inputs['Scale'].default_value=p['noise_scale'];noise.inputs['Detail'].default_value=2.
    l.new(geo.outputs['Position'],noise.inputs['Vector'])
    ramp=n.new('ShaderNodeValToRGB');color=material_linear_color(p) if 'base_chroma_rgb' in p else colorsys.hsv_to_rgb(p['base_hue'],p['saturation'],p['value']);amp=p['texture_color_strength']
    for e,factor in zip(ramp.color_ramp.elements,[1-amp,1+amp]):e.color=tuple(min(1,max(0,x*factor)) for x in color)+(1.,)
    l.new(noise.outputs['Fac'],ramp.inputs['Fac']);l.new(ramp.outputs['Color'],bs.inputs['Base Color'])
    tissue_height=None
    if cfg.get('tissue_patterns',{}).get('enabled',False):
        tissue_height=add_tissue_nodes(n,l,geo,noise,ramp,bs,p,cfg['tissue_patterns'])
    bump=n.new('ShaderNodeBump');bump.inputs['Strength'].default_value=p['bump_strength'];bump.inputs['Distance'].default_value=p['bump_distance_mm']
    if tissue_height is not None:
        blend=n.new('ShaderNodeMixRGB');blend.inputs[0].default_value=p.get('pattern_strength',.3);l.new(noise.outputs['Fac'],blend.inputs[1]);l.new(tissue_height,blend.inputs[2]);l.new(blend.outputs[0],bump.inputs['Height'])
    else:l.new(noise.outputs['Fac'],bump.inputs['Height'])
    l.new(bump.outputs['Normal'],bs.inputs['Normal']);l.new(bs.outputs['BSDF'],out.inputs['Surface'])
    info=n.new('ShaderNodeObjectInfo');aov=n.new('ShaderNodeOutputAOV');aov.aov_name='FineID';aov.name='FineID'
    l.new(info.outputs['Object Index'],aov.inputs['Value'])
    return mat


def add_tissue_nodes(n,l,geo,noise,ramp,bs,p,t):
    def math(op,left,right=None):
        node=n.new('ShaderNodeMath');node.operation=op
        if isinstance(left,(int,float)):node.inputs[0].default_value=left
        else:l.new(left,node.inputs[0])
        if right is not None:
            if isinstance(right,(int,float)):node.inputs[1].default_value=right
            else:l.new(right,node.inputs[1])
        return node.outputs[0]
    pattern=p.get('pattern','membrane')
    if pattern in ['lobules','lung_septa','granular']:
        vor=n.new('ShaderNodeTexVoronoi');vor.feature='F1' if pattern=='granular' else 'DISTANCE_TO_EDGE';vor.inputs['Scale'].default_value=t['lobule_scale_per_mm']*(1.6 if pattern=='lung_septa' else 1)
        l.new(geo.outputs['Position'],vor.inputs['Vector']);field=math('LESS_THAN',vor.outputs['Distance'],.06)
    elif pattern in ['rings','axial']:
        attr=n.new('ShaderNodeAttribute');attr.attribute_name='tissue_axial_mm' if pattern=='rings' else 'tissue_transverse_mm'
        spacing=t['ring_spacing_mm'] if pattern=='rings' else t['axial_spacing_mm']
        phase=math('ADD',math('MULTIPLY',attr.outputs['Fac'],6.283185/spacing),math('MULTIPLY',noise.outputs['Fac'],1.4))
        field=math('MULTIPLY',math('SINE',phase),.5);field=math('ADD',field,.5)
    else:field=noise.outputs['Fac']
    factor=math('SUBTRACT',1.,math('MULTIPLY',field,p.get('pattern_strength',.3)))
    color=n.new('ShaderNodeMixRGB');color.blend_type='MULTIPLY';color.inputs[0].default_value=1.;l.new(ramp.outputs['Color'],color.inputs[1]);l.new(factor,color.inputs[2])
    if p.get('vascular_strength',0)>0:
        vector=n.new('ShaderNodeVectorMath');vector.operation='SCALE';vector.inputs['Scale'].default_value=1/t['vascular_tile_mm'];l.new(geo.outputs['Position'],vector.inputs[0])
        tex=n.new('ShaderNodeTexImage');tex.image=VASCULAR_IMAGE;tex.projection='BOX';tex.projection_blend=.25;tex.extension='REPEAT';l.new(vector.outputs['Vector'],tex.inputs['Vector'])
        mix=n.new('ShaderNodeMixRGB');mix.blend_type='MIX';l.new(math('MULTIPLY',tex.outputs['Color'],p['vascular_strength']),mix.inputs[0]);l.new(color.outputs[0],mix.inputs[1]);vessel=dict(p);ch=np.array(p['base_chroma_rgb']);delta=min(p.get('vessel_chroma_delta_r',.08),1-ch[0]);ch[1:]*=(1-ch[0]-delta)/(1-ch[0]);ch[0]+=delta;vessel['base_chroma_rgb']=ch.tolist();vessel['base_value_srgb']=p.get('base_value_srgb',.65)*.8;mix.inputs[2].default_value=tuple(material_linear_color(vessel))+(1.,)
        l.new(mix.outputs[0],bs.inputs['Base Color'])
    else:l.new(color.outputs[0],bs.inputs['Base Color'])
    return field


def configure(scene,config):
    if config['samples']!=1:raise ValueError('ID AOV requires exactly one EEVEE sample; refuse blended ID pixels')
    global VASCULAR_IMAGE
    if config.get('tissue_patterns',{}).get('enabled',False):
        t=config['tissue_patterns'];a=vascular_tile(t['vascular_tile_size'],t['vascular_seed']);rgba=np.stack([a,a,a,np.ones_like(a)],axis=-1)
        VASCULAR_IMAGE=bpy.data.images.new('Procedural branching vascular mask',width=a.shape[1],height=a.shape[0],float_buffer=True)
        VASCULAR_IMAGE.colorspace_settings.name='Non-Color';VASCULAR_IMAGE.pixels.foreach_set(rgba.ravel());VASCULAR_IMAGE.update()
    scene.render.engine=config.get('engine','BLENDER_EEVEE_NEXT')
    if scene.render.engine=='BLENDER_EEVEE_NEXT':scene.eevee.taa_render_samples=1
    scene.world=bpy.data.worlds.new('DarkThorax');scene.world.use_nodes=True
    scene.world.node_tree.nodes['Background'].inputs['Strength'].default_value=config['light']['environment_strength']
    a=scene.view_layers[0].aovs.add();a.name='FineID';a.type='VALUE'
    light=bpy.data.lights.new('ScopePoint','POINT');light.energy=config['light']['energy'];light.shadow_soft_size=config['light']['radius_mm'];light.use_shadow=True
    light.use_custom_distance=True;light.cutoff_distance=config['light']['cutoff_distance_mm']
    # Point light has inverse-square attenuation inherently; no constant-falloff nodes.
    obj=bpy.data.objects.new('ScopePoint',light);scene.collection.objects.link(obj)
    mats={i:make_material(i,config) for i in range(31)}
    return mats,obj
