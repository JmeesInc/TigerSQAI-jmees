"""bpy procedural materials; scalar pass_index AOV; bump never alters silhouette."""
import colorsys
import bpy


def make_material(class_id,cfg):
    p=cfg['classes'].get(str(class_id),cfg['classes'].get(class_id))
    if p is None:raise ValueError(f'Missing class material {class_id}')
    mat=bpy.data.materials.new(f'Tier1_{class_id}');mat.use_nodes=True
    n=mat.node_tree.nodes;l=mat.node_tree.links;n.clear()
    out=n.new('ShaderNodeOutputMaterial');bs=n.new('ShaderNodeBsdfPrincipled')
    bs.inputs['Roughness'].default_value=p['roughness'];bs.inputs['Specular IOR Level'].default_value=p['specular']
    bs.inputs['Metallic'].default_value=0.;bs.inputs['Alpha'].default_value=1.
    geo=n.new('ShaderNodeNewGeometry');noise=n.new('ShaderNodeTexNoise');noise.noise_dimensions='3D'
    noise.inputs['Scale'].default_value=p['noise_scale'];noise.inputs['Detail'].default_value=2.
    l.new(geo.outputs['Position'],noise.inputs['Vector'])
    ramp=n.new('ShaderNodeValToRGB');color=colorsys.hsv_to_rgb(p['base_hue'],p['saturation'],p['value']);amp=p['texture_color_strength']
    for e,factor in zip(ramp.color_ramp.elements,[1-amp,1+amp]):e.color=tuple(min(1,max(0,x*factor)) for x in color)+(1.,)
    l.new(noise.outputs['Fac'],ramp.inputs['Fac']);l.new(ramp.outputs['Color'],bs.inputs['Base Color'])
    bump=n.new('ShaderNodeBump');bump.inputs['Strength'].default_value=p['bump_strength'];bump.inputs['Distance'].default_value=p['bump_distance_mm']
    l.new(noise.outputs['Fac'],bump.inputs['Height']);l.new(bump.outputs['Normal'],bs.inputs['Normal']);l.new(bs.outputs['BSDF'],out.inputs['Surface'])
    info=n.new('ShaderNodeObjectInfo');aov=n.new('ShaderNodeOutputAOV');aov.aov_name='FineID';aov.name='FineID'
    l.new(info.outputs['Object Index'],aov.inputs['Value'])
    return mat


def configure(scene,config):
    if config['samples']!=1:raise ValueError('ID AOV requires exactly one EEVEE sample; refuse blended ID pixels')
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
