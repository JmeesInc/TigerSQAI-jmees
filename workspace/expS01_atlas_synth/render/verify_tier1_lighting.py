"""Run with Blender --background --python; isolated inverse-square integration test."""
import json
import sys
from pathlib import Path
import bpy
import numpy as np

bpy.ops.wm.read_factory_settings(use_empty=True)
s=bpy.context.scene;s.render.engine='BLENDER_EEVEE_NEXT';s.eevee.taa_render_samples=1
s.render.resolution_x=64;s.render.resolution_y=64;s.render.resolution_percentage=100;s.render.filter_size=.01
s.world=bpy.data.worlds.new('Dark');s.world.use_nodes=True;s.world.node_tree.nodes['Background'].inputs['Strength'].default_value=0
cam=bpy.data.cameras.new('cam');cam.clip_end=1000;ob=bpy.data.objects.new('cam',cam);s.collection.objects.link(ob);s.camera=ob
light=bpy.data.lights.new('point','POINT');light.energy=60000;light.shadow_soft_size=0;light.use_custom_distance=True;light.cutoff_distance=100000.;lo=bpy.data.objects.new('point',light);s.collection.objects.link(lo)
bpy.ops.mesh.primitive_plane_add(size=400,location=(0,0,-50));plane=bpy.context.object
m=bpy.data.materials.new('Lambert');m.use_nodes=True;n=m.node_tree.nodes;n.clear();out=n.new('ShaderNodeOutputMaterial');diff=n.new('ShaderNodeBsdfDiffuse');diff.inputs['Color'].default_value=(.5,.5,.5,1);m.node_tree.links.new(diff.outputs[0],out.inputs['Surface']);plane.data.materials.append(m)
outdir=Path(sys.argv[sys.argv.index('--')+1]);outdir.mkdir(parents=True,exist_ok=True)
s.render.image_settings.file_format='OPEN_EXR_MULTILAYER';s.render.image_settings.color_depth='32'
for d in [50,100]:
 plane.location.z=-d;s.render.filepath=str(outdir/f'{d}.exr');bpy.ops.render.render(write_still=True)
print('Read linear EXRs with host OpenEXR; expected center near/far ratio = 4.')
