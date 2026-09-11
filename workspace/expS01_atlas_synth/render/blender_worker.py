"""Persistent bpy worker. stdin NDJSON -> raw Object Index/Depth multilayer EXR."""
import json
import sys
import time
import traceback
from pathlib import Path
import bpy
import numpy as np
from mathutils import Matrix,Vector
from mathutils.bvhtree import BVHTree

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from render.util import load_meshes,atomic_json


MATERIALS={}

def make_object(item,collection):
    mesh=bpy.data.meshes.new(item['name'])
    mesh.from_pydata(item['v'].tolist(),[],item['f'].tolist())
    mesh.update()
    obj=bpy.data.objects.new(item['name'],mesh)
    collection.objects.link(obj)
    obj.pass_index=int(item['fine_id'])
    if MATERIALS:obj.data.materials.append(MATERIALS[obj.pass_index])
    return obj


def tree(objects):
    vertices=[];faces=[]
    for item in objects:
        offset=len(vertices)
        vertices.extend(item['v'].tolist())
        faces.extend((item['f']+offset).tolist())
    return BVHTree.FromPolygons(vertices,faces,all_triangles=True)


def collision(bvh,camera,cfg):
    port=Vector(camera['port_mm']); tip=Vector(camera['tip_mm'])
    direction=(tip-port).normalized(); distance=(tip-port).length
    radius=cfg['shaft_radius_mm']
    right=Vector(camera['camera_to_world_blender_mm'][0][:3])
    right=direction.cross(Vector((0,0,1)))
    if right.length<0.01:right=direction.cross(Vector((0,1,0)))
    right.normalize();up=direction.cross(right)
    if cfg['enforce_shaft_collision']:
        for offset in [Vector((0,0,0)),radius*right,-radius*right,radius*up,-radius*up]:
            hit=bvh.ray_cast(port+offset,direction,distance)
            if hit[0] is not None:return 'shaft intersects anatomy or cover'
        # Conservative clearance along a finite-radius shaft. Distance to a surface
        # is 1-Lipschitz: radius + half step bounds unsampled intervals.
        steps=max(1,int(np.ceil(distance/cfg['shaft_clearance_step_mm'])))
        step=distance/steps
        for value in np.linspace(0,distance,steps+1):
            nearest=bvh.find_nearest(port+float(value)*direction)
            if nearest[0] is not None and nearest[3]<radius+step/2:
                return 'shaft radial clearance'
    nearest=bvh.find_nearest(tip)
    if nearest[0] is not None and nearest[3]<cfg['tip_clearance_mm']:
        return 'tip clearance'
    if cfg['enforce_target_line_of_sight']:
        delta=Vector(camera['target_mm'])-tip
        hit=bvh.ray_cast(tip,delta.normalized(),max(0.,delta.length-2))
        if hit[0] is not None:return 'target occluded'
    return None


def setup(config,base):
    bpy.ops.wm.read_factory_settings(use_empty=True)
    scene=bpy.context.scene
    scene.unit_settings.system='METRIC'
    scene.unit_settings.scale_length=0.001
    scene.render.engine='CYCLES'
    scene.cycles.samples=int(config['samples'])
    scene.cycles.use_denoising=False
    scene.cycles.max_bounces=1
    scene.cycles.use_adaptive_sampling=False
    scene.render.film_transparent=True
    scene.render.filter_size=float(config['filter_size'])
    if hasattr(scene.cycles,'filter_width'):scene.cycles.filter_width=float(config['filter_size'])
    if hasattr(scene.render,'use_motion_blur'):scene.render.use_motion_blur=False
    device=config['device'].upper()
    if device!='CPU' and (not config.get('emit_rgb',False) or config.get('materials',{}).get('engine')=='CYCLES'):
        prefs=bpy.context.preferences.addons['cycles'].preferences
        prefs.compute_device_type=device
        prefs.get_devices()
        active=False
        for d in prefs.devices:
            d.use=d.type==device
            active|=d.use
        if not active:raise RuntimeError(f'No {device} device available (no silent CPU fallback)')
        scene.cycles.device='GPU'
    else:scene.cycles.device='CPU'
    layer=scene.view_layers[0]
    layer.name='Labels'
    layer.use_pass_object_index=True
    layer.use_pass_z=True
    try:
        scene.render.image_settings.file_format='OPEN_EXR_MULTILAYER'
    except TypeError:
        scene.render.image_settings.media_type='MULTI_LAYER_IMAGE'
        scene.render.image_settings.file_format='OPEN_EXR_MULTILAYER'
        scene.render.image_settings.use_exr_interleave=True
    scene.render.image_settings.color_depth='32'
    scene.render.image_settings.exr_codec='ZIP'
    scene.render.resolution_percentage=100
    scene.render.use_compositing=False
    scene.render.use_sequencer=False
    scene.render.threads_mode='FIXED'
    scene.render.threads=int(config.get('threads_per_worker',2))
    global MATERIALS
    MATERIALS={}
    if config.get('emit_rgb',False):
        from render.materials import configure
        MATERIALS,_=configure(scene,config['materials'])
    for item in base:make_object(item,scene.collection)
    cam=bpy.data.cameras.new('RigidScope')
    cam.type='PERSP';cam.sensor_fit='HORIZONTAL';cam.sensor_width=config['scope']['sensor_width_mm']
    cam.clip_start,cam.clip_end=config['clip_mm']
    cam.dof.use_dof=False
    obj=bpy.data.objects.new('RigidScope',cam)
    scene.collection.objects.link(obj);scene.camera=obj
    return scene,obj


def main():
    config_path,base_path=sys.argv[sys.argv.index('--')+1:]
    config=json.loads(Path(config_path).read_text())
    base=load_meshes(base_path)
    scene,cam=setup(config,base)
    generated=[]
    bvh=None;geometry_key=None
    for line in sys.stdin:
        job=json.loads(line)
        if job.get('stop'):break
        start=time.perf_counter()
        try:
            for obj in generated:
                mesh=obj.data
                bpy.data.objects.remove(obj,do_unlink=True)
                if mesh.users==0:bpy.data.meshes.remove(mesh)
            generated=[]
            extra=load_meshes(job['geometry'])
            if bvh is None or job.get('geometry_key')!=geometry_key:
                bvh=tree(base+[o for o in extra if o['fine_id']!=1])
                geometry_key=job.get('geometry_key')
            reason=collision(bvh,job['camera'],config['scope'])
            if reason:
                atomic_json(job['response'],{'accepted_geometry':False,'reason':reason})
                continue
            for instrument in [o for o in extra if o['fine_id']==1]:
                # Each instrument record has centerline entry and tip; avoid traversing organs.
                delta=Vector(instrument['tip_mm'])-Vector(instrument['port_mm'])
                hit=bvh.ray_cast(Vector(instrument['port_mm']),delta.normalized(),max(0.,delta.length-2.))
                if hit[0] is not None:
                    raise ValueError('instrument crosses anatomy')
            generated=[make_object(o,scene.collection) for o in extra]
            camera=job['camera'];w,h=camera['distortion']['render_size']
            scene.render.resolution_x=w;scene.render.resolution_y=h
            cam.matrix_world=Matrix(camera['camera_to_world_blender_mm'])
            cam.data.lens=camera['K'][0][0]*cam.data.sensor_width/w
            k=camera['distortion']['render_K'];aspect=k[0][0]/k[1][1]
            scene.render.pixel_aspect_x=max(1.,1./aspect)
            scene.render.pixel_aspect_y=max(1.,aspect)
            cam.data.shift_x=((w-1)/2-k[0][2])/w
            cam.data.shift_y=(k[1][2]-(h-1)/2)*aspect/w
            if config.get('emit_rgb',False):scene.objects['ScopePoint'].location=cam.location
            scene.render.filepath=job['raw_exr']
            bpy.context.view_layer.update()
            bpy.ops.render.render(write_still=True)
            atomic_json(job['response'],{'accepted_geometry':True,'render_seconds':time.perf_counter()-start,
                        'blender_version':bpy.app.version_string,'filter_size':scene.render.filter_size,
                        'pass_method':'pass_index -> scalar FineID AOV' if config.get('emit_rgb',False) else 'Object Index / pass_index; no RGB decoding','engine':scene.render.engine})
        except Exception as error:
            atomic_json(job['response'],{'accepted_geometry':False,'reason':str(error),
                                        'traceback':traceback.format_exc()})


if __name__=='__main__':main()
