"""bpy: bake donor UV samples to per-triangle atlas charts; geometry untouched."""
import json,sys,time
from pathlib import Path
import bpy,numpy as np
from mathutils import Vector
from mathutils.bvhtree import BVHTree
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from render.texture_transfer import barycentric, sample_image, chart_grid


def main():
    c=json.loads(Path(sys.argv[sys.argv.index('--')+1]).read_text());out=Path(c['output']);cfg=c['transfer']
    bpy.ops.wm.read_factory_settings(use_empty=True)
    bpy.ops.import_scene.fbx(filepath=str(Path(c['asset'])/cfg['fbx']))
    donors=[o for o in bpy.data.objects if o.type=='MESH' and cfg['lung_material'] in [m.name for m in o.data.materials]]
    if len(donors)!=1:raise ValueError('Expected one donor lung surface mesh')
    donor=donors[0];m=donor.data;m.calc_loop_triangles()
    v=np.array([donor.matrix_world@p.co for p in m.vertices]);axes=cfg['donor_to_ras_axes'];v=np.stack([v[:,abs(k)-1]*np.sign(k) for k in axes],axis=1)
    f=np.array([t.vertices[:] for t in m.loop_triangles],np.int32)
    uv=np.array([[m.uv_layers.active.data[i].uv[:] for i in t.loops] for t in m.loop_triangles])
    if not np.isfinite(uv).all() or uv.min()<-1e-5 or uv.max()>1.00001:raise ValueError('Unexpected donor UV range')
    meta=json.loads(Path(c['atlas_mesh']).with_suffix('.json').read_text());atlas=np.load(c['atlas_mesh']);texture_archive=np.load(out/'source_pixels.npz');textures={k:texture_archive[k].copy() for k in texture_archive.files};texture_archive.close()
    objs=[o for o in meta['objects'] if o['fine_id']==18];records=[];arrays={};resolution=int(cfg['pixels_per_face']);weights,chart_corners=chart_grid(resolution)
    for side in ['left','right']:
        items=[o for o in objs if side in o['name']]
        tv=np.concatenate([atlas[o['key']+'_v'] for o in items]);tlo=tv.min(0);thi=tv.max(0)
        # Split donor lungs at their mid-sagittal bbox plane after axis mapping.
        mid=(v[:,0].min()+v[:,0].max())/2;keep=(v[f].mean(1)[:,0]>mid) if side=='right' else (v[f].mean(1)[:,0]<=mid)
        sf=f[keep];suv=uv[keep];sv=v[np.unique(sf)];lo=sv.min(0);hi=sv.max(0);normalized=(v-lo)/(hi-lo)
        bvh=BVHTree.FromPolygons(normalized.tolist(),sf.tolist(),all_triangles=True)
        for o in items:
            start=time.perf_counter();key=o['key'];verts=atlas[key+'_v'];faces=atlas[key+'_f'];mapped=(verts-tlo)/(thi-tlo)
            grid=int(np.ceil(np.sqrt(len(faces))));size=grid*resolution;baked=np.zeros((size,size,4),dtype=np.float32);outuv=np.zeros((len(faces),3,2),dtype=np.float32);errors=[]
            for i,face in enumerate(faces):
                points=weights@mapped[face];samples=[]
                for point in points:
                    hit,normal,idx,distance=bvh.find_nearest(Vector(point))
                    if hit is None:raise ValueError('No donor correspondence')
                    bc=barycentric(np.asarray(hit),normalized[sf[idx]])
                    source_uv=bc@suv[idx];samples.append(source_uv);errors.append(distance)
                su=np.array(samples);color=sample_image(textures['color'],su);height=sample_image(textures['height'],su)
                x=(i%grid)*resolution;y=(i//grid)*resolution
                baked[y:y+resolution,x:x+resolution,:3]=color.reshape(resolution,resolution,3)
                baked[y:y+resolution,x:x+resolution,3]=height.reshape(resolution,resolution)
                outuv[i]=(chart_corners+np.array([x,y]))/size
            arrays[key+'_f']=faces;arrays[key+'_uv']=outuv
            # Image array y=0 is bottom (UV convention); host flips once on PNG save.
            np.save(out/(key+'_pixels.npy'),np.rint(baked*255).clip(0,255).astype(np.uint8))
            records.append({'name':o['name'],'key':key,'vertices':len(verts),'faces':len(faces),'resolution':size,
              'distance_normalized_p50_p95_max':np.quantile(errors,[.5,.95,1]).tolist(),'seconds':time.perf_counter()-start})
            print(records[-1],flush=True)
    np.savez_compressed(out/'mapping.npz',**arrays)
    (out/'manifest.json').write_text(json.dumps({'schema_version':1,'objects':records,'transfer':cfg,
      'method':'per-side bbox -> nearest donor surface -> barycentric donor UV -> triangle chart bake',
      'normal_map_used':False,'geometry_changed':False,'source_donor':donor.name},indent=2))
if __name__=='__main__':main()
