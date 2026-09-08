"""Voxel union of existing atlas surfaces; fat/pleura and monotone irregular dissection."""
import numpy as np
import trimesh
from scipy import ndimage as ndi
from skimage.measure import marching_cubes


def mesh_from_mask(mask, origin, pitch):
    if not np.any(mask):
        return np.empty((0,3),np.float32),np.empty((0,3),np.int32)
    # Padding closes shells on ROI boundaries.
    v, f, _, _ = marching_cubes(np.pad(mask.astype(np.float32),1), 0.5, spacing=(pitch,)*3)
    return (v+origin-pitch).astype(np.float32), f.astype(np.int32)


def build_volume(objects, atlas, config):
    pitch = float(config['voxel_mm'])
    lo, hi = np.asarray(atlas['roi_mm'],dtype=float)
    origin = np.floor(lo/pitch)*pitch
    shape = np.ceil((hi-origin)/pitch).astype(int)+1
    if np.prod(shape)>config['max_grid_voxels']:
        raise ValueError(f'Grid {shape} exceeds max_grid_voxels')
    core = np.zeros(shape,dtype=bool)
    esophagus = np.zeros_like(core)
    for obj in objects:
        if obj['fine_id'] not in config['envelope_class_ids']:
            continue
        mesh = trimesh.Trimesh(obj['v'],obj['f'],process=False)
        # Subdivision voxelizer needs no rtree/Embree; fill encloses the original surfaces.
        vox = mesh.voxelized(pitch,method='subdivide').fill()
        idx = np.rint((vox.points-origin)/pitch).astype(int)
        idx = idx[np.all((idx>=0)&(idx<shape),axis=1)]
        core[tuple(idx.T)] = True
        if obj['fine_id']==6:
            esophagus[tuple(idx.T)] = True
    if not core.any() or not esophagus.any():
        raise ValueError('Envelope requires nonempty atlas anatomy and esophagus')
    distance = ndi.distance_transform_edt(~core,sampling=pitch).astype(np.float32)
    eso_distance = ndi.distance_transform_edt(~esophagus,sampling=pitch).astype(np.float32)
    rng = np.random.default_rng(config['noise_seed'])
    coarse = np.maximum(3,np.ceil(shape*pitch/config['noise_correlation_mm']).astype(int))
    noise = ndi.zoom(rng.normal(size=coarse).astype(np.float32),shape/coarse,order=1)
    noise = noise[tuple(slice(0,int(s)) for s in shape)]
    noise /= max(float(noise.std()),1e-6)
    noise = np.clip(noise,-2,2)
    thickness = config['fat_thickness_mm'] + config['noise_amplitude_mm']*noise
    thickness = np.maximum(pitch,thickness)
    outer = (distance<=thickness)|(eso_distance<=config['esophageal_fat_thickness_mm'])
    fat = outer & ~core
    eso_fat = fat & (eso_distance<=config['esophageal_fat_thickness_mm'])
    # Distance outside the union, not individual disconnected membrane spheres.
    pleura = (~outer)&(ndi.distance_transform_edt(~outer,sampling=pitch)<=max(pitch,config['pleura_thickness_mm']))
    return {'origin':origin,'pitch':pitch,'fat':fat,'eso_fat':eso_fat,'pleura':pleura,
            'covered_solid':outer|pleura,'noise':noise}


def membrane_mesh(volume,removed,thickness):
    """Outer union surface with actual open windows, then a thin closed rim.

    Meshing a 1-voxel hollow label independently creates coincident inner surfaces
    at diagonal voxels. Extracting only the exterior avoids those triangle leaks.
    """
    v,f=mesh_from_mask(volume['covered_solid'],volume['origin'],volume['pitch'])
    grid=(v-volume['origin'])/volume['pitch']
    vertex_cut=ndi.map_coordinates(removed.astype(np.uint8),grid.T,order=0,mode='nearest')>0
    center_grid=(v[f].mean(axis=1)-volume['origin'])/volume['pitch']
    center_cut=ndi.map_coordinates(removed.astype(np.uint8),center_grid.T,order=0,mode='nearest')>0
    f=f[~(vertex_cut[f].any(axis=1)|center_cut)]
    if not len(f):return v,f
    mesh=trimesh.Trimesh(v,f,process=False)
    normals=mesh.vertex_normals
    count=len(v)
    vertices=np.concatenate([v+normals*thickness/2,v-normals*thickness/2])
    edges=np.concatenate([f[:,[0,1]],f[:,[1,2]],f[:,[2,0]]])
    _,inverse,counts=np.unique(np.sort(edges,axis=1),axis=0,return_inverse=True,return_counts=True)
    boundary=edges[counts[inverse]==1]
    sides=np.concatenate([np.column_stack([boundary[:,0],boundary[:,1],boundary[:,1]+count]),
                          np.column_stack([boundary[:,0],boundary[:,1]+count,boundary[:,0]+count])])
    faces=np.concatenate([f,f[:,::-1]+count,sides])
    return vertices.astype(np.float32),faces.astype(np.int32)


def cut_mask(volume, landmarks, config, progress):
    if not 0<=progress<=1:
        raise ValueError('progress must be in [0,1]')
    shape=volume['fat'].shape
    coords = np.ogrid[tuple(slice(0,n) for n in shape)]
    xyz = [volume['origin'][i]+coords[i]*volume['pitch'] for i in range(3)]
    removed = np.zeros(shape,dtype=bool)
    phase_name = config['phases'][0]['name']
    for phase in config['phases']:
        start,end=phase['interval']
        if progress>=start:
            phase_name=phase['name']
        fraction=np.clip((progress-start)/(end-start),0,1)
        if fraction<=0:
            continue
        for window in phase['windows']:
            window_start,window_end=window.get('interval',phase['interval'])
            window_fraction=np.clip((progress-window_start)/(window_end-window_start),0,1)
            if window_fraction<=0:continue
            center=np.asarray(landmarks[window['anchor']])+window['offset_mm']
            radii=np.asarray(window['radii_mm'])
            # Growth only; fixed patient noise means increasing t never restores tissue.
            rho=np.sqrt(sum(((xyz[i]-center[i])/radii[i])**2 for i in range(3)))
            irregular=rho+volume['noise']*config['noise_amplitude_mm']/float(np.min(radii))
            removed |= irregular <= window_fraction**(1/3)
    return removed,phase_name


def dissect(volume, landmarks, config, progress, rng):
    removed,phase = cut_mask(volume,landmarks,config,progress)
    fat=volume['fat']&~removed
    pleura=volume['pleura']&~removed
    resection=np.zeros_like(fat)
    r=config['resection']
    if r['enabled'] and rng.random()<r['probability'] and removed.any():
        adjacent=ndi.distance_transform_edt(~removed,sampling=volume['pitch'])<=r['rim_mm']
        resection=fat & adjacent & (volume['noise']>r['noise_threshold'])
    labels=[(20,fat&~volume['eso_fat']&~resection),
            (7,fat&volume['eso_fat']&~resection),(25,resection)]
    result=[]
    for class_id,mask in labels:
        v,f=mesh_from_mask(mask,volume['origin'],volume['pitch'])
        if len(f):result.append({'name':f'procedural_{class_id}','fine_id':class_id,'v':v,'f':f})
    v,f=membrane_mesh(volume,removed,config['pleura_surface_thickness_mm'])
    if len(f):result.append({'name':'pleural_membrane','fine_id':10,'v':v,'f':f})
    b=config['blood']
    if b['enabled'] and rng.random()<b['probability'] and removed.any():
        from .camera import normal as sample_normal,unit
        # Only exposed remaining-fat boundary adjacent to the opened window.
        surface=fat&ndi.binary_dilation(removed)
        indices=np.argwhere(surface)
        # Gradient points out of the residual tissue into the dissected window.
        gradient=np.gradient(ndi.gaussian_filter(removed.astype(np.float32),1.0),volume['pitch'])
        normals=np.column_stack([g[tuple(indices.T)] for g in gradient])
        lengths=np.linalg.norm(normals,axis=1)
        keep=lengths>1e-6
        indices,normals,lengths=indices[keep],normals[keep],lengths[keep]
        normals/=lengths[:,None]
        gravity=unit(b['gravity_ras'])
        keep=normals@gravity < -0.15 # upward supporting faces only
        indices,normals=indices[keep],normals[keep]
        points=indices*volume['pitch']+volume['origin']
        if len(points):
            scores=points@gravity
            keep=scores>=np.quantile(scores,b['dependent_quantile'])
            points,normals=points[keep],normals[keep]
            chosen=int(rng.integers(len(points)))
            normal=normals[chosen]
            center=points[chosen]+normal*(volume['pitch']*.5+b['depth_mm']*.35)
            radius=sample_normal(rng,b['radius_mm'])
            # A small flattened liquid patch on the dependent exposed rim.
            patch=trimesh.creation.icosphere(subdivisions=2,radius=1)
            a=np.eye(3)[np.argmin(np.abs(normal))]
            right=unit(np.cross(normal,a)); up=np.cross(normal,right)
            rotation=np.column_stack([right,up,normal])
            patch.vertices=(patch.vertices*np.array([radius,radius,b['depth_mm']]))@rotation.T+center
            result.append({'name':'blood_pool','fine_id':24,'v':patch.vertices.astype(np.float32),'f':patch.faces.astype(np.int32)})
    return result,{'progress':progress,'phase':phase,'removed_envelope_voxels':int(np.count_nonzero(removed&(volume['fat']|volume['pleura']))),
                   'remaining_fat_voxels':int(fat.sum()),'remaining_pleura_voxels':int(pleura.sum()),
                   'protocol_status':'illustrative_order_requires_surgical_review'}
