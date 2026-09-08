"""Camera targets and aperture size from the actual t-dependent removed pleural surface."""
import numpy as np
from scipy import ndimage as ndi
from scipy.spatial import cKDTree
from .volume import cut_mask


def context(volume,landmarks,dissection,t,ports=(),options=None):
    options=options or {}
    removed,_=cut_mask(volume,landmarks,dissection,t)
    outer=volume['covered_solid'];surface=outer&~ndi.binary_erosion(outer)
    opened=np.argwhere(surface&removed)*volume['pitch']+volume['origin']
    retained=np.argwhere(surface&~removed)*volume['pitch']+volume['origin']
    windows=[]
    if not len(opened) or not len(retained):return {'windows':[],'exposed_landmarks':[],'removed':removed,'volume':volume}
    tree=cKDTree(retained)
    gradient=np.gradient(ndi.gaussian_filter(outer.astype(float),1.0))
    for phase in dissection['phases']:
        for number,w in enumerate(phase['windows']):
            a,b=w.get('interval',phase['interval']);growth=np.clip((t-a)/(b-a),0,1)**(1/3)
            if growth<=0:continue
            center=np.array(landmarks[w['anchor']])+w['offset_mm'];radii=np.array(w['radii_mm'])*growth
            selected=opened[np.sum(((opened-center)/radii)**2,axis=1)<1.2**2]
            # Right-facing aperture: retain anterior/posterior width but not far-side surfaces.
            selected=selected[selected[:,0]>=center[0]]
            if len(selected)<4:continue
            # Opening center near the nominal window center in the tangent YZ plane.
            for port_index,port in enumerate(ports or [{'position_mm':center+[100,0,0]}]):
                point=np.asarray(port['position_mm']);indices=np.rint((selected-volume['origin'])/volume['pitch']).astype(int)
                normals=-np.column_stack([g[tuple(indices.T)] for g in gradient])
                direction=point-selected;direction/=np.maximum(np.linalg.norm(direction,axis=1,keepdims=True),1e-9)
                cosine=np.sum(normals*direction,axis=1)/np.maximum(np.linalg.norm(normals,axis=1),1e-9)
                facing=selected[cosine>options.get('minimum_facing_cosine',.5)]
                if not len(facing):continue
                ranked=np.argsort(np.linalg.norm((facing-center)/radii,axis=1))
                centers=[]
                for rank in ranked:
                    opening=facing[rank]
                    if any(np.linalg.norm(opening-q)<options.get('patch_spacing_mm',8.) for q in centers):continue
                    centers.append(opening)
                    radius=float(tree.query(opening)[0])
                    windows.append({'name':f"{phase['name']}:{number}:port{port_index}:patch{len(centers)}",'anchor':w['anchor'],'port_index':port_index,'center_mm':opening.tolist(),'nominal_center_mm':center.tolist(),'effective_radius_mm':radius,'nominal_radii_mm':radii.tolist(),'center_distance_weight':float(np.exp(-options.get('center_preference_strength',2.)*np.sum(((opening-center)/radii)**2)))})
                    if len(centers)>=options.get('maximum_patches_per_window_port',12):break
    exposed=[]
    for name,point in landmarks.items():
        index=np.rint((np.array(point)-volume['origin'])/volume['pitch']).astype(int)
        if np.all(index>=0) and np.all(index<removed.shape) and removed[tuple(index)]:exposed.append(name)
    return {'windows':windows,'exposed_landmarks':exposed,'removed':removed,'volume':volume}


def is_removed(point,ctx):
    v=ctx['volume'];idx=np.rint((np.asarray(point)-v['origin'])/v['pitch']).astype(int)
    return bool(np.all(idx>=0) and np.all(idx<ctx['removed'].shape) and ctx['removed'][tuple(idx)])
