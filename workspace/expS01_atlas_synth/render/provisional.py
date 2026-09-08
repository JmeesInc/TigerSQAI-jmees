"""Explicitly provisional surfaces constrained by existing atlas geometry, never verified anatomy."""
import numpy as np
import trimesh
from scipy import ndimage as ndi
from .volume import mesh_from_mask


def heart_surface(sources,cfg,exclusions=()):
    pitch=cfg['voxel_mm']
    points=[]
    for o in sources:
        mesh=trimesh.Trimesh(o['v'],o['f'],process=True)
        points.append(mesh.voxelized(pitch).fill().points)
    pts=np.concatenate(points)
    pad=cfg['margin_mm']+(cfg['closing_iterations']+3)*pitch
    origin=np.floor((pts.min(0)-pad)/pitch)*pitch
    shape=np.ceil((pts.max(0)+pad-origin)/pitch).astype(int)+1
    if np.prod(shape)>20000000:raise ValueError('Heart proxy grid too large')
    solid=np.zeros(shape,bool);idx=np.rint((pts-origin)/pitch).astype(int);solid[tuple(idx.T)]=True
    solid=ndi.binary_closing(solid,iterations=cfg['closing_iterations']) if cfg['closing_iterations'] else solid
    solid=ndi.binary_fill_holes(solid)
    solid=ndi.distance_transform_edt(~solid,sampling=pitch)<=cfg['margin_mm']
    # Exclude a clearance volume around the original IPV meshes AFTER dilation.
    # Subtracting before closing/dilation would simply fill the venous ostia again.
    protected=np.zeros(shape,bool)
    for obj in exclusions:
        vox=trimesh.Trimesh(obj['v'],obj['f'],process=True).voxelized(pitch).fill()
        indices=np.rint((vox.points-origin)/pitch).astype(int)
        indices=indices[np.all((indices>=0)&(indices<shape),axis=1)]
        protected[tuple(indices.T)]=True
    if protected.any():
        solid &= ndi.distance_transform_edt(~protected,sampling=pitch)>cfg.get('ipv_clearance_mm',4.)
    return mesh_from_mask(solid,origin,pitch)


def ligament_surface(lung,vein,mediastinum,side,cfg):
    """Double pleural strip below IPV joining existing medial lower-lobe/mediastinal points."""
    top=vein.mean(0)
    bottom=max(float(lung[:,2].min()+cfg['z_band_mm']),float(top[2]-cfg['length_mm']))
    if bottom>=top[2]-2:raise ValueError('No lung below IPV for ligament proxy')
    left=[];right=[]
    for z in np.linspace(top[2]-1,bottom,cfg['samples']):
        pool=lung[(abs(lung[:,2]-z)<cfg['z_band_mm']) & (abs(lung[:,1]-top[1])<cfg['y_band_mm'])]
        if not len(pool):raise ValueError('Ligament lung anchor unavailable; review ROI/landmarks')
        # R medial=min X; L medial=max X; prefer same craniocaudal level.
        score=side*pool[:,0]+.3*abs(pool[:,2]-z)
        a=pool[np.argmin(score)].copy()
        pool=mediastinum[abs(mediastinum[:,2]-a[2])<cfg['z_band_mm']]
        pool=pool[(side*(a[0]-pool[:,0]))>0]
        if not len(pool):raise ValueError('Ligament mediastinal anchor unavailable')
        b=pool[np.argmin(np.linalg.norm(pool-a,axis=1))].copy()
        left.append(a);right.append(b)
    v=np.stack([left,right],axis=1).reshape(-1,3)
    f=[]
    for i in range(len(left)-1):
        j=2*i;f.extend([[j,j+1,j+3],[j,j+3,j+2]])
    f=np.array(f);mesh=trimesh.Trimesh(v,f,process=False);n=mesh.vertex_normals*cfg['thickness_mm']/2
    boundary=mesh.edges[trimesh.grouping.group_rows(mesh.edges_sorted,require_count=1)]
    size=len(v)
    side_faces=np.concatenate([np.column_stack([boundary[:,0],boundary[:,1],boundary[:,1]+size]),np.column_stack([boundary[:,0],boundary[:,1]+size,boundary[:,0]+size])])
    return np.concatenate([v+n,v-n]),np.concatenate([f,f[:,::-1]+size,side_faces])


def add_proxies(objects,atlas):
    cfg=atlas.get('provisional',{})
    sources=[o for o in objects if o.get('role')=='proxy_source']
    objects=[o for o in objects if o.get('role')!='proxy_source']
    records=[]
    def add(cid,name,v,f,parents,method):
        record={'fine_id':cid,'name':name,'status':'provisional_anatomy_proxy','source_objects':parents,'method':method,'source_url':atlas['source_url'],'license':atlas['license']}
        objects.append(record|{'role':'provisional','v':v,'f':f});records.append(record)
    if not cfg.get('enabled',False):return objects,records
    if cfg['pericardium']['enabled']:
        if {o['name'] for o in sources}!=set(cfg['pericardium']['source_objects']):raise ValueError('Missing verified existing heart source meshes')
        exclusions=[o for o in objects if o['fine_id']==12] if cfg['pericardium'].get('subtract_ipv',True) else []
        v,f=heart_surface(sources,cfg['pericardium'],exclusions);add(11,'provisional_pericardium',v,f,[o['name'] for o in sources]+[o['name'] for o in exclusions],'Heart union plus dilation, then original IPV clearance subtraction; provisional, not true pericardial reflections')
    if cfg['pulmonary_ligaments']['enabled']:
        for side,cid,word in [(1,8,'right'),(-1,9,'left')]:
            # CT replacement may merge lungs/veins: exact lobe geometry required here.
            ln='Inferior lobe of '+word+' lung';vn=word.capitalize()+' inferior pulmonary vein'
            match={o['name']:o for o in objects}
            if ln not in match or vn not in match:raise ValueError('Ligament proxy requires original lobe and IPV identities; disable ligament proxies for merged CT lungs')
            medi=np.concatenate([o['v'][np.unique(o['f'])] for o in objects if o['fine_id'] in [6,11,15]])
            v,f=ligament_surface(match[ln]['v'][np.unique(match[ln]['f'])],match[vn]['v'][np.unique(match[vn]['f'])],medi,side,cfg['pulmonary_ligaments'])
            add(cid,'provisional_'+word+'_inferior_pulmonary_ligament',v,f,[ln,vn,'mediastinal surfaces'],'IPV-inferior double fold tethered to existing medial lung and mediastinum; not validated patient anatomy')
    return objects,records
