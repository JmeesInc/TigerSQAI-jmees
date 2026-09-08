"""Optional 3D polyharmonic TPS (U(r)=-r) and public NIfTI-mask surface import."""
import numpy as np
from scipy.spatial.distance import cdist


class ThinPlateSpline3D:
    def __init__(self,source,target,regularization=1e-3):
        self.source=np.asarray(source,dtype=float)
        target=np.asarray(target,dtype=float)
        if self.source.shape!=target.shape or self.source.ndim!=2 or self.source.shape[1]!=3 or len(target)<5:
            raise ValueError('TPS requires >=5 paired 3D landmarks')
        p=np.column_stack([np.ones(len(target)),self.source])
        if np.linalg.matrix_rank(p)<4:
            raise ValueError('TPS landmarks must span 3D, not a plane/line')
        k=-cdist(self.source,self.source)
        matrix=np.block([[k+regularization*np.eye(len(k)),p],[p.T,np.zeros((4,4))]])
        coeff=np.linalg.solve(matrix,np.vstack([target,np.zeros((4,3))]))
        self.weights,self.affine=coeff[:-4],coeff[-4:]

    def __call__(self,points):
        points=np.asarray(points,dtype=float)
        result=[]
        for chunk in np.array_split(points,max(1,int(np.ceil(len(points)/20000)))):
            result.append(-cdist(chunk,self.source)@self.weights+np.column_stack([np.ones(len(chunk)),chunk])@self.affine)
        return np.concatenate(result)

    def jacobian_determinants(self,points,epsilon=0.25):
        gradients=[]
        for axis in np.eye(3):
            gradients.append((self(points+axis*epsilon)-self(points-axis*epsilon))/(2*epsilon))
        return np.linalg.det(np.stack(gradients,axis=2))


def apply_patient(objects,ribs,landmarks,patient):
    import nibabel as nib
    from skimage.measure import marching_cubes
    pairs=patient['landmark_pairs']
    groups={p['group'] for p in pairs}
    if not {'skeleton','trachea','aorta'}<=groups:
        raise ValueError('Provide skeleton, trachea, and aorta landmark groups')
    warp=ThinPlateSpline3D([p['atlas_ras_mm'] for p in pairs],[p['ct_ras_mm'] for p in pairs],patient.get('regularization',1e-3))
    all_points=np.concatenate([o['v'][::max(1,len(o['v'])//200)] for o in objects])
    det=warp.jacobian_determinants(all_points)
    if det.min()<=patient.get('minimum_jacobian_determinant',0.05):
        raise ValueError('TPS fold or excessive compression detected; revise landmarks')
    for o in objects:o['v']=warp(o['v']).astype(np.float32)
    ribs={k:warp(v) for k,v in ribs.items()}
    landmarks={k:warp([v])[0].tolist() for k,v in landmarks.items()}
    replacements=[]
    for item in patient['masks']:
        fine_id=int(item['fine_id'])
        if fine_id not in [3,6,15,17,18]:
            raise ValueError('CT replacements here are limited to Task A category (b)')
        if not item.get('source_url') or not item.get('license') or not item.get('sha256'):
            raise ValueError('CT mask provenance requires source_url, license, sha256')
        from .util import sha256
        if sha256(item['path'])!=item['sha256']:
            raise ValueError(f"CT mask hash mismatch: {item['path']}")
        image=nib.load(item['path'])
        if image.header.get_xyzt_units()[0]!='mm':
            raise ValueError('NIfTI spatial units must explicitly be mm')
        data=np.asanyarray(image.dataobj)
        mask=np.isin(data,item['label_values']) if 'label_values' in item else data>0
        if not mask.any():raise ValueError('Empty CT replacement mask')
        vertices,faces,_,_=marching_cubes(np.pad(mask.astype(np.float32),1),0.5)
        vertices=nib.affines.apply_affine(image.affine,vertices-1)
        replacements.append({'name':item['name'],'fine_id':fine_id,'v':vertices.astype(np.float32),'f':faces.astype(np.int32)})
    replaced={o['fine_id'] for o in replacements}
    objects=[o for o in objects if o['fine_id'] not in replaced]+replacements
    return objects,ribs,landmarks,{'method':'3D polyharmonic TPS U(r)=-r','landmark_count':len(pairs),
        'min_sampled_jacobian':float(det.min()),'max_sampled_jacobian':float(det.max()),
        'landmark_fit_rmse_mm':float(np.sqrt(np.mean((warp([p['atlas_ras_mm'] for p in pairs])-np.array([p['ct_ras_mm'] for p in pairs]))**2))),
        'note':'Positive sampled Jacobians do not prove global injectivity; inspect alignment'}
