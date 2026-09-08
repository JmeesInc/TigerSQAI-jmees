"""Float Object Index -> uint8 grayscale; aligned nearest label/depth lens warp."""
from pathlib import Path
import numpy as np
import OpenEXR
from PIL import Image


def read_passes(path):
    with OpenEXR.File(str(path),separate_channels=True) as file:
        channels=file.channels()
        def get(tokens):
            hits=[k for k in channels if any(token.lower() in k.lower() for token in tokens)]
            if len(hits)!=1:raise ValueError(f'Ambiguous/missing pass {tokens}: {list(channels)}')
            return channels[hits[0]].pixels.copy()
        index=get(['IndexOB','Object Index'])
        depth=get(['Depth.Z'])
    if not np.isfinite(index).all() or np.max(np.abs(index-np.rint(index)))>1e-5:
        raise ValueError('Object Index contains nonintegral/invalid values; refuse color or AA recovery')
    if index.min()<0 or index.max()>30:raise ValueError('Index outside 0..30')
    return np.rint(index).astype(np.uint8),depth.astype(np.float32)


def remap(index,depth,camera,grid):
    sx,sy,valid=grid
    label=index[sy,sx].copy()
    distance=depth[sy,sx].copy()
    label[~valid]=0
    # Cycles perspective Depth.Z is axial camera depth (verified by off-axis planes
    # in Blender 4.5 and 5.2). Do not divide by a ray-length factor a second time.
    axial=distance
    axial[(~valid)|(~np.isfinite(axial))|(distance>=1e9)]=0.
    return label,axial.astype(np.float32)


def reject_reason(label,valid,config,progress):
    q=config['quality'];counts=np.bincount(label.ravel(),minlength=31)
    if valid.mean()<q['minimum_valid_optics_fraction']:return 'lens coverage'
    if counts.max()/label.size>=q['maximum_class_fraction']:return 'class dominance'
    if counts[0]/label.size>q['maximum_background_fraction']:return 'background dominance'
    classes=q.get('visible_class_count',{})
    if classes.get('enabled',False):
        n=int(np.count_nonzero(counts[0 if classes.get('includes_background',False) else 1:]>=classes.get('minimum_pixels',1)))
        if not classes['min']<=n<=classes['max']:return 'visible class count'
    major=q['main_class_ids'][:]
    if progress<=q['covered_phase_t_max']:major+=q['covered_phase_class_ids']
    if counts[major].sum()<q['minimum_main_pixels']:return 'no main structure'
    return None


def class_count_probability(label,config):
    """Optional soft population prior within the hard class-count band."""
    q=config['quality'].get('visible_class_count',{})
    prior=q.get('soft_preference')
    if not q.get('enabled',False) or prior is None:return 1.
    counts=np.bincount(label.ravel(),minlength=31)
    n=np.count_nonzero(counts[0 if q.get('includes_background',False) else 1:]>=q.get('minimum_pixels',1))
    if prior.get('proposal_counts'):
        support=np.arange(q['min'],q['max']+1)
        proposal=np.array([prior['proposal_counts'].get(int(k),prior['proposal_counts'].get(str(k),0)) for k in support],float)
        if np.any(proposal<=0):raise ValueError('Soft prior needs positive pilot counts for the complete hard-band support')
        weight=np.exp(-.5*((support-prior['mean'])/prior['sd'])**2)/proposal
        probability=weight/weight.max()
        return float(probability[n-q['min']]) if q['min']<=n<=q['max'] else 0.
    return float(np.exp(-.5*((n-prior['mean'])/prior['sd'])**2))


def write_outputs(output,frame_id,label,depth):
    output=Path(output)
    # PIL mode L is 8-bit grayscale, not palette RGB.
    Image.fromarray(label).save(output/'label'/f'{frame_id}.png')
    OpenEXR.File({'compression':OpenEXR.ZIP_COMPRESSION,'type':OpenEXR.scanlineimage},
                 {'Z':np.ascontiguousarray(depth,dtype=np.float32)}).write(str(output/'depth'/f'{frame_id}.exr'))
