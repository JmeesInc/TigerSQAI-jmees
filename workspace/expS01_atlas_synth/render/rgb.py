"""Combined linear RGB -> shared nearest lens warp -> tone mapping -> RGB PNG."""
import numpy as np
import OpenEXR
from PIL import Image


def read_rgb(raw):
    with OpenEXR.File(str(raw),separate_channels=True) as exr:
        channels=exr.channels();names=[]
        for c in 'RGB':
            hits=[k for k in channels if k.endswith('.Combined.'+c)]
            if len(hits)!=1:raise ValueError('Missing/ambiguous Combined RGB')
            names.append(hits[0])
        return np.stack([channels[k].pixels for k in names],axis=-1)


def write_rgb(raw,path,grid,depth,cfg):
    sx,sy,valid=grid;rgb=read_rgb(raw)[sy,sx].copy();rgb[~valid]=0
    if not np.isfinite(rgb).all():raise ValueError('Nonfinite RGB')
    if cfg['haze']['enabled']:
        transmission=np.exp(-cfg['haze']['extinction_per_mm']*depth)[...,None]
        rgb=rgb*transmission+np.asarray(cfg['haze']['color_linear'])*(1-transmission)
    rgb=np.maximum(rgb,0)*2**cfg['tone']['exposure_stops'];rgb=rgb/(1+rgb)
    rgb=np.where(rgb<=.0031308,12.92*rgb,1.055*rgb**(1/2.4)-.055)
    Image.fromarray(np.rint(np.clip(rgb,0,1)*255).astype(np.uint8)).save(path)
