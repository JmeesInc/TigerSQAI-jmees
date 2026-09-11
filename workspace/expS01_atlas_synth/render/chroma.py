"""Explicit color-space conversion; optional canonical aggregate chroma schema."""
import numpy as np


def red_saturation_chroma(red,saturation):
    # Assumption: R is max and B min. r and HSV S alone do not identify G/B order.
    blue=red*(1-saturation);green=1-red-blue
    if not 0<=blue<=green<=red:raise ValueError('R-max/B-min chroma assumption incompatible with supplied r/S')
    return [red,green,blue]


def material_linear_color(p):
    c=np.asarray(p['base_chroma_rgb'],float)
    if c.shape!=(3,) or np.any(c<0) or not np.isclose(c.sum(),1,atol=1e-4):raise ValueError('Chroma must have three nonnegative components summing to 1')
    c=c/c.max()*p.get('base_value_srgb',.65)
    return np.where(c<=.04045,c/12.92,((c+.055)/1.055)**2.4).tolist()
