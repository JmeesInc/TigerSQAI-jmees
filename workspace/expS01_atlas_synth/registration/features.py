"""Interpretable mask descriptors and capability-masked distances, no learned models."""
import numpy as np
from PIL import Image


def read_label(path,size=None,tolerance=.02):
    with Image.open(path) as im:
        if im.mode not in ['L','P']:raise ValueError(f'{path}: require grayscale or indexed IDs, never RGB')
        a=np.array(im);original=im.size
        if a.max()>30:raise ValueError(f'{path}: fine ID outside 0..30')
        if size is not None:
            if abs((im.width/im.height)/(size[0]/size[1])-1)>tolerance:raise ValueError(f'{path}: aspect mismatch; supply explicit crop/intrinsics policy')
            a=np.asarray(im.resize(tuple(size),Image.Resampling.NEAREST))
    return a,original


def describe(a,minimum_pixels=1):
    h,w=a.shape;counts=np.bincount(a.ravel(),minlength=31)
    present=counts>=minimum_pixels;area=counts/a.size
    centroid=np.zeros((31,2));moment=np.zeros((31,3))
    for c in np.flatnonzero(present):
        y,x=np.where(a==c);x=(x+.5)/w;y=(y+.5)/h
        centroid[c]=[x.mean(),y.mean()];x=x-x.mean();y=y-y.mean();moment[c]=[np.mean(x*x),np.mean(y*y),np.mean(x*y)]
    adjacency=np.zeros((31,31),bool)
    for first,second in [(a[:,:-1],a[:,1:]),(a[:-1,:],a[1:,:])]:
        different=first!=second;u=first[different];v=second[different];adjacency[u,v]=True;adjacency[v,u]=True
    return {'present':present,'area':area,'centroid':centroid,'moment':moment,'adjacency':adjacency}


def distances(query,bank,ids,weights):
    """Missing supported classes are penalized; missing centroid values are never treated as zero observations."""
    ids=np.array(ids);q=query['present'][ids];p=bank['present'][:,ids]
    common=p&q;union=p|q
    presence=(p!=q).sum(1)/np.maximum(union.sum(1),1)
    area=np.abs(bank['area'][:,ids]-query['area'][ids]).sum(1)/np.maximum(bank['area'][:,ids].sum(1)+query['area'][ids].sum(),1e-9)
    center=(np.linalg.norm(bank['centroid'][:,ids]-query['centroid'][ids],axis=2)/np.sqrt(2)*common).sum(1)/np.maximum(common.sum(1),1)
    moment=(np.linalg.norm(bank['moment'][:,ids]-query['moment'][ids],axis=2)*common).sum(1)/np.maximum(common.sum(1),1)
    qa=query['adjacency'][np.ix_(ids,ids)];ba=bank['adjacency'][:,ids][:,:,ids]
    # Pairwise edge Jaccard; edges involving only one-sided presence remain mismatches.
    adj=(ba!=qa).sum((1,2))/np.maximum((ba|qa).sum((1,2)),1)
    terms={'presence':presence,'area':area,'centroid':center,'moment':moment,'adjacency':adj}
    return sum(weights[k]*v for k,v in terms.items())/sum(weights.values())


def weighted_iou(real,synth,ids,weights,minimum_pixels=1,real_background_policy='ignore'):
    eligible=np.zeros(31,bool);eligible[ids]=True
    # Excluded foreground (e.g. tools) is unknown occlusion on either side.
    # Real background is unknown FOV; synthetic background under real anatomy remains an error.
    valid=((real==0)|eligible[real]) & ((synth==0)|eligible[synth])
    if real_background_policy=='ignore':valid &= real!=0
    elif real_background_policy!='compare':raise ValueError('Unknown real_background_policy')
    common=[];scores={};numerator=denominator=0.
    for c in ids:
        a=(real==c)&valid;b=(synth==c)&valid;union=np.count_nonzero(a|b)
        if union<minimum_pixels:continue
        score=np.count_nonzero(a&b)/union;scores[str(c)]=float(score)
        numerator+=weights[c]*score;denominator+=weights[c]
        if np.count_nonzero(a)>=minimum_pixels and np.count_nonzero(b)>=minimum_pixels:common.append(c)
    fraction=min(float((eligible[real]&valid).mean()),float((eligible[synth]&valid).mean()))
    return {'weighted_iou':numerator/denominator if denominator else 0.,'class_iou':scores,'common_visible_classes':common,'comparable_pixel_fraction':fraction,'valid_pixel_fraction':float(valid.mean())}
