"""Deterministic synthetic vascular texture, not an anatomical vessel map."""
import numpy as np


def vascular_tile(size=512,seed=611):
    rng=np.random.default_rng(seed);im=np.zeros((size,size),np.float32)
    def stroke(a,b,width,value):
        a=np.array(a);b=np.array(b);lo=np.maximum(0,np.floor(np.minimum(a,b)-width).astype(int));hi=np.minimum(size,np.ceil(np.maximum(a,b)+width+1).astype(int))
        if np.any(hi<=lo):return
        yy,xx=np.mgrid[lo[1]:hi[1],lo[0]:hi[0]];v=b-a;den=max(float(v@v),1e-8);t=np.clip(((xx-a[0])*v[0]+(yy-a[1])*v[1])/den,0,1)
        distance=np.hypot(xx-a[0]-t*v[0],yy-a[1]-t*v[1]);patch=im[lo[1]:hi[1],lo[0]:hi[0]];np.maximum(patch,np.clip(width/2+.7-distance,0,1)*value,out=patch)
    def branch(x,y,angle,length,width,level):
        points=[(x,y)];steps=8
        for _ in range(steps):
            angle+=rng.normal(0,.08);x+=np.cos(angle)*length/steps;y+=np.sin(angle)*length/steps;points.append((x,y))
        # Periodic copies avoid seams at tile boundaries; tapered descendant branches.
        for ox in [-size,0,size]:
            for oy in [-size,0,size]:
                for a,b in zip(points[:-1],points[1:]):stroke((a[0]+ox,a[1]+oy),(b[0]+ox,b[1]+oy),max(1,width),(160+level*12)/255)
        if level:
            for sign in [-1,1]:branch(x,y,angle+sign*rng.uniform(.35,.75),length*rng.uniform(.55,.75),width*.65,level-1)
    for x,y,angle in [(size*.2,size*.1,.9),(size*.8,size*.8,3.8)]:branch(x,y,angle,size*.22,4.,5)
    return (im*.6+np.roll(im,1,0)*.1+np.roll(im,-1,0)*.1+np.roll(im,1,1)*.1+np.roll(im,-1,1)*.1)


def texture_axes(vertices):
    """Read-only principal axial coordinate; local straight-axis approximation."""
    v=np.asarray(vertices);center=v.mean(0);_,_,vt=np.linalg.svd(v-center,full_matrices=False)
    axis=vt[0];axis*=1 if axis[np.argmax(np.abs(axis))]>=0 else -1
    return (v-center)@axis


def texture_transverse(vertices):
    v=np.asarray(vertices);center=v.mean(0);_,_,vt=np.linalg.svd(v-center,full_matrices=False)
    axis=vt[1];axis*=1 if axis[np.argmax(np.abs(axis))]>=0 else -1
    return (v-center)@axis
