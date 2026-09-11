"""Pure numeric helpers for appearance-only donor surface baking."""
import numpy as np

def barycentric(point,tri):
    a,b,c=tri;v0=b-a;v1=c-a;v2=point-a
    d00=v0@v0;d01=v0@v1;d11=v1@v1;d20=v2@v0;d21=v2@v1;den=d00*d11-d01*d01
    if abs(den)<1e-20:return np.array([1.,0.,0.])
    y=(d11*d20-d01*d21)/den;z=(d00*d21-d01*d20)/den
    w=np.clip([1-y-z,y,z],0,1);return w/w.sum()

def sample_image(image,uv):
    h,w=image.shape[:2];x=np.clip(uv[:,0],0,1)*(w-1);y=(1-np.clip(uv[:,1],0,1))*(h-1)
    x0=x.astype(int);y0=y.astype(int);x1=np.minimum(x0+1,w-1);y1=np.minimum(y0+1,h-1);dx=x-x0;dy=y-y0
    if image.ndim==3:dx=dx[:,None];dy=dy[:,None]
    return image[y0,x0]*(1-dx)*(1-dy)+image[y0,x1]*dx*(1-dy)+image[y1,x0]*(1-dx)*dy+image[y1,x1]*dx*dy

def chart_grid(size):
    if size<6:raise ValueError('pixels_per_face must be >=6')
    yy,xx=np.mgrid[:size,:size];u=(xx.ravel()+.5-1.5)/(size-3);v=(yy.ravel()+.5-1.5)/(size-3)
    # Extend triangle edges into chart padding to prevent inter-chart bleed.
    w=np.maximum(np.stack([1-u-v,u,v],1),0);w/=w.sum(1,keepdims=True)
    corners=np.array([[1.5,1.5],[size-1.5,1.5],[1.5,size-1.5]])
    return w,corners
