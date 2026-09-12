"""Optional compiled CPU rasterizer; geometry.render retains a NumPy fallback."""
import math
import numpy as np
from numba import njit

@njit(cache=True)
def rasterize(vertices, colours, K, rgb, depth):
    height,width=depth.shape
    for idx in range(len(vertices)):
        # Sutherland-Hodgman clipping against the camera near plane.
        source=vertices[idx]
        poly=np.empty((4,3),np.float64);n=0
        for j in range(3):
            a=source[j];b=source[(j+1)%3]
            if a[2]>=.2:
                poly[n]=a;n+=1
            if (a[2]>=.2)!=(b[2]>=.2):
                poly[n]=a+(b-a)*(.2-a[2])/(b[2]-a[2]);n+=1
        for k in range(1,n-1):
            v=np.empty((3,3),np.float64)
            v[0]=poly[0];v[1]=poly[k];v[2]=poly[k+1]
            h=v@K.T;uv=h[:,:2]/h[:,2:3]
            xmin=max(math.floor(min(uv[0,0],uv[1,0],uv[2,0])),0)
            xmax=min(math.ceil(max(uv[0,0],uv[1,0],uv[2,0])),width-1)
            ymin=max(math.floor(min(uv[0,1],uv[1,1],uv[2,1])),0)
            ymax=min(math.ceil(max(uv[0,1],uv[1,1],uv[2,1])),height-1)
            a,b,c=uv[0],uv[1],uv[2]
            den=(b[1]-c[1])*(a[0]-c[0])+(c[0]-b[0])*(a[1]-c[1])
            if abs(den)<1e-10:continue
            for y in range(ymin,ymax+1):
                for x in range(xmin,xmax+1):
                    u=((b[1]-c[1])*(x+.5-c[0])+(c[0]-b[0])*(y+.5-c[1]))/den
                    w=((c[1]-a[1])*(x+.5-c[0])+(a[0]-c[0])*(y+.5-c[1]))/den
                    t=1-u-w
                    if u<0 or w<0 or t<0:continue
                    z=1/max(u/v[0,2]+w/v[1,2]+t/v[2,2],1e-15)
                    if z<depth[y,x]:
                        depth[y,x]=z
                        rgb[y,x]=colours[idx]
