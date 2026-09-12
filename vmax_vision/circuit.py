"""Surveyed planar track polygon with signed edge distance (metres)."""
import numpy as np


class CircuitMap:
    def __init__(self, polygon):
        self.polygon=np.asarray(polygon,float)
        if self.polygon.ndim!=2 or self.polygon.shape[1]!=2 or len(self.polygon)<3 or not np.isfinite(self.polygon).all():
            raise ValueError('track polygon needs at least three finite XY vertices')
        if np.allclose(self.polygon[0],self.polygon[-1]):self.polygon=self.polygon[:-1]
        a=self.polygon;b=np.roll(a,-1,axis=0)
        if np.any(np.linalg.norm(b-a,axis=1)<1e-8):raise ValueError('duplicate adjacent polygon vertices')
        if abs(np.sum(a[:,0]*b[:,1]-b[:,0]*a[:,1]))<1e-8:raise ValueError('degenerate polygon')
        # Reject self-crossing polygons; holes and grade separation need a richer map.
        def orient(a,b,c):
            u,v=b-a,c-a
            return float(u[0]*v[1]-u[1]*v[0])
        for i in range(len(a)):
            for j in range(i+1,len(a)):
                if j in ((i+1)%len(a),(i-1)%len(a)):continue
                if orient(a[i],b[i],a[j])*orient(a[i],b[i],b[j])<0 and orient(a[j],b[j],a[i])*orient(a[j],b[j],b[i])<0:
                    raise ValueError('self-intersecting polygon')

    def signed_excess(self, points):
        points=np.asarray(points,float);shape=points.shape[:-1];p=points.reshape(-1,2)
        distance=np.full(len(p),np.inf);inside=np.zeros(len(p),bool)
        a=self.polygon;b=np.roll(a,-1,axis=0)
        for u,v in zip(a,b):
            d=v-u;t=np.clip(((p-u)@d)/(d@d),0,1)
            distance=np.minimum(distance,np.linalg.norm(p-u-t[:,None]*d,axis=1))
            if v[1]!=u[1]:
                cross=((u[1]>p[:,1])!=(v[1]>p[:,1])) & (p[:,0]<(v[0]-u[0])*(p[:,1]-u[1])/(v[1]-u[1])+u[0])
                inside^=cross
        return np.where(inside,-distance,distance).reshape(shape)
