"""VMAX deterministic metre-based 3D synthetic dataset. Python + numpy/scipy/Pillow."""
import json, math, subprocess, pathlib
import numpy as np
from scipy.optimize import brentq
from PIL import Image, ImageDraw

OUT=pathlib.Path(__file__).parent/'output'; OUT.mkdir(exist_ok=True)
FPS=24; N=96; W,H=640,360; RADIUS=40.; HALF=7.; LINE=.1
KEYS=['front_left','front_right','rear_left','rear_right']
CONTACT=np.array([[1.8,.82], [1.8,-.82],[-1.8,.82],[-1.8,-.82]])
T=np.arange(N)/FPS; S=8+12*T

def center(s):
    s=np.asarray(s); a=np.clip(s,0,math.pi*20)/40
    p=np.stack([40*np.sin(a),40*(1-np.cos(a))],axis=-1)
    p[...,0]+=np.minimum(s,0); p[...,1]+=np.maximum(s-math.pi*20,0)
    normal=np.stack([-np.sin(a),np.cos(a)],axis=-1)
    return p,normal

def distance(p):
    """Independent exact closest distances to entry segment, circular arc, exit segment."""
    p=np.asarray(p); x,y=p[...,0],p[...,1]
    entry=np.hypot(x-np.clip(x,-30,0),y)
    exit=np.hypot(x-40,y-np.clip(y,40,70))
    a=np.clip(np.arctan2(x,40-y),0,np.pi/2)
    arc=np.hypot(x-40*np.sin(a),y-(40-40*np.cos(a)))
    return np.minimum(np.minimum(entry,exit),arc)

def pose(offset):
    c,n=center(S); p=c-n*offset[:,None]
    v=np.gradient(p,1/FPS,axis=0,edge_order=2)
    heading=np.arctan2(v[:,1],v[:,0]); f=np.stack([np.cos(heading),np.sin(heading)],1)
    left=np.stack([-f[:,1],f[:,0]],1)
    q=p[:,None,:]+CONTACT[None,:,0,None]*f[:,None,:]+CONTACT[None,:,1,None]*left[:,None,:]
    excess=distance(q)-HALF
    return p,heading,q,excess

def solve(desired):
    # Calibrate the peak of a keyframed lateral path, with heading recomputed
    # from that same path on each scalar solver evaluation.
    lo=float(desired.min()); hi=float(desired.max())
    if hi-lo<1e-10:
        offset=brentq(lambda o:pose(np.full(N,o))[3].min(1).max()-hi,.85,12,xtol=1e-12)
        return pose(np.full(N,offset))
    base=HALF+.82+lo
    shape=(desired-lo)/(hi-lo)
    amp=brentq(lambda a:pose(base+a*shape)[3].min(1).max()-hi,0,8,xtol=1e-12)
    return pose(base+amp*shape)

def profile(target,start=36,end=65,ramp=16):
    d=np.full(N,-1.5)
    for i in range(N):
        if start<=i<end: d[i]=target
        elif start-ramp<i<start: d[i]=-1.5+(target+1.5)*(.5-.5*np.cos(np.pi*(i-start+ramp)/ramp))
        elif end<=i<end+ramp: d[i]=-1.5+(target+1.5)*(.5+.5*np.cos(np.pi*(i-end+1)/ramp))
    return d

def camera(name,pos,target,fov):
    pos=np.array(pos,float); target=np.array(target,float); f=(target-pos); f/=np.linalg.norm(f)
    right=np.cross(f,[0,0,1]); right/=np.linalg.norm(right); down=np.cross(f,right)
    rot=np.array([right,down,f]); focal=H/2/np.tan(np.deg2rad(fov)/2)
    K=np.array([[focal,0,W/2],[0,focal,H/2],[0,0,1.]])
    hom=K@np.column_stack([rot[:,0],rot[:,1],-rot@pos]); hom/=hom[2,2]
    return dict(name=name,position_m=pos.tolist(),look_at_m=target.tolist(),vertical_fov_deg=fov,
      resolution=[W,H],fps=FPS,K=K.tolist(),R_world_to_camera=rot.tolist(),ground_plane_homography=hom.tolist(),near_m=.2)

CAMS=[camera('entry_low',[16,-8,4],[40,23,0],30),camera('exit_low',[55,21,4],[39,48,0],66),camera('wide',[63,-28,15],[25,17,0],62)]

def project(p,cam):
    c=(np.asarray(p)-cam['position_m'])@np.array(cam['R_world_to_camera']).T
    h=c@np.array(cam['K']).T
    return h[:,:2]/h[:,2,None],c[:,2]

# Real polygonal 3D car; local x forward, y left, z up. Tyres touch z=0.
def box(center,size,color):
    c=np.array(center); s=np.array(size)/2
    v=np.array([[x,y,z] for x in [-1,1] for y in [-1,1] for z in [-1,1]])*s+c
    faces=[[0,4,6,2],[1,3,7,5],[0,1,5,4],[2,6,7,3],[0,2,3,1],[4,5,7,6]]
    return [(v[[q[0],q[i],q[i+1]]],color) for q in faces for i in [1,2]]

def rod(a,b,r,color,segments=10):
    a=np.array(a,float); b=np.array(b,float); axis=b-a; axis/=np.linalg.norm(axis)
    u=np.cross(axis,[0,0,1] if abs(axis[2])<.9 else [0,1,0]); u/=np.linalg.norm(u); v=np.cross(axis,u)
    rings=[np.array([p+r*(np.cos(t)*u+np.sin(t)*v) for t in np.arange(segments)*2*np.pi/segments]) for p in [a,b]]
    tri=[]
    for i in range(segments):
        j=(i+1)%segments
        for q in [[rings[0][i],rings[1][i],rings[1][j]],[rings[0][i],rings[1][j],rings[0][j]],[a,rings[0][j],rings[0][i]],[b,rings[1][i],rings[1][j]]]: tri.append((np.array(q),color))
    return tri

def car_mesh(color=(225,42,43)):
    m=box((-.3,0,.32),(3.8,1.15,.48),color)+box((1.8,0,.34),(1.3,.35,.25),color)
    m+=box((2.5,0,.18),(.6,2,.12),(30,32,36))+box((-2.45,0,.85),(.65,1.9,.12),color)
    m+=box((-2.35,0,.55),(.18,.18,.6),(30,32,36))+box((-.5,0,.65),(1.0,.6,.3),(25,28,30))
    for x,y in CONTACT:
        m+=rod([x,y-.18,.36],[x,y+.18,.36],.36,(20,22,25),16)
        m+=rod([x,y-.185,.36],[x,y+.185,.36],.15,(120,125,130),12)
        m+=rod([x*.7,0,.3],[x,y,.36],.035,(40,40,45))
    for a,b in [([.2,-.4,.6],[-.6,-.4,1.0]),([-.6,-.4,1.0],[-.6,.4,1.0]),([-.6,.4,1.0],[.2,.4,.6]),([.2,0,.6],[-.6,0,1.0])]: m+=rod(a,b,.045,(240,240,240))
    return m

def strip(s,lo,hi,z,color):
    c,n=center(s); a=c+n*lo; b=c+n*hi; m=[]
    for i in range(len(s)-1):
        q=np.array([[*a[i],z],[*b[i],z],[*b[i+1],z],[*a[i+1],z]])
        m.extend([(q[[0,1,2]],color),(q[[0,2,3]],color)])
    return m

def track_mesh():
    s=np.linspace(-30,math.pi*20+30,260)
    m=strip(s,-24,24,-.04,(99,128,89))+strip(s,-11,11,-.02,(168,157,134))+strip(s,-7,7,0,(59,64,70))
    for sign in [-1,1]:
        m+=strip(s,min(sign*6.9,sign*7),max(sign*6.9,sign*7),.00001,(245,245,239))
        for i in range(len(s)-1): m+=strip(s[i:i+2],min(sign*7,sign*7.8),max(sign*7,sign*7.8),.003,(190,35,37) if i%4<2 else (230,230,223))
    return m

def clip_near(v,near=.2):
    out=[]
    for a,b in zip(v,np.roll(v,-1,axis=0)):
        if a[2]>=near: out.append(a)
        if (a[2]>=near)!=(b[2]>=near): out.append(a+(b-a)*(near-a[2])/(b[2]-a[2]))
    return np.array(out)

def render(mesh,cam,base=None):
    if base is None: rgb=np.full((H,W,3),(164,190,207),np.uint8); depth=np.full((H,W),np.inf)
    else: rgb,depth=[a.copy() for a in base]
    rot=np.array(cam['R_world_to_camera']); K=np.array(cam['K'])
    for vertices,color in mesh:
        normal=np.cross(vertices[1]-vertices[0],vertices[2]-vertices[0]); normal/=max(np.linalg.norm(normal),1e-12)
        shade=.65+.35*abs(normal@np.array([.3,-.4,.866]))
        col=np.clip(np.array(color)*shade,0,255).astype(np.uint8)
        poly=clip_near((vertices-cam['position_m'])@rot.T)
        for k in range(1,len(poly)-1):
            v=poly[[0,k,k+1]]; h=v@K.T; uv=h[:,:2]/h[:,2,None]
            xmin,ymin=np.maximum(np.floor(uv.min(0)),[0,0]).astype(int); xmax,ymax=np.minimum(np.ceil(uv.max(0)),[W-1,H-1]).astype(int)
            if xmin>xmax or ymin>ymax: continue
            a,b,c=uv; den=(b[1]-c[1])*(a[0]-c[0])+(c[0]-b[0])*(a[1]-c[1])
            if abs(den)<1e-10: continue
            yy,xx=np.mgrid[ymin:ymax+1,xmin:xmax+1]; xx=xx+.5; yy=yy+.5
            u=((b[1]-c[1])*(xx-c[0])+(c[0]-b[0])*(yy-c[1]))/den
            w=((c[1]-a[1])*(xx-c[0])+(a[0]-c[0])*(yy-c[1]))/den; t=1-u-w
            iz=u/v[0,2]+w/v[1,2]+t/v[2,2]; z=1/np.maximum(iz,1e-15)
            view=depth[ymin:ymax+1,xmin:xmax+1]; mask=(u>=0)&(w>=0)&(t>=0)&(z<view)
            view[mask]=z[mask]; rgb[ymin:ymax+1,xmin:xmax+1][mask]=col
    return rgb,depth

def events(rows):
    out=[]; start=None
    for i in range(len(rows)+1):
        active=i<len(rows) and rows[i]['is_violation']
        if active and start is None: start=i
        if not active and start is not None:
            out.append(dict(start_frame=start,end_frame_inclusive=i-1,start_time_s=start/FPS,end_time_exclusive_s=i/FPS,frame_count=i-start,peak_margin_m=max(r['min_excess_m'] for r in rows[start:i]))); start=None
    return out

def save_json(path,data): path.write_text(json.dumps(data,indent=2))

def main():
    global S
    report={'geometry':[],'cameras':{},'scenarios':{}}
    for s in [-20,0,15,35,60,80]:
        c,n=center(s); width=np.linalg.norm((c+n*7)-(c-n*7)); gap=np.linalg.norm((c+n*7)-(c+n*(6.9+.1)))
        assert abs(width-14)<1e-12 and gap<1e-12
        report['geometry'].append(dict(s_m=s,width_m=float(width),boundary_offset_m=7,white_line_inner_offset_m=6.9,boundary_outer_edge_gap_m=float(gap)))
    # Explicit line semantics and all-four regression controls.
    controls={str(y):float(distance(np.array([[-10.,y]]))[0]-7) for y in [6.9,6.95,7.,7.05]}
    assert controls['6.95']<0 and controls['7.0']==0 and controls['7.05']>0
    report['line_controls_excess_m']=controls
    track=track_mesh(); bases={}
    for cam in CAMS:
        s=(np.arange(20000)+.5)/20000*(math.pi*20); c,n=center(s)
        p=np.concatenate([c-n*7,c+n*7]); world=np.column_stack([p,np.zeros(len(p))]); uv,d=project(world,cam)
        hom=np.array(cam['ground_plane_homography']); hh=np.column_stack([p,np.ones(len(p))])@hom.T; uvh=hh[:,:2]/hh[:,2,None]
        valid=d>.2; err=np.max(np.abs(uv[valid]-uvh[valid])); assert err<1e-7
        inv=np.column_stack([uv[valid],np.ones(valid.sum())])@np.linalg.inv(hom).T
        back=np.max(np.linalg.norm(inv[:,:2]/inv[:,2,None]-p[valid],axis=1)); assert back<1e-8
        visible=valid&(uv[:,0]>=0)&(uv[:,0]<W)&(uv[:,1]>=0)&(uv[:,1]<H)
        # Each boundary sampled uniformly in angle; lengths differ by radius.
        cov=(visible[:20000].mean()*47+visible[20000:].mean()*33)/80*100
        cam['boundary_coverage_percent']=float(cov)
        report['cameras'][cam['name']]=dict(projection_max_error_px=float(err),backprojection_max_error_m=float(back),coverage_percent=float(cov))
        bases[cam['name']]=render(track,cam)
        Image.fromarray(bases[cam['name']][0]).save(OUT/(cam['name']+'_track.png'))
    save_json(OUT/'calibration.json',CAMS)
    print('GEOMETRY/CAMERAS',json.dumps(report),flush=True)
    scenarios={'clean_lap':[np.full(N,-6.)]}
    for margin in [.05,.15,.30,.60]: scenarios[f'violation_{margin:.2f}m']=[profile(margin)]
    scenarios['near_miss']=[np.full(N,-.35)]
    scenarios['side_by_side']=[profile(.30),np.full(N,-2.4)]
    blip=np.full(N,-.35); blip[47:49]=.30
    sustained=np.full(N,-.35); sustained[35:65]=.30
    scenarios['sustained_vs_blip']=[blip,sustained]
    model=car_mesh(); obj=[]
    for i,(v,col) in enumerate(model):
        obj.extend('v '+' '.join(map(str,p)) for p in v); obj.append(f'f {i*3+1} {i*3+2} {i*3+3}')
    (OUT/'car.obj').write_text('\n'.join(obj))
    for name,profiles in scenarios.items():
        folder=OUT/name; folder.mkdir(exist_ok=True); cars=[]; allrows=[]; summary={}
        for ci,desired in enumerate(profiles):
            S=8+12*T+(8 if name=='sustained_vs_blip' and ci==1 else 0)
            p,h,q,e=solve(desired); rows=[]
            for i in range(N):
                rows.append(dict(frame_idx=i,t=float(T[i]),car_id=f'car_{ci+1}',world_position=p[i].tolist(),heading_rad=float(h[i]),contact_points_world=dict(zip(KEYS,q[i].tolist())),corner_excess_m=dict(zip(KEYS,e[i].tolist())),max_excess_m=float(e[i].max()),min_excess_m=float(e[i].min()),is_violation=bool(e[i].min()>0),speed_mps=float(np.linalg.norm(np.gradient(p,1/FPS,axis=0,edge_order=2)[i]))))
            assert all(not r['is_violation'] or desired[i]>-1.5 for i,r in enumerate(rows))
            assert abs(e.min(1).max()-desired.max())<1e-8
            if name in ['near_miss','clean_lap']: assert not any(r['is_violation'] for r in rows)
            if name=='near_miss': assert all(r['max_excess_m']>0 for r in rows)
            summary[f'car_{ci+1}']=dict(events=events(rows),peak_target_error_m=float(abs(e.min(1).max()-desired.max())),violation_frames=sum(r['is_violation'] for r in rows),peak_margin_m=float(e.min(1).max()))
            cars.append((p,h,q,e,car_mesh((225,42,43) if ci==0 else (30,150,225)))); allrows+=rows
        save_json(folder/'ground_truth.json',dict(scenario=name,fps=FPS,frames=sorted(allrows,key=lambda r:(r['frame_idx'],r['car_id'])),events=summary))
        report['scenarios'][name]=summary
        print(name,json.dumps(summary),flush=True)
        for cam in CAMS:
            proc=subprocess.Popen(['ffmpeg','-y','-loglevel','error','-f','rawvideo','-pix_fmt','rgb24','-s',f'{W}x{H}','-r',str(FPS),'-i','-','-an','-c:v','libx264','-crf','22','-pix_fmt','yuv420p',str(folder/(cam['name']+'.mp4'))],stdin=subprocess.PIPE)
            for i in range(N):
                mesh=[]
                for p,h,q,e,m in cars:
                    co,si=np.cos(h[i]),np.sin(h[i]); rot=np.array([[co,-si,0],[si,co,0],[0,0,1]])
                    mesh.extend((v@rot.T+[*p[i],0],col) for v,col in m)
                rgb,dep=render(mesh,cam,bases[cam['name']]); proc.stdin.write(rgb.tobytes())
                if i==48:
                    Image.fromarray(rgb).save(folder/(cam['name']+'_frame.png'))
                    im=Image.fromarray(rgb); draw=ImageDraw.Draw(im)
                    s=np.linspace(-30,93,1000); c,n=center(s)
                    for sign in [-1,1]:
                        boundary=np.column_stack([c+sign*n*7,np.full(len(c),.015)]); uv,d=project(boundary,cam)
                        for j in range(len(uv)-1):
                            if min(d[j:j+2])>.2 and np.max(np.abs(uv[j:j+2]))<10000: draw.line([tuple(uv[j]),tuple(uv[j+1])],fill=(255,210,0),width=2)
                    for ci,(p,h,q,e,m) in enumerate(cars):
                        uv,d=project(np.column_stack([q[i],np.zeros(4)]),cam)
                        for j,((x,y),z) in enumerate(zip(uv,d)):
                            if z>.2 and 0<=x<W and 0<=y<H: draw.ellipse((x-4,y-4,x+4,y+4),fill=(255,40,60) if e[i,j]>0 else (30,255,120))
                        draw.text((10,12+ci*17),f'car_{ci+1} min={e[i].min():+.6f}m max={e[i].max():+.6f}m',fill='white')
                    draw.text((10,H-20),'Debug: yellow=outer line edge; red=outside; green=inside',fill='white')
                    im.save(folder/(cam['name']+'_debug.png'))
            proc.stdin.close(); assert proc.wait()==0
    save_json(OUT/'verification.json',report)
    print('COMPLETE',flush=True)

if __name__=='__main__': main()
