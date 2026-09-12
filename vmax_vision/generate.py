"""Seeded scene generation with separate inference and sealed-label manifests.

Uses the existing VMAX corner and assets. Camera/trajectory/appearance diversity
is not unseen-track or unseen-asset generalization. CPU backend is deliberately
low fidelity; OpenGL retains the v2 renderer. Neither is photorealistic.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import math
from pathlib import Path
import subprocess
import numpy as np
from PIL import Image, ImageEnhance, ImageFilter
import geometry as g
from .evidence import sha256


def make_specs(count, seed, fps=24, seconds=3, width=640, height=360):
    if count < 1 or fps < 1 or seconds <= 0 or width < 64 or height < 64 or width%2 or height%2:
        raise ValueError('count/fps/duration positive; dimensions even and at least 64')
    out=[]
    for i in range(count):
        family=hashlib.sha256(f'vmax-scene-v4:{seed}:{i}'.encode()).hexdigest()
        rng=np.random.default_rng(int(family[:16],16))
        # Assignment precedes rendering; all derivatives inherit the family.
        split=['train','validation','test'][0 if int(family[-8:],16)%100<70 else (1 if int(family[-8:],16)%100<85 else 2)]
        out.append(dict(id=family[:16],family_id=family,split=split,seed=int(family[:8],16),
            fps=fps,frames=max(3,round(seconds*fps)),width=width,height=height,
            speed_mps=float(rng.uniform(8,22)),start_s=float(rng.uniform(8,18)),
            offset_base_m=float(rng.uniform(3,6.5)),offset_peak_m=float(rng.uniform(6.8,9.2)),
            excursion_centre=float(rng.uniform(.3,.7)),excursion_width=float(rng.uniform(.05,.25)),
            cars=int(rng.choice([1,2])),same_livery=bool(rng.random()<.3),
            camera_position=[float(rng.uniform(52,70)),float(rng.uniform(-28,-12)),float(rng.uniform(10,22))],
            camera_target=[25,18,0],fov=float(rng.uniform(42,65)),
            colours=rng.integers(25,235,(2,3)).tolist(),brightness=float(rng.uniform(.7,1.2)),
            blur_px=float(rng.choice([0,0,.4,.8])),noise_sigma=float(rng.uniform(0,3)),
            crf=int(rng.integers(18,30)),telemetry_sigma_m=1.5,telemetry_latency_s=.15))
    for spec in out:
        midpoint = spec['start_s'] + spec['speed_mps'] * (spec['frames']-1) / (2*spec['fps'])
        centre, normal = g.center(midpoint)
        aim = centre - normal * 6
        spec['camera_target'] = [float(aim[0]),float(aim[1]),0.0]
    return out


def trajectory(spec):
    t=np.arange(spec['frames'])/spec['fps']; n=len(t); allrows=[]; travel=[]
    for ci in range(spec['cars']):
        ss=spec['start_s']+spec['speed_mps']*t+ci*9
        centre,normal=g.center(ss)
        u=np.arange(n)/max(n-1,1)
        pulse=np.exp(-.5*((u-spec['excursion_centre'])/spec['excursion_width'])**2)
        offset=np.full(n,3.0) if ci else spec['offset_base_m']+(spec['offset_peak_m']-spec['offset_base_m'])*pulse
        p=centre-normal*offset[:,None];v=np.gradient(p,1/spec['fps'],axis=0)
        h=np.arctan2(v[:,1],v[:,0]);dist=np.r_[0,np.cumsum(np.linalg.norm(np.diff(p,axis=0),axis=1))];travel.append(dist)
        for i in range(n):
            c,s=np.cos(h[i]),np.sin(h[i]);rot=np.array([[c,-s],[s,c]])
            contacts=g.CONTACT@rot.T+p[i];excess=g.distance(contacts)-g.HALF
            # Independently sample rectangular contact patches in the simulator.
            xx,yy=np.meshgrid(np.linspace(-.06,.06,7),np.linspace(-.18,.18,19))
            patch=np.stack([xx.ravel(),yy.ravel()],axis=1)@rot.T
            patch_values=g.distance(contacts[:,None,:]+patch[None,:,:])-g.HALF
            covering_radius=math.sqrt(2)*.01
            footprint_lower=patch_values.min(axis=1)-covering_radius
            allrows.append(dict(frame_idx=i,time_s=float(t[i]),car_id=f'car_{ci+1}',world_position=p[i].tolist(),
                heading_rad=float(h[i]),contacts_world=contacts.tolist(),point_margin_m=float(excess.min()),
                footprint_margin_m=float(footprint_lower.min()),speed_mps=float(np.linalg.norm(v[i])),
                footprint_is_violation=bool((footprint_lower>0).all()),point_is_violation=bool((excess>0).all())))
    return sorted(allrows,key=lambda r:(r['frame_idx'],r['car_id'])),travel


def generate(destination,count=10,seed=0,fps=24,seconds=3,width=640,height=360,backend='cpu',plan_only=False):
    dest=Path(destination)
    if (dest/'manifest.json').exists() or (dest/'sealed'/'labels.json').exists():
        raise ValueError('destination already contains a dataset; choose a new directory')
    dest.mkdir(parents=True,exist_ok=True); (dest/'sealed').mkdir(exist_ok=True)
    specs=make_specs(count,seed,fps,seconds,width,height)
    (dest/'sealed'/'scenes.json').write_text(json.dumps(specs,indent=2))
    if plan_only:
        print(f'{count} reproducible scene specifications written; videos not rendered')
        return
    g.W,g.H=width,height
    renderer=None
    if backend=='opengl':
        import render as r
        r.WIDTH,r.HEIGHT=width,height;g.W,g.H=width,height
        renderer=r.Renderer()
    track=g.track_mesh() if renderer is None else None
    manifest={'schema_version':1,'dataset_kind':'synthetic_fixed_vmax_corner','seed':seed,
              'limitations':['same track and vehicle geometry as development set','not a real-footage benchmark'], 'clips':[]}
    truth={'schema_version':1,'clips':{}}
    for k,spec in enumerate(specs):
        rng=np.random.default_rng(spec['seed']); cid=spec['id']; folder=dest/cid;folder.mkdir()
        rows,travel=trajectory(spec)
        cam=g.camera(cid,spec['camera_position'],spec['camera_target'],spec['fov']);cam['fps']=fps
        main_rows=[row for row in rows if row['car_id']=='car_1']
        world=np.array([[*row['world_position'],0] for row in main_rows])
        uv,depth=g.project(world,cam)
        visible=(depth>.5)&(uv[:,0]>=0)&(uv[:,0]<width)&(uv[:,1]>=0)&(uv[:,1]<height)
        coverage=float(visible.mean())
        if coverage<.5:
            raise ValueError(f'scene {cid} has insufficient projected car coverage ({coverage:.0%}); use another seed or shorter duration')
        byframe={i:[row for row in rows if row['frame_idx']==i] for i in range(spec['frames'])}
        colours=spec['colours']; colours[1]=colours[0] if spec['same_livery'] else colours[1]
        if renderer:
            for obj in renderer.bodies:
                for handle in obj:handle.release()
            renderer.bodies=[renderer.upload(r.body(tuple(col))) for col in colours]
        else:
            base=g.render(track,cam);meshes=[g.car_mesh(tuple(c)) for c in colours]
        video=folder/'original.mp4'
        cmd=['ffmpeg','-y','-loglevel','error','-f','rawvideo','-pix_fmt','rgb24','-s',f'{width}x{height}','-r',str(fps),'-i','-','-an','-c:v','libx264','-preset','fast','-crf',str(spec['crf']),'-pix_fmt','yuv420p','-movflags','+faststart',str(video)]
        proc=subprocess.Popen(cmd,stdin=subprocess.PIPE)
        try:
            for i,current in byframe.items():
                if renderer:
                    im=renderer.frame(cam,current,[d[i] for d in travel])
                else:
                    mesh=[]
                    for ci,row in enumerate(current):
                        c,s=np.cos(row['heading_rad']),np.sin(row['heading_rad']);rot=np.array([[c,-s,0],[s,c,0],[0,0,1]])
                        mesh.extend((vertices@rot.T+[*row['world_position'],0],col) for vertices,col in meshes[ci])
                    im=Image.fromarray(g.render(mesh,cam,base)[0])
                im=ImageEnhance.Brightness(im).enhance(spec['brightness'])
                if spec['blur_px']:im=im.filter(ImageFilter.GaussianBlur(spec['blur_px']))
                pixels=np.asarray(im).astype(float)+rng.normal(0,spec['noise_sigma'],(height,width,3))
                pixels=np.clip(pixels,0,255).astype('uint8');proc.stdin.write(pixels.tobytes())
                if i==spec['frames']//2:Image.fromarray(pixels).save(folder/'thumbnail.jpg')
        finally:
            proc.stdin.close()
            if proc.wait()!=0:raise RuntimeError('FFmpeg failed to encode scene')
        # Telemetry is noisy context, separate from private exact labels.
        tel=[]
        for row in rows:
            if row['frame_idx']%max(1,round(fps/5))==0 and rng.random()>.1:
                tel.append(dict(time_s=row['time_s']+.15,car_id=row['car_id'],
                    position_m=(np.array(row['world_position'])+rng.normal(0,1.5,2)).tolist(),
                    sigma_m=1.5,source='synthetic noisy position',latency_s=.15))
        (folder/'telemetry.json').write_text(json.dumps(tel))
        entry=dict(id=cid,family_id=spec['family_id'],split=spec['split'],video=f'{cid}/original.mp4',
            video_sha256=sha256(video),fps=fps,frames=spec['frames'],projected_primary_centre_coverage=coverage,thumbnail=f'{cid}/thumbnail.jpg',
            calibration={'homography':cam['ground_plane_homography'],'sigma_m':0.0,'source':'exact synthetic camera; oracle benchmark'},
            camera_spec=cam,telemetry=f'{cid}/telemetry.json')
        manifest['clips'].append(entry);truth['clips'][cid]={'frames':rows,'video_sha256':entry['video_sha256']}
        print(f'{k+1}/{count} {cid} {spec["split"]}',flush=True)
    (dest/'manifest.json').write_text(json.dumps(manifest,indent=2))
    (dest/'sealed'/'labels.json').write_text(json.dumps(truth))
    return manifest


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out',required=True);p.add_argument('--count',type=int,default=10);p.add_argument('--seed',type=int,default=0)
    p.add_argument('--fps',type=int,default=24);p.add_argument('--seconds',type=float,default=3)
    p.add_argument('--width',type=int,default=640);p.add_argument('--height',type=int,default=360)
    p.add_argument('--backend',choices=['cpu','opengl'],default='cpu');p.add_argument('--plan-only',action='store_true')
    a=p.parse_args(argv);generate(a.out,a.count,a.seed,a.fps,a.seconds,a.width,a.height,a.backend,a.plan_only)

if __name__=='__main__':main()
