"""Seeded scene generation with separate inference and sealed-label manifests.

Uses the existing VMAX corner and assets. Camera/trajectory/appearance diversity
is not unseen-track or unseen-asset generalization. CPU backend is deliberately
low fidelity; OpenGL retains the v2 renderer. Neither is photorealistic.

Each incident is rendered from a ring of cameras spaced evenly around it (a
360-degree view of the same event, not one random angle), so a boundary call
can be checked against whichever angle actually sees the tyre. All angles of
one incident share `family_id` and the same trajectory/ground truth; only the
camera differs.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import math
import time
import os
from pathlib import Path
import subprocess
import numpy as np
from PIL import Image, ImageEnhance, ImageFilter
import geometry as g
from .evidence import sha256


def make_specs(count, seed, fps=24, seconds=3, width=960, height=540, angles=6):
    if count < 1 or fps < 1 or seconds <= 0 or width < 64 or height < 64 or width%2 or height%2:
        raise ValueError('count/fps/duration positive; dimensions even and at least 64')
    if angles < 1:
        raise ValueError('angles must be at least 1')
    out=[]
    for i in range(count):
        family=hashlib.sha256(f'vmax-scene-v4:{seed}:{i}'.encode()).hexdigest()
        rng=np.random.default_rng(int(family[:16],16))
        # Assignment precedes rendering; all derivatives inherit the family.
        split=['train','validation','test'][0 if int(family[-8:],16)%100<70 else (1 if int(family[-8:],16)%100<85 else 2)]
        out.append(dict(id=family[:16],family_id=family,split=split,seed=int(family[:8],16),
            fps=fps,frames=max(3,round(seconds*fps)),width=width,height=height,angles=angles,
            speed_mps=float(rng.uniform(8,22)),start_s=float(rng.uniform(8,18)),
            offset_base_m=float(rng.uniform(3,6.5)),offset_peak_m=float(rng.uniform(6.8,9.2)),
            excursion_centre=float(rng.uniform(.3,.7)),excursion_width=float(rng.uniform(.05,.25)),
            cars=int(rng.choice([1,2])),same_livery=bool(rng.random()<.3),
            ring_radius=float(rng.uniform(16,24)),ring_height=float(rng.uniform(5,9)),ring_fov=float(rng.uniform(45,60)),
            colours=rng.integers(25,235,(2,3)).tolist(),brightness=float(rng.uniform(.7,1.2)),
            blur_px=float(rng.choice([0,0,.4,.8])),noise_sigma=float(rng.uniform(0,3)),
            crf=int(rng.integers(18,30)),telemetry_sigma_m=1.5,telemetry_latency_s=.15))
    for spec in out:
        midpoint = spec['start_s'] + spec['speed_mps'] * (spec['frames']-1) / (2*spec['fps'])
        centre, normal = g.center(midpoint)
        aim = centre - normal * 6
        spec['incident_centre'] = [float(aim[0]), float(aim[1]), 0.0]
    return out


def ring_cameras(incident_id, target, count, radius, height, fov):
    """A 360-degree ring of `count` cameras spaced evenly in azimuth around `target`."""
    cams=[]
    for k in range(count):
        theta = 2*math.pi*k/count
        pos = [target[0]+radius*math.cos(theta), target[1]+radius*math.sin(theta), height]
        cams.append(g.camera(f'{incident_id}_a{k:02d}', pos, target, fov))
    return cams


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
            contacts_3d=np.c_[contacts,g.kerb_height(excess)]
            # Independently sample rectangular contact patches in the simulator.
            xx,yy=np.meshgrid(np.linspace(-.06,.06,7),np.linspace(-.18,.18,19))
            patch=np.stack([xx.ravel(),yy.ravel()],axis=1)@rot.T
            patch_values=g.distance(contacts[:,None,:]+patch[None,:,:])-g.HALF
            covering_radius=math.sqrt(2)*.01
            footprint_lower=patch_values.min(axis=1)-covering_radius
            allrows.append(dict(frame_idx=i,time_s=float(t[i]),car_id=f'car_{ci+1}',world_position=p[i].tolist(),
                heading_rad=float(h[i]),contacts_world=contacts_3d.tolist(),point_margin_m=float(excess.min()),
                footprint_margin_m=float(footprint_lower.min()),speed_mps=float(np.linalg.norm(v[i])),
                footprint_is_violation=bool((footprint_lower>0).all()),point_is_violation=bool((excess>0).all())))
    return sorted(allrows,key=lambda r:(r['frame_idx'],r['car_id'])),travel


def _render_angle(cam,rows,byframe,travel,colours,track,renderer,spec,dest,rng,meshes=None):
    cid=cam['name'];folder=dest/cid;folder.mkdir()
    width,height=spec['width'],spec['height']
    main_rows=[row for row in rows if row['car_id']=='car_1']
    world=np.array([[*row['world_position'],0] for row in main_rows])
    uv,depth=g.project(world,cam)
    visible=(depth>.5)&(uv[:,0]>=0)&(uv[:,0]<width)&(uv[:,1]>=0)&(uv[:,1]<height)
    coverage=float(visible.mean())
    if coverage<.5:
        raise ValueError(f'incident {spec["id"]} angle {cid} has insufficient projected car coverage ({coverage:.0%}); adjust the ring radius/height or use another seed')
    render_cam=dict(cam)
    scale=spec.get('supersample',1)
    render_cam['resolution']=[width*scale,height*scale]
    render_cam['K']=(np.diag([scale,scale,1])@np.asarray(cam['K'])).tolist()
    if not renderer:
        base=g.render(track,render_cam)
        if spec.get('appearance') == 'enhanced':
            from .appearance import surface, shadows
            base, surface_world, surface_valid = surface(base, render_cam)
    video=folder/'original.mp4'
    cmd=[os.environ.get('FFMPEG','ffmpeg'),'-y','-loglevel','error','-f','rawvideo','-pix_fmt','rgb24','-s',f'{width}x{height}','-r',str(spec['fps']),'-i','-','-an','-c:v','libx264','-preset','fast','-crf',str(spec['crf']),'-pix_fmt','yuv420p','-movflags','+faststart',str(video)]
    proc=subprocess.Popen(cmd,stdin=subprocess.PIPE)
    try:
        for i,current in byframe.items():
            if renderer:
                im=renderer.frame(cam,current,[d[i] for d in travel])
            else:
                frame_base = shadows(base, surface_world, surface_valid, current) if spec.get('appearance') == 'enhanced' else base
                im=Image.fromarray(g.render(meshes[i],render_cam,frame_base)[0])
                if scale>1:im=im.resize((width,height),Image.Resampling.LANCZOS)
            im=ImageEnhance.Brightness(im).enhance(spec['brightness'])
            if spec['blur_px']:im=im.filter(ImageFilter.GaussianBlur(spec['blur_px']))
            pixels=np.asarray(im).astype(float)+rng.normal(0,spec['noise_sigma'],(height,width,3))
            pixels=np.clip(pixels,0,255).astype('uint8');proc.stdin.write(pixels.tobytes())
            if i==spec['frames']//2:Image.fromarray(pixels).save(folder/'thumbnail.jpg')
    finally:
        proc.stdin.close()
        if proc.wait()!=0:raise RuntimeError('FFmpeg failed to encode scene')
    return video,coverage


def generate(destination,count=10,seed=0,fps=24,seconds=3,width=960,height=540,backend='cpu',plan_only=False,angles=6,supersample=2,appearance='classic'):
    if supersample not in (1,2,3):raise ValueError('supersample must be 1, 2 or 3')
    if backend!='cpu' and supersample!=1:raise ValueError('supersampling currently requires CPU backend')
    if appearance not in ('classic','enhanced'):raise ValueError('unknown appearance')
    if appearance == 'enhanced' and backend != 'cpu':raise ValueError('enhanced appearance requires CPU backend')
    started=time.perf_counter()
    dest=Path(destination)
    if (dest/'manifest.json').exists() or (dest/'sealed'/'labels.json').exists():
        raise ValueError('destination already contains a dataset; choose a new directory')
    dest.mkdir(parents=True,exist_ok=True); (dest/'sealed').mkdir(exist_ok=True)
    specs=make_specs(count,seed,fps,seconds,width,height,angles)
    for spec in specs:
        spec['supersample']=supersample
        spec['appearance']=appearance
    (dest/'sealed'/'scenes.json').write_text(json.dumps(specs,indent=2))
    if plan_only:
        print(f'{count} incidents x {angles} angles reproducible; videos not rendered')
        return
    g.W,g.H=width,height
    renderer=None
    if backend=='opengl':
        import render as r
        r.WIDTH,r.HEIGHT=width,height;g.W,g.H=width,height
        renderer=r.Renderer()
    track=g.track_mesh() if renderer is None else None
    manifest={'schema_version':3,'generator':{'width':width,'height':height,'fps':fps,'seconds':seconds,'supersample':supersample,'backend':backend,'appearance':appearance,'kerb_height_m':g.KERB_HEIGHT_M},'dataset_kind':'synthetic_fixed_vmax_corner','seed':seed,'angles':angles,
              'limitations':['same track and vehicle geometry as development set','not a real-footage benchmark'], 'clips':[]}
    truth={'schema_version':1,'clips':{}}
    for k,spec in enumerate(specs):
        rng=np.random.default_rng(spec['seed']); tel_rng=np.random.default_rng(spec['seed']+1)
        rows,travel=trajectory(spec)
        colours=spec['colours']; colours[1]=colours[0] if spec['same_livery'] else colours[1]
        byframe={i:[row for row in rows if row['frame_idx']==i] for i in range(spec['frames'])}
        # Geometry depends on the incident/frame, never on the camera. Build once
        # and reuse across every angle, including independently elevated wheels.
        meshes={}
        if renderer is None:
            for i,current in byframe.items():
                mesh=[]
                for ci,row in enumerate(current):
                    c,s=np.cos(row['heading_rad']),np.sin(row['heading_rad'])
                    rot=np.array([[c,-s,0],[s,c,0],[0,0,1]])
                    body=g.car_mesh(tuple(colours[ci]),[pt[2] for pt in row['contacts_world']])
                    mesh.extend((v@rot.T+[*row['world_position'],0],col) for v,col in body)
                meshes[i]=mesh
        tel=[]
        for row in rows:
            if row['frame_idx']%max(1,round(fps/5))==0 and tel_rng.random()>.1:
                tel.append(dict(time_s=row['time_s']+.15,car_id=row['car_id'],
                    position_m=(np.array(row['world_position'])+tel_rng.normal(0,1.5,2)).tolist(),
                    sigma_m=1.5,source='synthetic noisy position',latency_s=.15))
        telemetry_path=f'{spec["id"]}_telemetry.json'
        (dest/telemetry_path).write_text(json.dumps(tel))
        if renderer:
            renderer.bodies=[renderer.upload(r.body(tuple(col))) for col in colours]
        cams=ring_cameras(spec['id'],spec['incident_centre'],spec['angles'],spec['ring_radius'],spec['ring_height'],spec['ring_fov'])
        for cam in cams:
            cam['fps']=fps
            video,coverage=_render_angle(cam,rows,byframe,travel,colours,track,renderer,spec,dest,rng,meshes)
            cid=cam['name']
            entry=dict(id=cid,family_id=spec['family_id'],split=spec['split'],video=f'{cid}/original.mp4',
                video_sha256=sha256(video),fps=fps,frames=spec['frames'],projected_primary_centre_coverage=coverage,
                thumbnail=f'{cid}/thumbnail.jpg',angle_index=int(cid.rsplit("_a",1)[1]),angle_count=spec['angles'],
                calibration={'homography':cam['ground_plane_homography'],'sigma_m':0.0,'source':'exact synthetic camera; oracle benchmark'},
                camera_spec=cam,telemetry=telemetry_path)
            manifest['clips'].append(entry);truth['clips'][cid]={'frames':rows,'video_sha256':entry['video_sha256']}
        print(f'{k+1}/{count} {spec["id"]} {spec["split"]} ({spec["angles"]} angles)',flush=True)
    manifest['generation_seconds']=time.perf_counter()-started
    (dest/'manifest.json').write_text(json.dumps(manifest,indent=2))
    (dest/'sealed'/'labels.json').write_text(json.dumps(truth))
    return manifest


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out',required=True);p.add_argument('--count',type=int,default=10);p.add_argument('--seed',type=int,default=0)
    p.add_argument('--fps',type=int,default=24);p.add_argument('--seconds',type=float,default=3)
    p.add_argument('--width',type=int,default=960);p.add_argument('--height',type=int,default=540)
    p.add_argument('--supersample',type=int,choices=[1,2,3],default=2)
    p.add_argument('--angles',type=int,default=6,help='cameras spaced around the 360-degree ring per incident')
    p.add_argument('--backend',choices=['cpu','opengl'],default='cpu');p.add_argument('--plan-only',action='store_true')
    p.add_argument('--appearance',choices=['classic','enhanced'],default='classic')
    a=p.parse_args(argv);generate(a.out,a.count,a.seed,a.fps,a.seconds,a.width,a.height,a.backend,a.plan_only,a.angles,a.supersample,a.appearance)

if __name__=='__main__':main()
