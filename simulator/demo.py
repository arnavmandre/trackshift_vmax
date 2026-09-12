"""Render a fresh, unscored 1080p60 demo collection, independent of training.

Usage: python -m simulator.demo --out PATH
"""
import argparse
import json
import os
from pathlib import Path
import subprocess
import time
import numpy as np
from PIL import Image, ImageDraw
import geometry as g
from .appearance import surface
from .generate import make_specs, trajectory
from .evidence import sha256


def scenery():
    ground = np.array([[-200,-200,-.06],[250,-200,-.06],[250,250,-.06],[-200,250,-.06]])
    mesh = [(ground[[0,1,2]],(99,128,89)),(ground[[0,2,3]],(99,128,89))] + g.track_mesh()
    # Trackside rails and alternating crash barriers, outside the runoff.
    stations = np.linspace(-25,85,90)
    for side in [-1,1]:
        centres, normals = g.center(stations)
        points = centres + normals*side*13
        for i in range(len(points)-1):
            a,b = points[i],points[i+1]
            for z in [.55,.85]:
                mesh += g.rod([*a,z],[*b,z],.065,(153,164,170),6)
            if i%3 == 0:
                mesh += g.box([*a,.45],[.12,.12,.9],(115,124,128))
    # Sparse tree line beyond the barriers gives the circuit depth and scale.
    for station in np.linspace(-20,90,22):
        centre,normal = g.center(station)
        p = centre+normal*22
        mesh += g.rod([*p,0],[*p,2.2],.18,(95,75,51),8)
        for z,r in [(1.4,1.45),(2.2,1.15),(3.,.8)]:
            ring = np.array([[p[0]+r*np.cos(a),p[1]+r*np.sin(a),z] for a in np.arange(10)*2*np.pi/10])
            tip = np.array([*p,z+1.7])
            mesh.extend((np.array([ring[i],ring[(i+1)%10],tip]),(48,91,58)) for i in range(10))
    return mesh


def vehicle(colour, heights):
    mesh = g.car_mesh(colour, heights)
    # Small geometric details improve silhouettes without moving tyre contacts.
    mesh += g.box((-.1,0,.568),(2.6,.16,.018),(237,238,228))
    for side in [-1,1]:
        mesh += g.box((-.6,side*.57,.39),(1.7,.025,.18),(27,35,45))
        mesh += g.box((-2.43,side*.93,.79),(.7,.055,.45),colour)
        mesh += g.box((2.5,side*.96,.24),(.65,.055,.28),colour)
        mesh += g.rod((.05,side*.35,.65),(.12,side*.65,.7),.025,(40,45,50),6)
        mesh += g.box((.12,side*.67,.71),(.19,.12,.1),colour)
    mesh += g.rod((-.45,0,.74),(-.45,0,.98),.17,(246,189,43),16)
    mesh += g.box((-.29,0,.89),(.025,.25,.065),(28,44,61))
    return mesh


def frame_shadow(base, world, valid, rows, cam):
    rgb = base[0].copy()
    for row in rows:
        p = np.asarray(row['world_position'])
        corners = np.array([[p[0]+x,p[1]+y,z] for x in [-4,4] for y in [-4,4] for z in [0,.1]])
        uv, depth = g.project(corners,cam)
        if np.any(depth <= .2):continue
        left,top = np.maximum(np.floor(uv.min(0)).astype(int),0)
        right,bottom = np.minimum(np.ceil(uv.max(0)).astype(int),cam['resolution'])
        if right<=left or bottom<=top:continue
        delta = world[top:bottom,left:right,:2]-p
        c,s = np.cos(row['heading_rad']),np.sin(row['heading_rad'])
        x,y = delta[...,0]*c+delta[...,1]*s, -delta[...,0]*s+delta[...,1]*c
        shadow = .4*np.exp(-1.5*((x/2.35)**4+(y/.95)**4))*valid[top:bottom,left:right]
        rgb[top:bottom,left:right] = (rgb[top:bottom,left:right]*(1-shadow[...,None])).astype(np.uint8)
    return rgb,base[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out',required=True)
    parser.add_argument('--seed',type=int,default=2026091307)
    parser.add_argument('--frames',type=int,default=240)
    args = parser.parse_args()
    if args.frames < 2:parser.error('--frames must be at least 2')
    dest = Path(args.out)
    dest.mkdir(parents=True,exist_ok=False)
    (dest/'sealed').mkdir()
    g.W,g.H = 1920,1080
    specs = make_specs(8,args.seed,60,args.frames/60,1920,1080,1)
    styles = [('scarlet_run',(211,37,44),6.5,1,[22,-26,13]),
              ('gold_kerb',(240,179,32),8.1,1,[-24,-23,10]),
              ('blue_duel',(32,126,222),8.6,2,[18,-28,18]),
              ('mint_sweep',(33,187,151),7.7,1,[-26,-19,16]),
              ('violet_chase',(143,76,210),8.3,2,[24,-25,11]),
              ('orange_apex',(239,105,28),7.2,1,[-22,-27,12]),
              ('pearl_pair',(223,229,238),8.8,2,[16,-30,20]),
              ('lime_exit',(153,211,47),6.8,1,[-28,-21,14])]
    track = scenery()
    manifest = dict(kind='fresh unscored synthetic demo; not an untouched blind benchmark after viewing',
                    seed=args.seed,width=1920,height=1080,fps=60,render_resolution=[2400,1350],clips=[],
                    limitations=['fixed VMAX corner','simplified car and suspension','approximate contact shadows','no real-world accuracy claim'])
    gallery = []
    started = time.perf_counter()
    for index,(spec,(name,colour,peak,cars,offset)) in enumerate(zip(specs,styles)):
        spec.update(speed_mps=6.5+(index%3)*.5,start_s=10.+index*1.5,offset_base_m=5.3,offset_peak_m=peak,
                    excursion_centre=.45+(index%3)*.05,excursion_width=.18+(index%3)*.03,cars=cars,split='demo')
        rows,_ = trajectory(spec)
        primary = [r for r in rows if r['car_id']=='car_1']
        aim = np.array([*primary[len(primary)//2]['world_position'],.15])
        cam = g.camera(name,aim+offset,aim,48)
        cam['fps']=60
        centres = np.array([[*r['world_position'],.4] for r in primary])
        uv,z = g.project(centres,cam)
        if not np.all((z>.2)&(uv[:,0]>40)&(uv[:,0]<1880)&(uv[:,1]>40)&(uv[:,1]<1040)):
            raise ValueError('camera does not cover complete primary trajectory')
        render_cam = dict(cam,resolution=[2400,1350],K=(np.diag([1.25,1.25,1])@np.asarray(cam['K'])).tolist())
        base,world,valid = surface(g.render(track,render_cam),render_cam)
        world = world.astype(np.float32)
        video = dest/f'{name}.mp4'
        command = [os.environ.get('FFMPEG','ffmpeg'),'-y','-loglevel','error','-f','rawvideo','-pix_fmt','rgb24',
                   '-s','1920x1080','-r','60','-i','-','-an','-c:v','libx264','-preset','fast','-crf','16',
                   '-pix_fmt','yuv420p','-movflags','+faststart',str(video)]
        proc = subprocess.Popen(command,stdin=subprocess.PIPE)
        try:
            for frame in range(args.frames):
                current = [r for r in rows if r['frame_idx']==frame]
                mesh = []
                for ci,row in enumerate(current):
                    c,s = np.cos(row['heading_rad']),np.sin(row['heading_rad'])
                    rotation = np.array([[c,-s,0],[s,c,0],[0,0,1]])
                    body = vehicle(colour if ci==0 else (231,233,240),[p[2] for p in row['contacts_world']])
                    mesh.extend((v@rotation.T+[*row['world_position'],0],col) for v,col in body)
                high = g.render(mesh,render_cam,frame_shadow(base,world,valid,current,render_cam))[0]
                rgb = np.asarray(Image.fromarray(high).resize((1920,1080),Image.Resampling.LANCZOS))
                proc.stdin.write(rgb.tobytes())
                if frame==args.frames//2:
                    Image.fromarray(rgb).save(dest/f'{name}.jpg',quality=95)
                    tile = Image.fromarray(rgb).resize((960,540))
                    ImageDraw.Draw(tile).text((16,16),name.replace('_',' ').upper(),fill='white')
                    gallery.append(tile)
        finally:
            proc.stdin.close()
            if proc.wait()!=0:raise RuntimeError('FFmpeg encoding failed')
        (dest/'sealed'/f'{name}.json').write_text(json.dumps(dict(spec=spec,frames=rows)))
        manifest['clips'].append(dict(video=video.name,sha256=sha256(video),camera=cam,frames=args.frames))
        (dest/'manifest.json').write_text(json.dumps(manifest,indent=2))
        print(f'{name} complete ({time.perf_counter()-started:.1f}s elapsed)',flush=True)
    sheet = Image.new('RGB',(1920,540*((len(gallery)+1)//2)))
    for i,tile in enumerate(gallery):sheet.paste(tile,((i%2)*960,(i//2)*540))
    sheet.save(dest/'gallery.jpg',quality=95)
    manifest['generation_seconds']=time.perf_counter()-started
    (dest/'manifest.json').write_text(json.dumps(manifest,indent=2))
    print('DEMO COMPLETE',flush=True)


if __name__=='__main__':main()
