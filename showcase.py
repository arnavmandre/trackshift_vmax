"""60 fps race-pace showcase, with per-frame camera calibration and telemetry."""
import json, subprocess, argparse
import numpy as np
from scipy.optimize import brentq
from PIL import ImageDraw, ImageFont
import render as r
g=r.g

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--preview',action='store_true');args=ap.parse_args()
    fps=60;count=240;t=np.arange(count)/fps
    # Scripted brake-into-corner / accelerate-out speed schedule, in metres/sec.
    progress_speed=32-12*np.sin(np.pi*t/4)**2+5*t/4
    progress=-24+np.r_[0,np.cumsum((progress_speed[1:]+progress_speed[:-1])/(2*fps))]
    g.FPS=fps;g.N=count;g.T=t;g.S=progress
    pulse=np.exp(-((t-2.15)/1.1)**2)
    amplitude=brentq(lambda a:g.pose(2+a*pulse)[3].min(1).max()-.15,0,12,xtol=1e-12)
    pos,heading,contact,excess=g.pose(2+amplitude*pulse)
    velocity=np.gradient(pos,1/fps,axis=0,edge_order=2);acc=np.gradient(velocity,1/fps,axis=0,edge_order=2)
    speed=np.linalg.norm(velocity,axis=1);f=velocity/speed[:,None];l=np.column_stack([-f[:,1],f[:,0]])
    lateral=(acc*l).sum(1)/9.80665;longitudinal=(acc*f).sum(1)
    distance=np.r_[0,np.cumsum(np.linalg.norm(np.diff(pos,axis=0),axis=1))]
    curvature=np.gradient(np.unwrap(heading))/np.maximum(np.gradient(distance),1e-9)
    rows=[]
    for i in range(count):
        # Moving presentation camera. Dataset cameras remain fixed.
        eye=pos[i]+f[i]*8.6-l[i]*7.2;target=pos[i]+f[i]*.35
        cam=g.camera('tracking_showcase',[*eye,3.1],[*target,.45],37)
        rows.append(dict(frame_idx=i,t=float(t[i]),car_id='car_1',world_position=pos[i].tolist(),heading_rad=float(heading[i]),contact_points_world=dict(zip(g.KEYS,contact[i].tolist())),corner_excess_m=dict(zip(g.KEYS,excess[i].tolist())),min_excess_m=float(excess[i].min()),max_excess_m=float(excess[i].max()),is_violation=bool(excess[i].min()>0),speed_mps=float(speed[i]),longitudinal_acceleration_mps2=float(longitudinal[i]),lateral_acceleration_g=float(lateral[i]),visual_steering_rad=float(np.arctan(3.6*curvature[i])),camera=cam))
    assert abs(excess.min(1).max()-.15)<1e-8
    assert abs(lateral).max()<3.0, 'Showcase exceeds chosen 3g kinematic demand budget'
    # Independently check OpenGL against each frame's pinhole camera.
    worst=0
    for row in rows:
        p=np.column_stack([list(row['contact_points_world'].values()),np.zeros(4)])
        uv,dep=g.project(p,row['camera']);clip=np.column_stack([p,np.ones(4)])@r.vp_camera(row['camera']).T
        ndc=clip[:,:3]/clip[:,3,None];gpu=np.column_stack([(ndc[:,0]+1)*r.WIDTH/2,(1-ndc[:,1])*r.HEIGHT/2])
        worst=max(worst,float(abs(uv-gpu).max()))
    assert worst<1e-8
    folder=r.OUT/'race_pace';folder.mkdir(exist_ok=True)
    data=dict(scenario='race_pace',fps=fps,frames=rows,events=g.events(rows),model='Kinematic scripted path; no tyre-force or aero solver',opengl_projection_error_px=worst)
    (folder/'ground_truth.json').write_text(json.dumps(data,indent=2))
    renderer=r.Renderer()
    if args.preview: indices=[129];proc=None
    else:
        indices=range(count)
        proc=subprocess.Popen(['ffmpeg','-y','-loglevel','error','-f','rawvideo','-pix_fmt','rgb24','-s',f'{r.WIDTH}x{r.HEIGHT}','-r',str(fps),'-i','-','-an','-c:v','libx264','-preset','fast','-crf','18','-pix_fmt','yuv420p','-movflags','+faststart',str(folder/'race_pace.mp4')],stdin=subprocess.PIPE)
    font='/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf'
    large=ImageFont.truetype(font,35);small=ImageFont.truetype(font,16);bold=ImageFont.truetype(font.replace('.ttf','-Bold.ttf'),23)
    for i in indices:
        row=rows[i];im=renderer.frame(row['camera'],[row],[distance[i]])
        raw=im.copy();d=ImageDraw.Draw(im)
        d.rounded_rectangle((28,26,370,104),radius=5,fill=(10,22,30));d.rectangle((28,26,34,104),fill=(242,48,45))
        d.text((49,36),'VMAX / RACE PACE',font=bold,fill='white');d.text((50,70),'SYNTHETIC MOTORSPORT SIMULATION',font=small,fill=(155,180,190))
        d.rounded_rectangle((28,r.HEIGHT-119,430,r.HEIGHT-27),radius=5,fill=(10,22,30))
        d.text((49,r.HEIGHT-110),f'{speed[i]*3.6:03.0f}',font=large,fill='white');d.text((130,r.HEIGHT-91),'km/h',font=small,fill=(162,183,193))
        d.text((210,r.HEIGHT-104),f'{abs(lateral[i]):.2f} g lateral',font=small,fill=(162,183,193))
        status='ALL FOUR OUT' if row['is_violation'] else 'WITHIN TRACK LIMITS'
        d.text((50,r.HEIGHT-64),status,font=small,fill=(255,82,71) if row['is_violation'] else (75,222,173))
        d.text((r.WIDTH-265,35),f'TYRE MARGIN {row["min_excess_m"]:+.3f} m',font=small,fill='white')
        if proc:proc.stdin.write(im.tobytes())
        if i==129:
            im.save(folder/'showcase.png');raw.save(folder/'car_detail.png');r.debug(raw,row['camera'],[row]).save(folder/'contact_debug.png')
    if proc:proc.stdin.close();assert proc.wait()==0
    print(json.dumps({'speed_range_kmh':[float(speed.min()*3.6),float(speed.max()*3.6)],'peak_lateral_g':float(abs(lateral).max()),'events':data['events'],'projection_error_px':worst}),flush=True)

if __name__=='__main__':main()
