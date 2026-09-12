"""New pretrained pose experiment; prior simulator/data reuse is disclosed."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
import time

import cv2
import numpy as np
from scipy.optimize import linear_sum_assignment

OUT=Path('experiment_out')
PLAN={'model':'yolo26n-pose.pt','epochs':20,'imgsz':416,'batch':8,'seed':12092026,
      'frame_stride':4,'thresholds':[.15,.3,.5],'blind_seed':2026091207,'blind_count':40,'blind_angles':4,
      'selection':'highest validation event F1, then lower margin P95',
      'keypoints':['front_left','front_right','rear_left','rear_right'],
      'initialization':'official COCO pretrained pose weights; no earlier VMAX weights',
      'limitations':['same synthetic corner/assets','exact synthetic camera calibration',
                     'projected contact labels do not establish actual tyre visibility',
                     'uncalibrated detector scores, no driver identity']}


def digest(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(p,obj):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    p.write_text(json.dumps(obj,indent=2,allow_nan=False))
def transform(points,H):
    a=np.c_[np.asarray(points),np.ones(len(points))]@np.asarray(H).T
    if np.any(np.abs(a[:,2])<1e-8):raise ValueError('singular point projection')
    return a[:,:2]/a[:,2:3]
def load_manifest(root):
    root=Path(root).resolve();m=json.loads((root/'manifest.json').read_text())
    seen={};families={};ids=set()
    for c in m['clips']:
        if c['id'] in ids:raise ValueError('duplicate clip')
        ids.add(c['id']);p=(root/c['video']).resolve()
        if not p.is_relative_to(root):raise ValueError('video escapes dataset root')
        if digest(p)!=c['video_sha256']:raise ValueError('video changed')
        for value,lookup in [(c['family_id'],families),(c['video_sha256'],seen)]:
            if value in lookup and lookup[value]!=c['split']:raise ValueError('split leakage')
            lookup[value]=c['split']
    return m


def prepare(root):
    """Export only train/validation images; no final-test input or annotations."""
    import geometry as sim
    root=Path(root);m=load_manifest(root)
    labels=json.loads((root/'sealed/labels.json').read_text())['clips']
    counts={}
    for split in ('train','validation'):
        count=0
        for c in m['clips']:
            if c['split']!=split:continue
            truth=labels[c['id']]
            if truth['video_sha256']!=c['video_sha256']:raise ValueError('label/video mismatch')
            frames={}
            for row in truth['frames']:frames.setdefault(row['frame_idx'],[]).append(row)
            cap=cv2.VideoCapture(str(root/c['video']));idx=0
            while True:
                ok,im=cap.read()
                if not ok:break
                if idx%PLAN['frame_stride']==0:
                    h,w=im.shape[:2];lines=[];camera=c['camera_spec']
                    for row in frames.get(idx,[]):
                        # Full 3D projection, not the flat ground-plane homography: a
                        # contact point elevated on a kerb must land where it actually
                        # appears, not where a flat-ground point at the same (x,y) would.
                        xy,contact_depth=sim.project(np.asarray(row['contacts_world']),camera)
                        if (contact_depth<=.5).any():continue
                        # Labels mark inferred/projected contacts, not confirmed visible tyres.
                        # Conservative approximate body extent for detection supervision.
                        pos=np.asarray(row['world_position']);angle=row['heading_rad']
                        rot=np.array([[np.cos(angle),-np.sin(angle)],[np.sin(angle),np.cos(angle)]])
                        corners=np.array([[x,y] for x in (-2.9,2.9) for y in (-1.1,1.1)])@rot.T+pos
                        vertices=np.array([[*p,z] for p in corners for z in (0,1.1)])
                        v=(vertices-np.array(camera['position_m']))@np.array(camera['R_world_to_camera']).T
                        proj=v@np.array(camera['K']).T
                        if (v[:,2]<=.5).any():continue
                        box=proj[:,:2]/proj[:,2:3]
                        lo=np.maximum(box.min(0),[0,0]);hi=np.minimum(box.max(0),[w-1,h-1])
                        if (hi-lo<3).any():continue
                        center=(lo+hi)/2/[w,h];size=(hi-lo)/[w,h]
                        k=[]
                        for x,y in xy:
                            inside=0<=x<w and 0<=y<h
                            k.extend([float(np.clip(x/w,0,1)) if inside else 0,
                                      float(np.clip(y/h,0,1)) if inside else 0,1 if inside else 0])
                        lines.append(' '.join(map(str,[0,*center,*size,*k])))
                    stem=f'{c["id"]}_{idx:04d}'
                    image=OUT/'dataset/images'/split/f'{stem}.jpg'
                    label=OUT/'dataset/labels'/split/f'{stem}.txt'
                    image.parent.mkdir(parents=True,exist_ok=True);label.parent.mkdir(parents=True,exist_ok=True)
                    if not cv2.imwrite(str(image),im):raise RuntimeError('image write failed')
                    label.write_text('\n'.join(lines));count+=1
                idx+=1
            cap.release()
            if idx!=c['frames']:raise ValueError('decoded frame count differs')
        if count==0:raise ValueError(f'empty {split}')
        counts[split]=count
    # No horizontal flip: preserve camera/track handedness during this first experiment.
    (OUT/'dataset.yaml').write_text(f'path: {(OUT/"dataset").resolve()}\ntrain: images/train\nval: images/validation\nkpt_shape: [4, 3]\nflip_idx: [1, 0, 3, 2]\nkpt_oks_sigmas: [0.05, 0.05, 0.05, 0.05]\nnames:\n  0: car\n')
    save(OUT/'dataset_provenance.json',{'manifest_sha256':digest(root/'manifest.json'),
        'sampled_images':counts,'train_families':[c['family_id'] for c in m['clips'] if c['split']=='train'],
        'validation_families':[c['family_id'] for c in m['clips'] if c['split']=='validation'],
        'reuse':'earlier prepared simulator clips; not newly captured event footage'})
    print('Dataset export:',counts,flush=True)


def train():
    import torch
    from ultralytics import YOLO
    device='cuda' if torch.cuda.is_available() else 'cpu'
    if device=='cpu':torch.set_num_threads(2)
    model=YOLO(PLAN['model'])
    save(OUT/'initialization.json',{'checkpoint':PLAN['model'],'sha256':digest(PLAN['model']),
        'device':device,'gpu':torch.cuda.get_device_name(0) if device=='cuda' else None})
    # Same fixed hyperparameters (epochs/imgsz/batch/seed) regardless of device; only
    # execution (device/workers/amp) adapts, so a local GPU run stays comparable to CI's.
    model.train(data=str(OUT/'dataset.yaml'),epochs=PLAN['epochs'],imgsz=PLAN['imgsz'],
        batch=PLAN['batch'],device=device,workers=(4 if device=='cuda' else 0),seed=PLAN['seed'],deterministic=True,
        optimizer='AdamW',lr0=.001,lrf=.05,patience=20,pretrained=True,
        mosaic=.3,close_mosaic=5,mixup=0,fliplr=0,flipud=0,degrees=5,translate=.1,
        scale=.3,perspective=0,hsv_h=.1,hsv_s=.4,hsv_v=.3,amp=(device=='cuda'),cache=False,
        project=str(OUT.resolve()),name='training',exist_ok=False,plots=False,save=True,verbose=False)
    actual=Path(model.trainer.save_dir)
    expected=OUT/'training'
    if actual.resolve()!=expected.resolve():
        expected.mkdir(parents=True,exist_ok=True)
        shutil.copytree(actual,expected,dirs_exist_ok=True)
    for name in ('best.pt','last.pt'):
        if not (expected/'weights'/name).is_file():raise RuntimeError(f'missing trained checkpoint: {name}')
    save(OUT/'training_output.json',{'actual_save_dir':str(actual),'archived_save_dir':str(expected.resolve())})
    print('TRAINING COMPLETE',flush=True)


def predict(root,split,weights,out):
    """Video-only model execution. No access to annotation files here."""
    from ultralytics import YOLO
    root=Path(root);m=load_manifest(root);model=YOLO(str(weights));records={}
    start=time.time()
    for c in m['clips']:
        if c['split']!=split:continue
        rows=[]
        for i,r in enumerate(model.predict(source=str(root/c['video']),stream=True,device='cpu',
                        imgsz=PLAN['imgsz'],conf=.1,max_det=8,verbose=False)):
            points=r.keypoints.data.cpu().numpy() if r.keypoints is not None else np.empty((0,4,3))
            if points.size and points.shape[1]!=4:raise ValueError('model does not have four tyre keypoints')
            boxes=r.boxes.xyxy.cpu().numpy();scores=r.boxes.conf.cpu().numpy()
            rows.append({'frame':i,'detections':[{'score':float(s),'box':b.tolist(),'keypoints':p.tolist()}
                    for s,b,p in zip(scores,boxes,points)]})
        if len(rows)!=c['frames']:raise ValueError('incomplete inference')
        records[c['id']]=rows
        print('Inferred',c['id'],flush=True)
    save(out,records)
    save(str(out)+'.receipt.json',{'predictions_sha256':digest(out),'manifest_sha256':digest(root/'manifest.json'),
         'weights_sha256':digest(weights),'split':split,'seconds':time.time()-start,'labels_read':False})
    return records


def smooth_track(track):
    """Constant-velocity Kalman smoother over one track's per-frame margin, plus the
    sub-frame instant its smoothed path crosses the boundary. Same idea tennis line
    calling uses (reconstruct the ball's trajectory rather than trust one noisy frame)
    applied to the car's estimated clearance instead of a single detection.
    Adds smoothed_margin_m/boundary_crossing_frame; leaves margin_m (the raw per-frame
    estimate) untouched so test_experiment.py's exact-recovery check still holds."""
    frames=[f['frame'] for f in track];z=[f['margin_m'] for f in track]
    x=np.array([z[0],0.]);P=np.eye(2)
    est=[];prev=frames[0]
    for fr,m in zip(frames,z):
        dt=max(fr-prev,1);prev=fr
        F=np.array([[1.,dt],[0.,1.]]);Q=.02*np.array([[dt**3/3,dt**2/2],[dt**2/2,dt]])
        x=F@x;P=F@P@F.T+Q
        y=m-x[0];S=P[0,0]+.01;K=P[:,0]/S
        x=x+K*y;P=P-np.outer(K,P[0])
        est.append(float(x[0]))
    crossing=None
    for a,b,fa,fb in zip(est,est[1:],frames,frames[1:]):
        if (a<=0)!=(b<=0):
            crossing=fa+(fb-fa)*(a/(a-b) if a!=b else .5);break
    for f,val in zip(track,est):
        f['smoothed_margin_m']=val;f['boundary_crossing_frame']=crossing
    return track


def groups(frames):
    out=[]
    for i in sorted(set(frames)):
        if not out or i!=out[-1][-1]+1:out.append([i])
        else:out[-1].append(i)
    return [(g[0],g[-1]) for g in out]
def overlap(a,b):
    n=max(0,min(a[1],b[1])-max(a[0],b[0])+1)
    return n/((a[1]-a[0]+1)+(b[1]-b[0]+1)-n)


def observations(c,raw,threshold):
    import geometry as sim
    Hinv=np.linalg.inv(c['calibration']['homography']);tracks=[]
    local=np.array([[1.8,.82],[1.8,-.82],[-1.8,.82],[-1.8,-.82]])
    xx,yy=np.meshgrid(np.linspace(-.06,.06,7),np.linspace(-.18,.18,19))
    patch=np.c_[xx.ravel(),yy.ravel()]
    for frame in raw:
        candidates=[];i=frame['frame']
        for d in frame['detections']:
            if d['score']<threshold:continue
            uv=np.asarray(d['keypoints'])[:,:2]
            q=transform(uv,Hinv)
            if not np.isfinite(q).all():continue
            position=q.mean(0)
            direction=q[:2].mean(0)-q[2:].mean(0)
            angle=np.arctan2(direction[1],direction[0])
            R=np.array([[np.cos(angle),-np.sin(angle)],[np.sin(angle),np.cos(angle)]])
            fitted=local@R.T+position
            footprint=fitted[:,None,:]+(patch@R.T)[None,:,:]
            margin=float((sim.distance(footprint)-sim.HALF).min()-np.sqrt(2)*.01)
            candidates.append({'frame':i,'position':position.tolist(),'margin_m':margin,
                 'score':d['score'],'keypoints':d['keypoints'],'box':d['box']})
        active=[j for j,t in enumerate(tracks) if i-t[-1]['frame']<=3]
        used=set()
        if active and candidates:
            costs=[]
            for j in active:
                t=tracks[j];pos=np.array(t[-1]['position'])
                if len(t)>1:
                    delta=t[-1]['frame']-t[-2]['frame']
                    pos+=(np.array(t[-1]['position'])-t[-2]['position'])*(i-t[-1]['frame'])/delta
                costs.append([np.linalg.norm(pos-v['position']) for v in candidates])
            costs=np.array(costs);rr,cc=linear_sum_assignment(costs)
            for r,k in zip(rr,cc):
                if costs[r,k]<=6:tracks[active[r]].append(candidates[k]);used.add(k)
        for k,v in enumerate(candidates):
            if k not in used:tracks.append([v])
    return [smooth_track(t) for t in tracks]


def score(root,split,predictions,threshold,out):
    root=Path(root);m=load_manifest(root)
    receipt=json.loads(Path(str(predictions)+'.receipt.json').read_text())
    if digest(predictions)!=receipt['predictions_sha256'] or digest(root/'manifest.json')!=receipt['manifest_sha256']:
        raise ValueError('sealed input changed')
    if split!=receipt['split']:raise ValueError('wrong split')
    raw=json.loads(Path(predictions).read_text());pool=[c for c in m['clips'] if c['split']==split]
    if set(raw)!={c['id'] for c in pool}:raise ValueError('incomplete prediction coverage')
    truth=json.loads((root/'sealed/labels.json').read_text())['clips']
    rows=[];errors=[];serrors=[];review={}
    for c in pool:
        gt=truth[c['id']]
        if gt['video_sha256']!=c['video_sha256']:raise ValueError('wrong truth video')
        bycar={}
        for f in gt['frames']:bycar.setdefault(f['car_id'],{})[f['frame_idx']]=f
        events=[(car,ab) for car,fs in bycar.items() for ab in groups(i for i,f in fs.items() if f['footprint_is_violation'])]
        tracks=observations(c,raw[c['id']],threshold);pred=[];err=[];serr=[];seen=set()
        for t in tracks:
            costs={car:float(np.mean([np.linalg.norm(np.array(f['position'])-fs[f['frame']]['world_position'])
                       for f in t if f['frame'] in fs])) for car,fs in bycar.items()}
            car=min(costs,key=costs.get) if costs else None
            if car and costs[car]>4:car=None
            if car:
                for f in t:
                    if f['frame'] in bycar[car]:
                        err.append(abs(f['margin_m']-bycar[car][f['frame']]['footprint_margin_m']))
                        serr.append(abs(f['smoothed_margin_m']-bycar[car][f['frame']]['footprint_margin_m']))
                        seen.add((car,f['frame']))
            # Trajectory-smoothed margin decides events, not one noisy frame's raw estimate.
            pred.extend((car,ab) for ab in groups(f['frame'] for f in t if f['smoothed_margin_m']>0))
        pairs=sorted([(overlap(p[1],g[1]),pi,gi) for pi,p in enumerate(pred) for gi,g in enumerate(events) if p[0]==g[0]],reverse=True)
        pp=set();gg=set()
        for value,pi,gi in pairs:
            if value>=.3 and pi not in pp and gi not in gg:pp.add(pi);gg.add(gi)
        rows.append({'clip':c['id'],'family_id':c.get('family_id',c['id']),'angle_index':c.get('angle_index'),
                     'truth_events':len(events),'matched':len(pp),'false_reports':len(pred)-len(pp),
                     'misses':len(events)-len(gg),'coverage':len(seen)/max(len(gt['frames']),1),
                     'margin_mae_m':float(np.mean(err)) if err else None,
                     'margin_mae_smoothed_m':float(np.mean(serr)) if serr else None})
        errors+=err;serrors+=serr;review[c['id']]=tracks
    tp=sum(r['matched'] for r in rows);fp=sum(r['false_reports'] for r in rows);fn=sum(r['misses'] for r in rows)
    p=tp/(tp+fp) if tp+fp else 0;r=tp/(tp+fn) if tp+fn else 0
    summary={'clips':len(rows),'true_positives':tp,'false_reports':fp,'missed_events':fn,
             'precision':p if tp+fp else None,'recall':r if tp+fn else None,'f1':2*p*r/(p+r) if p+r else 0,
             'margin_p50_m':float(np.median(errors)) if errors else None,
             'margin_p95_m':float(np.percentile(errors,95)) if errors else None,
             'margin_p50_smoothed_m':float(np.median(serrors)) if serrors else None,
             'margin_p95_smoothed_m':float(np.percentile(serrors,95)) if serrors else None,'threshold':threshold,
             'labels_sha256':digest(root/'sealed/labels.json'),'prediction_sha256':digest(predictions),
             'scope':'synthetic fixed-corner, exact camera; evaluator-only trajectory association; temporal IoU >= 0.3',
             'confidence':'uncalibrated model scores; not incident probabilities'}
    # Multi-angle consensus: fuse each incident's camera angles into one decision.
    # An incident counts matched if its best angle caught the event; confidence is
    # the fraction of angles that agree, surfaced to the steward alongside the score.
    by_family={}
    for row in rows:by_family.setdefault(row['family_id'],[]).append(row)
    incidents=[{'family_id':fam,'angles':len(mem),
                'angles_agreeing':sum(1 for x in mem if x['matched']>0),
                'truth_events':mem[0]['truth_events'],'matched':max(x['matched'] for x in mem),
                'false_reports':min(x['false_reports'] for x in mem),
                'confidence':sum(1 for x in mem if x['matched']>0)/len(mem)} for fam,mem in by_family.items()]
    ctp=sum(x['matched'] for x in incidents);cfp=sum(x['false_reports'] for x in incidents)
    cfn=sum(x['truth_events']-x['matched'] for x in incidents)
    cp=ctp/(ctp+cfp) if ctp+cfp else 0;cr=ctp/(ctp+cfn) if ctp+cfn else 0
    consensus_summary={'incidents':len(incidents),'true_positives':ctp,'false_reports':cfp,'missed_events':cfn,
             'precision':cp if ctp+cfp else None,'recall':cr if ctp+cfn else None,'f1':2*cp*cr/(cp+cr) if cp+cr else 0,
             'note':'an incident is matched if ANY of its camera angles caught the event; confidence is the fraction of angles that agree'}
    save(Path(out)/'evaluation.json',{'summary':summary,'clips':rows,'consensus_summary':consensus_summary,'incidents':incidents})
    with (Path(out)/'clips.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    save(Path(out)/'review_tracks.json',review)
    print(json.dumps(summary),flush=True)
    print('CONSENSUS',json.dumps(consensus_summary),flush=True)
    return summary


def select(root):
    candidates=[]
    for name in ('best','last'):
        weights=OUT/'training/weights'/f'{name}.pt';pred=OUT/f'{name}_validation_predictions.json'
        predict(root,'validation',weights,pred)
        for threshold in PLAN['thresholds']:
            s=score(root,'validation',pred,threshold,OUT/f'validation_{name}_{threshold}')
            candidates.append({'checkpoint':str(weights),'metrics':s})
    winner=max(candidates,key=lambda c:(c['metrics']['f1'],-(c['metrics']['margin_p95_m'] if c['metrics']['margin_p95_m'] is not None else float('inf'))))
    shutil.copyfile(winner['checkpoint'],OUT/'selected.pt')
    save(OUT/'selection.json',{'weights_sha256':digest(OUT/'selected.pt'),'winner':winner,'candidates':candidates,'blind_opened':False})
    print('VALIDATION SELECTED',json.dumps(winner),flush=True)


def final_test():
    selection=json.loads((OUT/'selection.json').read_text());weights=OUT/'selected.pt'
    if digest(weights)!=selection['weights_sha256']:raise ValueError('selected model changed')
    blind=OUT/'fresh_blind'
    subprocess.run([sys.executable,'-m','simulator.generate','--out',str(blind),'--count',str(PLAN['blind_count']),
                    '--seed',str(PLAN['blind_seed']),'--fps','12','--seconds','2','--width','480','--height','270',
                    '--angles',str(PLAN['blind_angles'])],check=True)
    m=json.loads((blind/'manifest.json').read_text())
    development=json.loads((OUT/'dataset_provenance.json').read_text())
    used=set(development['train_families']+development['validation_families'])
    for c in m['clips']:
        if c['family_id'] in used:raise ValueError('blind family leakage')
        c['split']='test'
    save(blind/'manifest.json',m)
    pred=OUT/'blind_predictions.json'
    predict(blind,'test',weights,pred)
    summary=score(blind,'test',pred,selection['winner']['metrics']['threshold'],OUT/'blind_metrics')
    print('FINAL FRESH BLIND RESULT',json.dumps(summary),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['prepare','train','select','test']);p.add_argument('--data',default='data/development');a=p.parse_args()
    OUT.mkdir(exist_ok=True)
    protocol=OUT/'protocol.json'
    if protocol.exists() and json.loads(protocol.read_text())!=PLAN:raise ValueError('protocol changed mid-experiment')
    save(protocol,PLAN)
    if a.stage=='prepare':prepare(a.data)
    elif a.stage=='train':train()
    elif a.stage=='select':select(a.data)
    else:final_test()
