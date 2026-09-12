"""Label-free manifest inference, sealed prediction receipts and separate scoring."""
from __future__ import annotations
import argparse
import csv
import json
from pathlib import Path
import time
import numpy as np
from .evidence import sha256, enrich


def resolve(root, relative):
    root=Path(root).resolve();p=(root/relative).resolve()
    if not p.is_relative_to(root):raise ValueError('manifest path escapes dataset root')
    return p


def validate_manifest(path):
    path=Path(path);data=json.loads(path.read_text()); ids=set();family_splits={};hash_splits={}
    if not data.get('clips'):raise ValueError('manifest has no clips')
    for c in data['clips']:
        if c['id'] in ids:raise ValueError('duplicate clip id')
        ids.add(c['id'])
        if c['split'] not in ('train','validation','test','external'):raise ValueError('invalid split')
        for value,seen in ((c['family_id'],family_splits),(c['video_sha256'],hash_splits)):
            if value in seen and seen[value]!=c['split']:raise ValueError('family or identical video crosses splits')
            seen[value]=c['split']
        if c['fps']<=0 or c['frames']<1:raise ValueError('invalid video timing')
        if sha256(resolve(path.parent,c['video']))!=c['video_sha256']:raise ValueError('video content changed')
        H=np.asarray(c['calibration']['homography'],float)
        if H.shape!=(3,3) or not np.isfinite(H).all() or np.linalg.matrix_rank(H)!=3:raise ValueError('invalid homography')
        if not np.isfinite(c['calibration']['sigma_m']) or c['calibration']['sigma_m']<0:raise ValueError('invalid calibration uncertainty')
    return data


def seal_run(out, manifest_path, records, metadata):
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    if (out/'receipt.json').exists():raise ValueError('sealed run already exists; use a fresh output directory')
    outputs={}
    for cid,result in records.items():
        name=f'{cid}.json'
        if Path(name).name!=name:raise ValueError('unsafe clip id')
        path=out/name;path.write_text(json.dumps(result,allow_nan=False));outputs[name]=sha256(path)
    receipt={'schema_version':1,'manifest_sha256':sha256(manifest_path),'outputs':outputs,**metadata}
    (out/'receipt.json').write_text(json.dumps(receipt,indent=2,allow_nan=False))
    return receipt


def infer(manifest_path,out,weights,split='test',footprint=True,device='cpu',contact_sigma=.10,systematic=.05):
    from .clips import Clip
    from .detector import VmaxDetector
    from .pipeline import run_clip
    from .footprints import apply
    data=validate_manifest(manifest_path);pool=[c for c in data['clips'] if c['split']==split]
    if not pool:raise ValueError(f'no clips in split {split}')
    if (Path(out)/'receipt.json').exists():raise ValueError('sealed run already exists')
    detector=VmaxDetector(weights=weights,device=device);results={};start=time.perf_counter()
    for c in pool:
        clip=Clip(c['id'],c['id'],resolve(Path(manifest_path).parent,c['video']),None,fps=c['fps'])
        from .circuit import CircuitMap
        track_map=CircuitMap(c["track_polygon_m"]) if "track_polygon_m" in c else None
        if data.get("dataset_kind")=="external_experimental" and track_map is None:
            raise ValueError("external footage requires a surveyed track polygon")
        result=run_clip(clip,detector,cache=False,min_frames=1,bridge=0,calibration=c['calibration'],identify=False,track_map=track_map)
        # The legacy two-colour roster is not valid identity evidence on new footage.
        for track in result['tracks'].values():
            track['attribution']={'car_id':None,'confidence':0.0,'rung':'unattributed','evidence':{'reason':'external identity roster not validated'}}
            for ev in track['events']:ev.update(driver_confidence=0.0,incident_score=0.0)
        if footprint:apply(result,track_map=track_map)
        enrich(result,contact_sigma,systematic)
        result.update(clip_id=c['id'],video_sha256=c['video_sha256'],split=split)
        results[c['id']]=result
        print(f'{c["id"]}: {sum(len(t["events"]) for t in result["tracks"].values())} review candidates',flush=True)
    elapsed=time.perf_counter()-start
    seal_run(out,manifest_path,results,{'weights_sha256':sha256(weights),'split':split,'seconds':elapsed,
        'geometry':'footprint' if footprint else 'point','device':device,'labels_read':False,
        'source_sha256':{p.name:sha256(p) for p in sorted(Path(__file__).parent.glob('*.py'))}})


def intervals(flags):
    out=[];start=prev=None
    for i in sorted(set(flags)):
        if prev is None or i!=prev+1:
            if start is not None:out.append((start,prev))
            start=i
        prev=i
    if start is not None:out.append((start,prev))
    return out


def iou(a,b):
    inter=max(0,min(a[1],b[1])-max(a[0],b[0])+1)
    return inter/(a[1]-a[0]+1+b[1]-b[0]+1-inter)


def score(manifest_path,run_dir,labels_path,out):
    data=validate_manifest(manifest_path);root=Path(run_dir)
    receipt=json.loads((root/'receipt.json').read_text())
    if receipt['manifest_sha256']!=sha256(manifest_path):raise ValueError('manifest differs from sealed run')
    # Validate every prediction before loading labels.
    results={}
    for name,digest in receipt['outputs'].items():
        p=resolve(root,name)
        if sha256(p)!=digest:raise ValueError('prediction changed after sealing')
        r=json.loads(p.read_text());results[r['clip_id']]=r
    expected={c['id'] for c in data['clips'] if c['split']==receipt['split']}
    if set(results)!=expected:raise ValueError('run does not cover the complete selected split')
    labels=json.loads(Path(labels_path).read_text())['clips'];rows=[];all_errors=[]
    for cid,r in results.items():
        truth=labels[cid]
        if truth['video_sha256']!=r['video_sha256']:raise ValueError('labels refer to different video')
        bycar={}
        for f in truth['frames']:bycar.setdefault(f['car_id'],{})[f['frame_idx']]=f
        field='footprint' if receipt['geometry']=='footprint' else 'point'
        events=[(car,ab) for car,fs in bycar.items() for ab in intervals(i for i,f in fs.items() if f[field+'_is_violation'])]
        predicted=[];errors=[];observed=set();identity_correct=identity_claims=0
        for tid,t in r['tracks'].items():
            costs={car:np.mean([np.linalg.norm(np.asarray(f['position_m'])-fs[f['frame_idx']]['world_position']) for f in t['frames'] if f['frame_idx'] in fs]) for car,fs in bycar.items()}
            costs={k:v for k,v in costs.items() if np.isfinite(v)}
            car=min(costs,key=costs.get) if costs else None
            if car and costs[car]>4:car=None
            claim=(t.get('attribution') or {}).get('car_id')
            if claim:identity_claims+=1;identity_correct+=int(claim==car)
            if car:
                for f in t['frames']:
                    if f['frame_idx'] in bycar[car]:
                        observed.add((car,f['frame_idx']))
                        errors.append(abs(f['margin_m']-bycar[car][f['frame_idx']][field+'_margin_m']))
            predicted.extend((car,(e['start_frame'],e['end_frame_inclusive'])) for e in t['events'])
        # Globally sorted greedy one-to-one temporal matches; extra reports remain false positives.
        pairs=sorted([(iou(p[1],t[1]),pi,ti) for pi,p in enumerate(predicted) for ti,t in enumerate(events) if p[0]==t[0]],reverse=True)
        used_p=set();used_t=set()
        for overlap,pi,ti in pairs:
            if overlap>=.3 and pi not in used_p and ti not in used_t:used_p.add(pi);used_t.add(ti)
        row={'clip':cid,'split':receipt['split'],'truth_events':len(events),'reported_events':len(predicted),
             'matched_events':len(used_p),'false_reports':len(predicted)-len(used_p),'missed_events':len(events)-len(used_t),
             'annotated_car_frame_coverage':len(observed)/max(sum(map(len,bycar.values())),1),
             'margin_mae_m':float(np.mean(errors)) if errors else None,
             'identity_claims':identity_claims,'identity_correct':identity_correct}
        rows.append(row);all_errors+=errors
    tp=sum(r['matched_events'] for r in rows);fp=sum(r['false_reports'] for r in rows);fn=sum(r['missed_events'] for r in rows)
    summary={'clips':len(rows),'event_precision':tp/(tp+fp) if tp+fp else None,'event_recall':tp/(tp+fn) if tp+fn else None,
             'true_positives':tp,'false_reports':fp,'missed_events':fn,
             'margin_p50_m':float(np.percentile(all_errors,50)) if all_errors else None,
             'margin_p95_m':float(np.percentile(all_errors,95)) if all_errors else None,
             'confidence_calibration':'not measured','matching':'same truth-associated car, temporal IoU >= 0.3, one-to-one',
             'receipt_sha256':sha256(root/'receipt.json'),'labels_sha256':sha256(labels_path)}
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    (out/'evaluation.json').write_text(json.dumps({'summary':summary,'clips':rows},indent=2))
    with (out/'clips.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    print(json.dumps(summary,indent=2));return summary


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__);sub=p.add_subparsers(dest='command',required=True)
    q=sub.add_parser('infer');q.add_argument('--manifest',required=True);q.add_argument('--out',required=True)
    q.add_argument('--weights',default='pipeline_out/vmaxnet.pt');q.add_argument('--split',choices=['train','validation','test','external'],default='test')
    q.add_argument('--point-benchmark',action='store_true');q.add_argument('--device',default='cpu')
    q.add_argument('--contact-sigma',type=float,default=.10);q.add_argument('--systematic-bound',type=float,default=.05)
    q=sub.add_parser('score');q.add_argument('--manifest',required=True);q.add_argument('--runs',required=True);q.add_argument('--labels',required=True);q.add_argument('--out',required=True)
    q=sub.add_parser('validate');q.add_argument('--manifest',required=True)
    a=p.parse_args(argv)
    if a.command=='infer':infer(a.manifest,a.out,a.weights,a.split,not a.point_benchmark,a.device,a.contact_sigma,a.systematic_bound)
    elif a.command=='score':score(a.manifest,a.runs,a.labels,a.out)
    else:print(f'{len(validate_manifest(a.manifest)["clips"])} clips verified')

if __name__=='__main__':main()
