"""Reuse frozen per-view predictions and infer previously excluded views before fusion.
Out-of-view ground truth does NOT suppress camera predictions in multiview scoring.
"""
import argparse
import hashlib
import json
import time
from pathlib import Path
import cv2
import numpy as np
import torch
import boundary_training as bt
from evaluate_extra import decision_summary
from multiview import fuse_records, predict_multiview


def benchmark_inference(suite, model, processor, config, backgrounds, device, frames_per_scenario=2):
    """Time fresh three-view decisions. Decode source videos before the timer.

    Model loading, video decoding, plotting and saving are excluded. The timed path
    includes cropping, preprocessing, inference, geometry and three-view fusion.
    """
    suite=Path(suite)
    manifest=json.loads((suite/'suite_manifest.json').read_text())
    if not manifest.get('complete'):raise ValueError('Incomplete simulator suite')
    groups=[]
    by_scenario={}
    for entry in manifest['clips']:
        video=suite/entry['video'];doc=json.loads(video.with_suffix('.json').read_text())
        by_scenario.setdefault(doc['scenario'],{})[doc['camera']['name']]=(video,doc)
    for scenario,cameras in sorted(by_scenario.items()):
        if set(cameras)!=set(manifest['cameras']):raise ValueError(f'Missing camera for {scenario}')
        count=len(next(iter(cameras.values()))[1]['frames'])
        indices=np.unique(np.linspace(count/(frames_per_scenario+1),
            count*frames_per_scenario/(frames_per_scenario+1),frames_per_scenario,dtype=int))
        for frame_idx in indices:
            frames=[]
            for camera_name in manifest['cameras']:
                video,doc=cameras[camera_name]
                cap=cv2.VideoCapture(str(video))
                if not cap.isOpened():raise RuntimeError(f'Cannot open {video}')
                try:
                    cap.set(cv2.CAP_PROP_POS_FRAMES,int(frame_idx));ok,bgr=cap.read()
                finally:cap.release()
                if not ok:raise RuntimeError(f'Cannot decode {video}:{frame_idx}')
                row=doc['frames'][int(frame_idx)]
                frames.append(dict(image=cv2.cvtColor(bgr,cv2.COLOR_BGR2RGB),camera=doc['camera'],
                    scenario=scenario,car_id=row['car_id'],frame_idx=int(frame_idx),time_s=row['t']))
            groups.append(frames)
    if not groups:raise ValueError('No benchmark groups')
    def sync():
        if str(device).startswith('cuda'):torch.cuda.synchronize(device)
    model.eval()
    # Warm up kernels without including first-call setup in the measurement.
    for _ in range(2):predict_multiview(model,processor,groups[0],backgrounds,config,device,manifest['cameras'])
    times=[]
    for frames in groups:
        sync();start=time.perf_counter()
        predict_multiview(model,processor,frames,backgrounds,config,device,manifest['cameras'])
        sync();times.append(time.perf_counter()-start)
    summary=dict(synchronized_checks=len(times),camera_views_per_check=len(manifest['cameras']),
        average_seconds_per_three_view_check=float(np.mean(times)),
        median_seconds_per_three_view_check=float(np.median(times)),
        p95_seconds_per_three_view_check=float(np.percentile(times,95)),
        average_seconds_per_camera_view=float(np.mean(times)/len(manifest['cameras'])),
        three_view_checks_per_second=float(1/np.mean(times)),
        excludes=['model_loading','video_decoding','disk_writes','plotting'],
        includes=['car_cropping','image_preprocessing','model_inference','geometry','view_fusion'],
        device=str(device))
    return summary


def run(source,output,device=None):
    source=Path(source);output=Path(output)
    if output.exists():raise ValueError(f'Refusing to overwrite {output}')
    settings=json.loads((source/'evaluation_config.json').read_text())
    checkpoint=Path(settings['checkpoint']);suite=Path(settings['suite'])
    if hashlib.sha256((checkpoint/'boundary_heatmaps.pt').read_bytes()).hexdigest()!=settings['checkpoint_sha256']:
        raise ValueError('Checkpoint has changed')
    manifest=json.loads((suite/'suite_manifest.json').read_text())
    if not manifest['complete']:raise ValueError('Incomplete suite')
    previous=json.loads((source/'expanded_test_predictions.json').read_text())
    cached={(r['video'],r['frame_idx']):r for r in previous}
    if len(cached)!=len(previous):raise ValueError('Duplicate cached predictions')
    output.mkdir(parents=True)
    device=device or ('cuda' if torch.cuda.is_available() else 'cpu')
    torch.set_num_threads(4)
    nvrtc_handle=bt.preload_nvrtc() if str(device).startswith('cuda') else None
    model,processor,config,backgrounds=bt.load_checkpoint(checkpoint,device)
    views=[];new_count=0;reused=0
    for clip in manifest['clips']:
        video=(suite/clip['video']).resolve()
        if hashlib.sha256(video.read_bytes()).hexdigest()!=clip['sha256']:raise ValueError(f'Video changed: {video}')
        doc=json.loads(video.with_suffix('.json').read_text());camera=doc['camera']
        cap=cv2.VideoCapture(str(video))
        if not cap.isOpened():raise RuntimeError(f'Cannot open {video}')
        try:
            for truth in doc['frames']:
                idx=truth['frame_idx']
                if idx%settings['stride']:continue
                key=(str(video),idx)
                if key in cached:
                    result=dict(cached[key]);reused+=1
                    if result['truth'] is not truth['is_violation']:raise ValueError('Cached truth mismatch')
                else:
                    cap.set(cv2.CAP_PROP_POS_FRAMES,idx);ok,bgr=cap.read()
                    if not ok:raise RuntimeError(f'Cannot decode {key}')
                    pred=bt.predict_frame(model,processor,cv2.cvtColor(bgr,cv2.COLOR_BGR2RGB),camera,backgrounds,config,device)
                    interval=pred.get('interval');points=pred.get('points_original_px')
                    result=dict(video=str(video),frame_idx=idx,truth=truth['is_violation'],
                        prediction=pred.get('prediction'),review_reason=pred.get('reason'),
                        points_original_px=points.tolist() if points is not None else None,
                        predicted_margin_m_interval=interval.tolist() if interval is not None else None)
                    new_count+=1
                result.update(scenario=doc['scenario'],car_id=truth['car_id'],time_s=truth['t'],camera_name=camera['name'])
                views.append(result)
        finally:cap.release()
        print(f"{doc['scenario']}/{camera['name']}: {len(views)} views ready",flush=True)
    if reused!=len(cached):raise ValueError('Cached predictions do not match the suite')
    fused=fuse_records(views,manifest['cameras'])
    timing=benchmark_inference(suite,model,processor,config,backgrounds,device)
    report=dict(combined=decision_summary(fused),individual_views_all_frames=decision_summary(views),
        by_scenario={s:decision_summary([r for r in fused if r['scenario']==s]) for s in sorted({r['scenario'] for r in fused})},
        review_reasons=dict(__import__('collections').Counter(r['review_reason'] for r in fused if r['review_reason'])),
        accepted_view_counts=dict(__import__('collections').Counter(r['accepted_view_count'] for r in fused)),
        view_records=len(views),synchronized_examples=len(fused),reused_predictions=reused,new_predictions=new_count,
        timing=timing,
        rule='Use unanimous non-review verdicts, even if only one view can decide. Conflict or no accepted view means review.',
        limitations='Same synthetic scene and cameras. No truth visibility gating; frame/car synchronization comes from simulator metadata. Decision fusion, not joint multi-view training or triangulation.')
    bt.save_json(output/'view_predictions.json',views);bt.save_json(output/'combined_predictions.json',fused)
    bt.save_json(output/'timing.json',timing)
    bt.save_json(output/'accuracy_report.json',report)
    bt.save_json(output/'evaluation_config.json',dict(source_run=str(source.resolve()),**settings,
        visibility_filter=False,fusion_rule=report['rule'],fusion_device=str(device)))
    score=report['combined']
    def pct(v):return 'N/A' if v is None else f'{v:.2f}%'
    text=['# Three-camera synthetic evaluation','',
        f"**Accuracy (reviews count as misses): {pct(score['accuracy_percent_reviews_count_as_misses'])}.**",'',
        f"{score['correct']} correct, {score['incorrect']} wrong, {score['review']} reviews across {len(fused)} synchronized examples ({len(views)} camera frames).",
        f"Decision coverage: {pct(score['decision_coverage_percent'])}; accuracy when decided: {pct(score['accuracy_percent_decided_only'])}.",'',
        f"Average inference time: **{timing['average_seconds_per_three_view_check']:.3f} s per three-camera check** "
        f"({timing['average_seconds_per_camera_view']:.3f} s per camera view; {timing['synchronized_checks']} timed checks).",
        'Timing includes crop, preprocessing, model, geometry and fusion; it excludes video decoding, model loading, plotting and file writes.','',
        report['rule'],'','| Scenario | Accuracy, reviews as misses | Examples | Reviews |','|---|---:|---:|---:|']
    for name,s in report['by_scenario'].items():text.append(f"| {name} | {pct(s['accuracy_percent_reviews_count_as_misses'])} | {s['frames']} | {s['review']} |")
    text.extend(['',report['limitations']])
    (output/'RESULTS.md').write_text('\n'.join(text)+'\n')
    print(json.dumps(report,indent=2));return report


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--source',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();run(args.source,args.output)
