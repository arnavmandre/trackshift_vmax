"""Evaluate the frozen trained checkpoint on the new held-out simulator suite.
No fitting, threshold tuning or background rebuilding occurs here.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path

os.environ.setdefault('HF_HUB_OFFLINE','1')
import cv2
import numpy as np
import torch
from PIL import Image
import boundary_training as bt


def decision_summary(rows):
    known=[r for r in rows if r['truth'] is not None]
    decided=[r for r in known if r['prediction'] is not None]
    correct=sum(r['truth']==r['prediction'] for r in decided)
    def pct(a,b):return 100*a/b if b else None
    return dict(frames=len(rows),known_truth_frames=len(known),ground_truth_review_frames=len(rows)-len(known),correct=correct,
                incorrect=sum(r['truth']!=r['prediction'] for r in decided),
                review=len(known)-len(decided),
                accuracy_percent_reviews_count_as_misses=pct(correct,len(known)),
                accuracy_percent_decided_only=pct(correct,len(decided)),
                decision_coverage_percent=pct(len(decided),len(known)),
                violation_recall_percent=pct(sum(r['truth'] is True and r['prediction'] is True for r in known),sum(r['truth'] is True for r in known)),
                legal_recall_percent=pct(sum(r['truth'] is False and r['prediction'] is False for r in known),sum(r['truth'] is False for r in known)),
                always_legal_baseline_percent=pct(sum(r['truth'] is False for r in known),len(known)))


def collect_records(suite,config,cache):
    """Keep crop failures in evaluation. Only geometrically out-of-view frames excluded."""
    manifest=json.loads((suite/'suite_manifest.json').read_text())
    if not manifest.get('complete'):raise ValueError('Generation is not complete')
    if manifest['purpose']!='held_out_test_only':raise ValueError('Not a held-out suite')
    cache.mkdir(parents=True,exist_ok=True)
    records=[];excluded=[];sampled=0
    for clip in manifest['clips']:
        video=suite/clip['video'];doc=json.loads(video.with_suffix('.json').read_text())
        if hashlib.sha256(video.read_bytes()).hexdigest()!=clip['sha256']:raise ValueError(f'Video changed: {video}')
        if doc['label_contract']['schema_version']!=bt.SCHEMA:raise ValueError('Wrong label schema')
        camera=doc['camera'];rows=doc['frames']
        assert len({r['car_id'] for r in rows})==1
        cap=cv2.VideoCapture(str(video))
        if not cap.isOpened():raise RuntimeError(f'Cannot open {video}')
        assert int(cap.get(cv2.CAP_PROP_FRAME_COUNT))==len(rows)
        try:
            for i,row in enumerate(rows):
                ok,bgr=cap.read()
                if not ok:raise RuntimeError(f'Cannot decode {video}:{i}')
                assert row['frame_idx']==i
                if i%config.frame_stride:continue
                sampled+=1
                xyz=np.array([p for w in bt.WHEELS for p in row['tyre_boundaries'][w]['ground_tread_endpoints_world_xyz']])
                local=(xyz-np.asarray(camera['position_m']))@np.asarray(camera['R_world_to_camera']).T
                projected=local@np.asarray(camera['K']).T
                uv=projected[:,:2]/projected[:,2,None]
                interval=bt.boundary_interval(xyz[:,:2],config.boundary_samples)
                np.testing.assert_allclose(interval,row['boundary_margin_m_interval'],atol=1e-8)
                assert bt.verdict(interval) is row['is_violation']
                meta=dict(video=str(video.resolve()),scenario=doc['scenario'],camera_name=camera['name'],
                          frame_idx=i,truth=row['is_violation'])
                in_view=(np.isfinite(uv).all() and np.all(local[:,2]>.2) and np.all(uv>=0)
                         and np.all(uv<np.asarray(camera['resolution'])))
                if not in_view:
                    excluded.append(dict(**meta,prediction=None,review_reason='ground_truth_endpoint_out_of_view'))
                    continue
                path=cache/f'{doc["scenario"]}_{camera["name"]}_{i:04d}.png'
                Image.fromarray(cv2.cvtColor(bgr,cv2.COLOR_BGR2RGB)).save(path)
                records.append(dict(image_path=str(path.resolve()),video=meta['video'],split='expanded_test',
                    scenario=doc['scenario'],frame_idx=i,camera=camera,points_original_px=uv.tolist(),
                    world=xyz[:,:2].tolist(),is_violation=row['is_violation'],true_margin_m_interval=interval.tolist()))
        finally:cap.release()
    assert records and {r['is_violation'] for r in records} >= {False,True}
    return records,excluded,dict(sampled=sampled,in_view=len(records),out_of_view=len(excluded),clips=len(manifest['clips']))


def run(suite,checkpoint,output,stride=2,device=None):
    suite=Path(suite);checkpoint=Path(checkpoint);output=Path(output)
    if output.exists():raise ValueError(f'Refusing to overwrite evaluation: {output}')
    output.mkdir(parents=True)
    torch.set_num_threads(min(4,os.cpu_count() or 1))
    device=device or ('cuda' if torch.cuda.is_available() else 'cpu')
    nvrtc_handle=bt.preload_nvrtc() if str(device).startswith('cuda') else None
    print(f'Loading frozen checkpoint on {device}',flush=True)
    model,processor,config,backgrounds=bt.load_checkpoint(checkpoint,device)
    config.frame_stride=stride
    records,excluded,audit=collect_records(suite,config,output/'frames')
    print(f'Dataset audit: {audit}',flush=True)
    bt.save_json(output/'data_audit.json',audit)
    bt.save_json(output/'excluded_frames.json',excluded)
    bt.save_json(output/'evaluation_config.json',dict(checkpoint=str(checkpoint.resolve()),
        checkpoint_sha256=hashlib.sha256((checkpoint/'boundary_heatmaps.pt').read_bytes()).hexdigest(),
        suite=str(suite.resolve()),stride=stride,device=device,training_performed=False,
        background_source='frozen checkpoint training backgrounds',scope='new trajectories, same synthetic scene and cameras'))
    metrics,results=bt.evaluate_records(model,processor,records,backgrounds,config,device,output,'expanded_test')
    for row,record in zip(results,records):
        row.update(scenario=record['scenario'],camera_name=record['camera']['name'])
    bt.save_json(output/'expanded_test_predictions.json',results)
    report=dict(in_view=decision_summary(results),
                all_sampled_frames_out_of_view_count_as_review=decision_summary(results+excluded),
                by_scenario={s:decision_summary([r for r in results if r['scenario']==s]) for s in sorted({r['scenario'] for r in results})},
                by_camera={c:decision_summary([r for r in results if r['camera_name']==c]) for c in sorted({r['camera_name'] for r in results})},
                scope='Frame-level accuracy on unseen synthetic trajectories, not real-world accuracy. Correlated frames are not independent trials.',
                audit=audit)
    bt.save_json(output/'accuracy_report.json',report)
    def display(value):return 'N/A' if value is None else f'{value:.2f}%'
    score=report['in_view']
    lines=['# Expanded synthetic test results', '',
           f"Accuracy (reviews count as misses): **{display(score['accuracy_percent_reviews_count_as_misses'])}**.",
           f"Correct: {score['correct']}/{score['known_truth_frames']}; wrong: {score['incorrect']}; review: {score['review']}.",
           f"Accuracy on decided frames: {display(score['accuracy_percent_decided_only'])}; decision coverage: {display(score['decision_coverage_percent'])}.",
           f"Out-of-view frames: {audit['out_of_view']}/{audit['sampled']}. Including those as reviews: "
           f"{display(report['all_sampled_frames_out_of_view_count_as_review']['accuracy_percent_reviews_count_as_misses'])}.",
           '', '| Scenario | In-view accuracy, reviews as misses | Frames | Reviews |',
           '|---|---:|---:|---:|']
    for scenario,values in report['by_scenario'].items():
        lines.append(f"| {scenario} | {display(values['accuracy_percent_reviews_count_as_misses'])} | {values['frames']} | {values['review']} |")
    lines.extend(['', report['scope'], '', 'Checkpoint and thresholds were frozen; no training or background rebuilding used these clips.'])
    (output/'RESULTS.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps(report,indent=2),flush=True)
    return report


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--suite',type=Path,default=Path('data/expanded_v1'))
    parser.add_argument('--checkpoint',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--stride',type=int,default=2)
    args=parser.parse_args()
    if args.stride<1:parser.error('--stride must be positive')
    run(args.suite,args.checkpoint,args.output,args.stride)
