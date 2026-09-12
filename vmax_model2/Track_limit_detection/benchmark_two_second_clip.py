"""Time all three camera views for a 2-second synthetic clip, including decoding.

Example:
  python benchmark_two_second_clip.py --scenario moderate_excursion \
      --output two_second_benchmark.json
"""
import argparse
import json
import time
from pathlib import Path

import cv2
import torch

import boundary_training as bt
from multiview import predict_multiview

CAMERAS = ('broadcast', 'exit', 'trackside')


def benchmark(data_dir, checkpoint, scenario, seconds, device):
    data_dir=Path(data_dir);root=data_dir/scenario
    checkpoint=Path(checkpoint)
    torch.set_num_threads(4)
    load_start=time.perf_counter()
    nvrtc_handle=bt.preload_nvrtc() if str(device).startswith('cuda') else None
    model,processor,config,backgrounds=bt.load_checkpoint(checkpoint,device)
    def sync():
        if str(device).startswith('cuda'):torch.cuda.synchronize(device)
    sync();load_seconds=time.perf_counter()-load_start
    docs={name:json.loads((root/f'{name}.json').read_text()) for name in CAMERAS}
    fps=docs[CAMERAS[0]]['fps']
    frames_to_check=int(round(seconds*fps))
    if frames_to_check<1:raise ValueError('Clip duration must include at least one frame')
    if any(d['fps']!=fps or len(d['frames'])<frames_to_check for d in docs.values()):
        raise ValueError('Cameras have inconsistent fps or insufficient frames')

    def frame(cap,name,idx):
        ok,bgr=cap.read()
        if not ok:raise RuntimeError(f'Cannot decode {name} frame {idx}')
        row=docs[name]['frames'][idx]
        return dict(image=cv2.cvtColor(bgr,cv2.COLOR_BGR2RGB),camera=docs[name]['camera'],
            scenario=scenario,car_id=row['car_id'],frame_idx=idx,time_s=row['t'])

    # Warm-up without including first-call kernel setup in the clip time.
    warm=[]
    for name in CAMERAS:
        cap=cv2.VideoCapture(str(root/f'{name}.mp4'))
        if not cap.isOpened():raise RuntimeError(f'Cannot open {name} video')
        try:warm.append(frame(cap,name,0))
        finally:cap.release()
    predict_multiview(model,processor,warm,backgrounds,config,device,CAMERAS)
    sync()

    start=time.perf_counter()
    caps={name:cv2.VideoCapture(str(root/f'{name}.mp4')) for name in CAMERAS}
    counts={'violation':0,'legal':0,'review':0}
    try:
        if not all(cap.isOpened() for cap in caps.values()):raise RuntimeError('Cannot open all camera videos')
        for idx in range(frames_to_check):
            views=[frame(caps[name],name,idx) for name in CAMERAS]
            result=predict_multiview(model,processor,views,backgrounds,config,device,CAMERAS)
            label='review' if result['prediction'] is None else ('violation' if result['prediction'] else 'legal')
            counts[label]+=1
    finally:
        for cap in caps.values():cap.release()
    sync();elapsed=time.perf_counter()-start
    return dict(scenario=scenario,device=str(device),fps=fps,seconds_of_video=seconds,
        frames_per_camera=frames_to_check,cameras=len(CAMERAS),
        model_load_seconds=load_seconds,processing_seconds_including_decode=elapsed,
        processing_seconds_per_combined_frame=elapsed/frames_to_check,
        real_time_factor=elapsed/seconds,verdicts=counts,
        timing_scope='Sequential three-view verdicts, video decode, crop, preprocessing, model, geometry and fusion; excludes model loading, display and network I/O.')


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--data-dir',type=Path,default=Path('data/expanded_v1'))
    parser.add_argument('--checkpoint',type=Path,default=Path('training_runs/boundary_heatmaps_20260912_191710_183913/best_model'))
    parser.add_argument('--scenario',default='moderate_excursion')
    parser.add_argument('--seconds',type=float,default=2.)
    parser.add_argument('--device',default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--output',type=Path)
    args=parser.parse_args()
    report=benchmark(args.data_dir,args.checkpoint,args.scenario,args.seconds,args.device)
    payload=json.dumps(report,indent=2)+'\n'
    if args.output:
        if args.output.exists():parser.error(f'Refusing to overwrite {args.output}')
        args.output.write_text(payload)
    print(payload,end='')
