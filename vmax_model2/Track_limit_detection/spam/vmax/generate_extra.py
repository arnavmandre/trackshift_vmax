"""Generate new held-out trajectories using the existing VMAX OpenGL renderer.
Run from project root: python spam/vmax/generate_extra.py --output data/expanded_v1
Never overwrites an existing dataset. Case names describe centre-path targets;
actual verdicts come from the ground-tread boundary contract.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time

os.environ.setdefault('LP_NUM_THREADS', '4')
import numpy as np
import geometry as g
import tyre_boundaries as tb

# Fixed before evaluation; no selection based on model performance.
CASES = [
    dict(name='legal_close', target=-.10, speed=11., origin=9., start=32, end=69, ramp=18),
    dict(name='legal_edge', target=.055, speed=12.5, origin=6., start=31, end=65, ramp=15),
    dict(name='shallow_excursion', target=.10, speed=11.5, origin=9., start=34, end=68, ramp=17),
    dict(name='moderate_excursion', target=.22, speed=12.8, origin=7., start=30, end=61, ramp=18),
    dict(name='large_excursion', target=.45, speed=10.5, origin=10., start=38, end=73, ramp=17),
    dict(name='severe_excursion', target=.90, speed=12.2, origin=8.5, start=33, end=64, ramp=19),
    dict(name='short_excursion', target=.35, speed=11.8, origin=8., start=46, end=50, ramp=12),
    dict(name='sustained_excursion', target=.20, speed=12.5, origin=7.5, start=25, end=76, ramp=15),
]


def make_document(case):
    previous = g.S.copy()
    try:
        g.S = case['origin'] + case['speed'] * g.T
        desired = g.profile(case['target'], case['start'], case['end'], case['ramp'])
        p, h, q, e = g.solve(desired)
    finally:
        g.S = previous
    travel = np.r_[0, np.cumsum(np.linalg.norm(np.diff(p, axis=0), axis=1))]
    curvature = np.gradient(np.unwrap(h)) / np.maximum(np.gradient(travel), 1e-9)
    steering = np.clip(np.arctan(3.6 * curvature), -.45, .45)
    speeds = np.linalg.norm(np.gradient(p, 1/g.FPS, axis=0, edge_order=2), axis=1)
    rows = [dict(frame_idx=i, t=float(g.T[i]), car_id='car_1', world_position=p[i].tolist(),
                 heading_rad=float(h[i]), visual_steering_rad=float(steering[i]),
                 contact_points_world=dict(zip(g.KEYS, q[i].tolist())),
                 corner_excess_m=dict(zip(g.KEYS, e[i].tolist())),
                 min_excess_m=float(e[i].min()), max_excess_m=float(e[i].max()),
                 is_violation=bool(e[i].min()>0), speed_mps=float(speeds[i])) for i in range(g.N)]
    return dict(scenario=case['name'], fps=g.FPS, frames=rows, generation_parameters=case), travel


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--output',type=Path,default=Path('data/expanded_v1'))
    args=parser.parse_args()
    if args.output.exists():
        raise SystemExit(f'Refusing to overwrite {args.output}; choose a new output path.')
    import render as r
    renderer=r.Renderer()  # Check EGL/dependencies before creating the dataset.
    args.output.mkdir(parents=True)
    manifest=dict(version=1, purpose='held_out_test_only', complete=False, cases=CASES,
                  frame_count_per_clip=g.N, fps=g.FPS, resolution=[r.WIDTH,r.HEIGHT],
                  cameras=[c['name'] for c in r.CAMS], clips=[],
                  limitations='Same synthetic scene, car and fixed cameras as training; new trajectories only.',
                  generator_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    def save():
        (args.output/'suite_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    save()
    started=time.monotonic()
    for case in CASES:
        doc,travel=make_document(case)
        folder=args.output/case['name'];folder.mkdir()
        for camera in r.CAMS:
            video=folder/(camera['name']+'.mp4')
            command=['ffmpeg','-nostdin','-n','-loglevel','error','-f','rawvideo','-pix_fmt','rgb24',
                     '-s',f'{r.WIDTH}x{r.HEIGHT}','-r',str(g.FPS),'-i','-','-an','-c:v','libx264',
                     '-threads','2','-preset','fast','-crf','19','-pix_fmt','yuv420p','-movflags','+faststart',str(video)]
            process=subprocess.Popen(command,stdin=subprocess.PIPE)
            try:
                for i,row in enumerate(doc['frames']):
                    im=renderer.frame(camera,[row],[travel[i]])
                    process.stdin.write(im.tobytes())
                    if i==48:
                        im.save(folder/(camera['name']+'.png'))
                        r.debug(im,camera,[row]).save(folder/(camera['name']+'_debug.png'))
            finally:
                process.stdin.close()
                status=process.wait()
            if status: raise RuntimeError(f'FFmpeg failed: {video}')
            enriched=tb.annotate_document(doc,camera)
            enriched['render_version']='VMAX v2 / boundary labels v1 / expanded held-out trajectories'
            video.with_suffix('.json').write_text(json.dumps(enriched,indent=2)+'\n')
            manifest['clips'].append(dict(video=str(video.relative_to(args.output)),
                sha256=hashlib.sha256(video.read_bytes()).hexdigest(),events=enriched['events']))
            save()
            print(f'{len(manifest["clips"])}/24 {video} ({time.monotonic()-started:.0f}s)',flush=True)
    manifest['complete']=True;save()


if __name__=='__main__':main()
