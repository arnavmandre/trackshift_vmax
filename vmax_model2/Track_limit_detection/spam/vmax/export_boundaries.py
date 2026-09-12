"""Export new boundary labels for already-rendered clips, without OpenGL or re-rendering.

python export_boundaries.py --input ../../data --output ../../data_boundaries
"""
import argparse
import json
import shutil
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw
import tyre_boundaries as tb


def draw_debug(image, rows, camera):
    im = image.copy()
    draw = ImageDraw.Draw(im)
    track = tb.track_boundary(camera)
    for boundary in track.values():
        pts = boundary['image']['pixels']
        for a, b in zip(pts, pts[1:]):
            if a is not None and b is not None and max(map(abs, a+b)) < 10000:
                draw.line([tuple(a), tuple(b)], fill=(255, 220, 0), width=2)
    for row in rows:
        for key, wheel in row['tyre_boundaries'].items():
            for field, color, width in [('lower_outline_image',(0,210,255),2),
                                         ('ground_tread_endpoints_image',(255,70,190),4)]:
                points = wheel[field]['pixels']
                for a, b in zip(points, points[1:]):
                    if a is not None and b is not None and max(map(abs,a+b)) < 10000:
                        draw.line([tuple(a),tuple(b)], fill=color, width=width)
            for point, inside in zip(wheel['ground_tread_endpoints_image']['pixels'],
                                     wheel['ground_tread_endpoints_image']['in_frame']):
                if inside:
                    x,y=point;draw.ellipse((x-3,y-3,x+3,y+3),fill=(255,70,190))
    draw.rectangle((0,0,im.width,46),fill=(10,20,30))
    draw.text((10,5),'Yellow: track boundary | Cyan: lower tyre outline | Pink: ground-tread boundary',fill='white')
    draw.text((10,24),'Synthetic smooth rigid envelope. Hidden edges remain labelled; not visible-pixel segmentation.',fill='white')
    return im


def export(input_dir, output_dir, make_previews=True):
    input_dir, output_dir = Path(input_dir).resolve(), Path(output_dir).resolve()
    if input_dir == output_dir or input_dir in output_dir.parents:
        raise ValueError('Use a separate sibling output directory, not the input directory or a child of it')
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f'Refusing to overwrite existing output: {output_dir}')
    output_dir.mkdir(parents=True,exist_ok=True)
    report = {'label_contract':tb.RULE,'clips':[]}
    for video in sorted(input_dir.glob('*/*.mp4')):
        label = video.with_suffix('.json')
        if not label.exists(): label = video.parent/'ground_truth.json'
        if not label.exists(): raise FileNotFoundError(f'No labels for {video}')
        data = json.loads(label.read_text())
        annotated = tb.annotate_document(data)
        folder = output_dir/video.parent.name;folder.mkdir(exist_ok=True)
        shutil.copy2(video,folder/video.name)
        # Match the existing filename convention, including moving-camera ground_truth.json.
        (folder/label.name).write_text(json.dumps(annotated,indent=2,allow_nan=False)+'\n')
        old = sum(bool(r.get('legacy_center_rule',r)['is_violation']) for r in data['frames'])
        new = sum(r['is_violation'] is True for r in annotated['frames'])
        review = sum(r['is_violation'] is None for r in annotated['frames'])
        report['clips'].append({'video':str(video.relative_to(input_dir)), 'car_frames':len(data['frames']),
                                'old_center_violation_car_frames':old,'boundary_violation_car_frames':new,
                                'review_car_frames':review})
        if make_previews:
            ids=sorted({r['frame_idx'] for r in annotated['frames']})
            frame_id=ids[len(ids)//2]
            cap=cv2.VideoCapture(str(video));cap.set(cv2.CAP_PROP_POS_FRAMES,frame_id)
            ok,bgr=cap.read();cap.release()
            if not ok: raise RuntimeError(f'Cannot decode {video} frame {frame_id}')
            rows=[r for r in annotated['frames'] if r['frame_idx']==frame_id]
            camera=annotated.get('camera') or rows[0]['camera']
            image=Image.fromarray(cv2.cvtColor(bgr,cv2.COLOR_BGR2RGB))
            draw_debug(image,rows,camera).save(folder/(video.stem+'_boundaries.png'))
        print(f'{video.parent.name}/{video.name}: centre={old}, boundary={new}, review={review}',flush=True)
    if not report['clips']: raise ValueError('No videos found')
    calibration=input_dir/'calibration.json'
    if calibration.exists():shutil.copy2(calibration,output_dir/calibration.name)
    (output_dir/'boundary_export_report.json').write_text(json.dumps(report,indent=2)+'\n')
    (output_dir/'README.md').write_text('''# Boundary-labelled VMAX videos

Labels: `vmax.tyre_boundaries.v1`. Videos are unchanged copies of the source render.
Pink debug edges mark the two ends of the 16 cm ground-level tread line in the
smooth rigid tyre envelope. Cyan marks the complete lower tyre profile, whose
shoulders sit above the road. No finite contact area or tyre deformation is invented.

Read `label_contract`, `tyre_boundaries`, `boundary_margin_m_interval`, and
`boundary_status` in each JSON. `is_violation` is true, false, or null (review).
The old point-rule results are under `legacy_center_rule` / `legacy_center_events`.
Scenario names still describe original centre-based trajectories; 0.05m does not
promise a 5 cm boundary violation. The updated `train_mask2former_swin_tiny.ipynb` reads this folder and trains eight
ground-tread endpoints using car crops and the helper `boundary_training.py`.
Use a fresh run; previous centre and object-query endpoint checkpoints are incompatible.

All geometric labels are amodal: `in_frame` does not establish visibility through
bodywork, other wheels, or scenery. Do not treat these polylines as visible tyre masks.
Moving-camera calibration is stored on each frame. Invert a ground homography only
for the z=0 tread endpoints; the raised shoulder outline needs 3D projection.

`boundary_export_report.json` compares old and new frame counts.
''')
    return report


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input',type=Path,default=Path(__file__).parent/'output')
    parser.add_argument('--output',type=Path,default=Path(__file__).parent/'output_boundaries')
    parser.add_argument('--no-previews',action='store_true')
    args=parser.parse_args()
    export(args.input,args.output,not args.no_previews)
