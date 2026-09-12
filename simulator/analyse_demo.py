"""Run both existing models on the demo queue without reading sealed labels."""
import argparse
import json
from pathlib import Path
import torch
from ultralytics import YOLO
import vmax_export as export
from vmax_live_server import run_better_model


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source',required=True)
    parser.add_argument('--destination',required=True)
    args=parser.parse_args()
    source,destination=Path(args.source),Path(args.destination)
    demo=json.loads((source/'manifest.json').read_text())
    queue_path=destination/'clips_manifest.json'
    queue=json.loads(queue_path.read_text())
    selection=json.loads(export.SELECTION.read_text())
    assert export.e.digest(export.WEIGHTS)==selection['weights_sha256']
    export.BLIND_ROOT=source
    model=YOLO(str(export.WEIGHTS))
    device='cuda' if torch.cuda.is_available() else 'cpu'
    for clip in demo['clips']:
        assert export.e.digest(source/clip['video'])==clip['sha256']
        cid='demo_'+clip['sha256'][:16]
        item=next(c for c in queue['clips'] if c['id']==cid)
        camera=clip['camera']
        entry=dict(id=cid,video=clip['video'],fps=demo['fps'],frames=clip['frames'],
                   camera_spec=camera,calibration={'homography':camera['ground_plane_homography']})
        export.export_clip(model,device,entry,selection['winner']['metrics']['threshold'],destination,item['name'][:-4])
    del model
    if torch.cuda.is_available():torch.cuda.empty_cache()
    for item in queue['clips']:
        print(f"SECOND MODEL: {item['id']}",flush=True)
        result=run_better_model(destination,item['id'])
        if result['status']!='done':raise RuntimeError(result)
        print(f"BOTH MODELS COMPLETE: {item['id']}",flush=True)
    for item in queue['clips']:
        item['playback_only']=False
        item['name']=item['name'].replace('1080p60, playback','1080p60, analysed')
        prediction_path=destination/f"{item['id']}.json"
        prediction=json.loads(prediction_path.read_text())
        prediction['video']['name']=item['name']
        prediction_path.write_text(json.dumps(prediction,indent=2))
    temporary=destination/'clips_manifest.analysed.tmp'
    temporary.write_text(json.dumps(queue,indent=2))
    temporary.replace(queue_path)
    print('ALL EIGHT CLIPS ANALYSED AND PUBLISHED',flush=True)


if __name__=='__main__':main()
