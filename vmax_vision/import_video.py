"""Import a local original video with an explicit surveyed planar circuit map.

Calibration is supplied, not silently inferred from the simulator. Predictions
remain experimental until validated on representative real footage.
"""
import argparse
import json
from pathlib import Path
import shutil
import subprocess
import numpy as np
from .evidence import sha256
from .circuit import CircuitMap


def import_video(video,calibration,out):
    video=Path(video);cal=json.loads(Path(calibration).read_text())
    CircuitMap(cal['track_polygon_m'])
    H=np.asarray(cal['homography'],float)
    if H.shape!=(3,3) or not np.isfinite(H).all() or np.linalg.matrix_rank(H)!=3:raise ValueError('invalid homography')
    if not np.isfinite(cal['sigma_m']) or cal['sigma_m']<=0:raise ValueError('real calibration requires positive measured/assumed sigma_m')
    if cal.get('ground_plane')!='locally_planar':raise ValueError('explicit ground_plane: locally_planar declaration required')
    probe=json.loads(subprocess.check_output(['ffprobe','-v','error','-select_streams','v:0','-count_frames','-show_entries','stream=avg_frame_rate,r_frame_rate,nb_read_frames,width,height','-of','json',str(video)]))['streams'][0]
    def rate(s):a,b=s.split('/');return float(a)/float(b)
    fps=rate(probe['avg_frame_rate']);nominal=rate(probe['r_frame_rate'])
    if abs(fps-nominal)>1e-3:raise ValueError('variable-rate input: create a CFR working copy while preserving the original')
    dest=Path(out)
    if dest.exists() and any(dest.iterdir()):raise ValueError('output directory must be empty')
    dest.mkdir(parents=True,exist_ok=True);target=dest/('original'+video.suffix.lower());shutil.copyfile(video,target)
    digest=sha256(target)
    manifest={'schema_version':1,'dataset_kind':'external_experimental','clips':[{
        'id':digest[:16],'family_id':digest,'split':'external','video':target.name,'video_sha256':digest,
        'fps':fps,'frames':int(probe['nb_read_frames']),'calibration':{k:cal[k] for k in ('homography','sigma_m')},
        'track_polygon_m':cal['track_polygon_m'],'source_note':'Unvalidated on real footage; inspect original evidence',
    }]}
    (dest/'manifest.json').write_text(json.dumps(manifest,indent=2));return manifest


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--video',required=True);p.add_argument('--calibration',required=True);p.add_argument('--out',required=True)
    a=p.parse_args(argv);import_video(a.video,a.calibration,a.out);print(f'Imported video: {a.out}/manifest.json')

if __name__=='__main__':main()
