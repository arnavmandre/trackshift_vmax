"""Temporary, explicitly artificial browser-test data. Not a demo or benchmark."""
import hashlib,json
from pathlib import Path
import subprocess


def write(root):
    root=Path(root);ds=root/'fresh_blind';ds.mkdir(parents=True,exist_ok=True)
    video=ds/'test.mp4'
    subprocess.run(['ffmpeg','-y','-loglevel','error','-f','lavfi','-i','testsrc2=size=480x270:rate=12','-t','2','-c:v','libx264','-pix_fmt','yuv420p','-movflags','+faststart',str(video)],check=True)
    digest=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
    save=lambda p,o:p.write_text(json.dumps(o))
    modelhash='1'*64
    manifest={'dataset_kind':'browser_test_fixture','clips':[{'id':'fixture','video':'test.mp4','video_sha256':digest(video),'fps':12,'frames':24,'camera_spec':{'resolution':[480,270]},'calibration':{'homography':[[1,0,0],[0,1,0],[0,0,1]]}}]}
    save(ds/'manifest.json',manifest);save(root/'selection.json',{'weights_sha256':modelhash,'winner':{'metrics':{'threshold':.3}}})
    save(root/'blind_predictions.json',{'fixture':'artificial browser fixture; not model predictions'})
    save(root/'blind_predictions.json.receipt.json',{'predictions_sha256':digest(root/'blind_predictions.json'),'manifest_sha256':digest(ds/'manifest.json'),'weights_sha256':modelhash})
    side=root/'blind_metrics';side.mkdir(exist_ok=True)
    frames=[{'frame':i,'position':[i,0],'margin_m':.2 if 5<=i<=8 else -.2,'score':.8,'box':[100,80,200,140],'keypoints':[[115,130,1],[120,100,1],[180,130,1],[180,100,1]]} for i in range(24)]
    save(side/'review_tracks.json',{'fixture':[frames]})
    save(root/'review_integrity.json',{'prediction_sha256':digest(root/'blind_predictions.json'),'review_tracks_sha256':digest(side/'review_tracks.json')})
    # Invalid sealed annotations demonstrate they are never parsed by the server.
    (ds/'sealed').mkdir(exist_ok=True);(ds/'sealed/labels.json').write_text('NOT VALID JSON — MUST NEVER BE OPENED')
    return root

if __name__=='__main__':
    import sys
    write(sys.argv[1])
