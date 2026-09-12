"""Local steward console. Standard library only; never reads sealed annotations."""
import argparse
import functools
import hashlib
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import math
import mimetypes
from pathlib import Path
import re
from urllib.parse import unquote, urlsplit

UI=Path(__file__).parent/'steward_ui'

def json_read(path,default=None):
    return json.loads(path.read_text()) if path.is_file() else default

def inside(root,relative):
    p=(root/relative).resolve()
    if not p.is_relative_to(root.resolve()):raise ValueError('path outside data directory')
    return p

def boundary(H):
    lines=[]
    for offset in (-7,7):
        world=[(x,-offset) for x in range(-30,1,2)]
        world += [((40+offset)*math.sin(a),40-(40+offset)*math.cos(a)) for a in [i*math.pi/120 for i in range(61)]]
        world += [(40+offset,y) for y in range(42,71,2)]
        uv=[]
        for x,y in world:
            z=H[2][0]*x+H[2][1]*y+H[2][2]
            uv.append([(H[0][0]*x+H[0][1]*y+H[0][2])/z,(H[1][0]*x+H[1][1]*y+H[1][2])/z] if abs(z)>1e-8 else None)
        lines.append(uv)
    return lines

def events(track):
    result=[]
    for f in sorted(track,key=lambda f:f['frame']):
        if f['margin_m']<=0:continue
        if not result or f['frame']!=result[-1]['end']+1:
            result.append({'start':f['frame'],'end':f['frame'],'peak':f['frame'],'margin':f['margin_m']})
        else:
            e=result[-1];e['end']=f['frame']
            if f['margin_m']>e['margin']:e.update(peak=f['frame'],margin=f['margin_m'])
    return result

class Session:
    def __init__(self,root):
        self.root=Path(root).resolve();self.dataset=self.root/'fresh_blind';self.videos={}
        self.manifest=json_read(self.dataset/'manifest.json',{'clips':[]})
        self.tracks=json_read(self.root/'blind_metrics/review_tracks.json',{})
        self.selection=json_read(self.root/'selection.json',{})
        self.evaluation=json_read(self.root/'blind_metrics/evaluation.json')
        self.receipt=json_read(self.root/'blind_predictions.json.receipt.json',{})
        self.verification='not available'
        pred=self.root/'blind_predictions.json'
        if pred.is_file() and self.receipt:
            actual=hashlib.sha256(pred.read_bytes()).hexdigest()
            if actual!=self.receipt.get('predictions_sha256'):raise ValueError('prediction receipt mismatch')
            if self.selection.get('weights_sha256')!=self.receipt.get('weights_sha256'):raise ValueError('selected model differs from prediction receipt')
            manifest_hash=hashlib.sha256((self.dataset/'manifest.json').read_bytes()).hexdigest()
            if manifest_hash!=self.receipt.get('manifest_sha256'):raise ValueError('manifest differs from prediction receipt')
            self.verification='prediction, model identity and manifest hashes verified'
        for c in self.manifest['clips']:
            cid=c['id']
            if not re.fullmatch(r'[A-Za-z0-9_-]+',cid):raise ValueError('unsafe clip identifier')
            p=inside(self.dataset,c['video'])
            if not p.is_file():raise ValueError(f'missing original video: {cid}')
            if hashlib.sha256(p.read_bytes()).hexdigest()!=c['video_sha256']:raise ValueError(f'video hash mismatch: {cid}')
            self.videos[cid]=p
    def payload(self):
        clips=[]
        for c in self.manifest['clips']:
            tracks=self.tracks.get(c['id'],[])
            # Review geometry must be derived from verified predictions, not unverified sidecar edits.
            # The recovery workflow hashes this sidecar after building it from experiment scoring.
            checks=json_read(self.root/'review_integrity.json',{})
            sidecar=self.root/'blind_metrics/review_tracks.json'
            trusted=bool(self.verification!='not available' and checks.get('prediction_sha256')==self.receipt.get('predictions_sha256') and sidecar.is_file() and checks.get('review_tracks_sha256')==hashlib.sha256(sidecar.read_bytes()).hexdigest())
            if tracks and not trusted:
                tracks=[]
            enriched=[{'id':i+1,'frames':t,'events':events(t)} for i,t in enumerate(tracks)]
            fs=[f for t in tracks for f in t]
            tel=json_read(inside(self.dataset,c['telemetry']),[]) if c.get('telemetry') else []
            clips.append({'id':c['id'],'video':'/media/'+c['id'],'video_hash':c['video_sha256'],
              'fps':c['fps'],'frames':c['frames'],'resolution':c.get('camera_spec',{}).get('resolution',[480,270]),
              'tracks':enriched,'analysed':trusted and c['id'] in self.tracks,
              'mean_score':sum(f['score'] for f in fs)/len(fs) if fs else None,
              'coverage':len({f['frame'] for f in fs})/c['frames'],
              'boundary':boundary(c['calibration']['homography']) if self.manifest.get('dataset_kind')=='synthetic_fixed_vmax_corner' else [],
              'telemetry':{'samples':len(tel),'source':'Synthetic noisy position; not fused into model predictions'},
              'candidate_count':sum(len(t['events']) for t in enriched)})
        return {'clips':clips,'model_hash':self.selection.get('weights_sha256'),
                'verification':self.verification,'metrics':self.evaluation,'model':'YOLO26 four-contact pose',
                'threshold':self.selection.get('winner',{}).get('metrics',{}).get('threshold'),
                'score_notice':'Detector scores are uncalibrated; they are not offence probabilities.'}

class Handler(BaseHTTPRequestHandler):
    def __init__(self,*args,session,**kwargs):self.session=session;super().__init__(*args,**kwargs)
    def do_HEAD(self):self.respond(head=True)
    def do_GET(self):self.respond()
    def respond(self,head=False):
        path=unquote(urlsplit(self.path).path)
        if path=='/api/session':
            data=json.dumps(self.session.payload(),allow_nan=False).encode()
            self.send_response(200);self.send_header('Content-Type','application/json');self.send_header('Content-Length',str(len(data)));self.send_header('Cache-Control','no-store');self.end_headers()
            if not head:self.wfile.write(data)
            return
        if path.startswith('/media/'):
            p=self.session.videos.get(path[len('/media/'):])
        else:
            p={'/':UI/'index.html','/app.js':UI/'app.js','/style.css':UI/'style.css'}.get(path)
        if p is None or not p.is_file():self.send_error(404);return
        size=p.stat().st_size;start=0;end=size-1;status=200
        if self.headers.get('Range'):
            match=re.fullmatch(r'bytes=(\d*)-(\d*)',self.headers['Range'])
            try:
                if not match or not any(match.groups()):raise ValueError()
                a,b=match.groups()
                if a:start=int(a);end=min(int(b),end) if b else end
                else:
                    n=int(b)
                    if n<=0:raise ValueError()
                    start=max(0,size-n)
                if start>=size or start>end:raise ValueError()
                status=206
            except ValueError:
                self.send_response(416);self.send_header('Content-Range',f'bytes */{size}');self.send_header('Content-Length','0');self.end_headers();return
        self.send_response(status)
        self.send_header('Content-Type',mimetypes.guess_type(p.name)[0] or 'application/octet-stream')
        self.send_header('Accept-Ranges','bytes');self.send_header('Content-Length',str(end-start+1))
        if status==206:self.send_header('Content-Range',f'bytes {start}-{end}/{size}')
        self.end_headers()
        if not head:
            try:
                with p.open('rb') as f:
                    f.seek(start);remaining=end-start+1
                    while remaining:
                        chunk=f.read(min(256*1024,remaining))
                        if not chunk:break
                        self.wfile.write(chunk);remaining-=len(chunk)
            except (BrokenPipeError,ConnectionResetError):pass
    def log_message(self,*args):pass

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--data',default='trained_model');p.add_argument('--port',type=int,default=8000);a=p.parse_args()
    session=Session(a.data)
    server=ThreadingHTTPServer(('127.0.0.1',a.port),functools.partial(Handler,session=session))
    print(f'Open http://127.0.0.1:{a.port} — {len(session.videos)} clips loaded. Ctrl+C to stop.',flush=True)
    try:server.serve_forever()
    except KeyboardInterrupt:pass
    finally:server.server_close()
