"""Serves the VMAXPROTO steward-review prototype with real model predictions
auto-loaded, via the prototype's own documented integration surface.

The prototype's HTML file on disk is never modified: it is read fresh and one
<script src="/autoload.js"> tag is appended before </body> at serve time.
Standard library only, matching review_server.py's approach.
"""
import argparse
import functools
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
from urllib.parse import unquote, urlsplit

REPO = Path('C:/Users/arnav/OneDrive/Desktop/VMAX_Trackshift')
PROTO_HTML = REPO / 'VMAXPROTO' / 'VMAX-Steward-Review-v2.1.html'
AUTOLOAD_JS = REPO / 'VMAXPROTO' / 'autoload.js'


class Handler(BaseHTTPRequestHandler):
    def __init__(self, *args, data_dir, **kwargs):
        self.data_dir = data_dir
        super().__init__(*args, **kwargs)

    def do_GET(self):
        path = unquote(urlsplit(self.path).path)
        try:
            if path == '/':
                html = PROTO_HTML.read_text(encoding='utf-8')
                html = html.replace('</body>', '<script src="/autoload.js"></script></body>')
                self.respond_bytes(html.encode('utf-8'), 'text/html; charset=utf-8')
            elif path == '/autoload.js':
                self.respond_bytes(AUTOLOAD_JS.read_bytes(), 'application/javascript')
            elif path == '/api/clips':
                manifest = json.loads((self.data_dir / 'clips_manifest.json').read_text())
                for c in manifest['clips']:
                    c['video_url'] = f"/video/{c['id']}.mp4"
                    c['predictions_url'] = f"/predictions/{c['id']}.json"
                self.respond_bytes(json.dumps(manifest).encode(), 'application/json')
            elif path.startswith('/video/'):
                self.respond_file(self.data_dir / path[len('/video/'):], 'video/mp4')
            elif path.startswith('/predictions/'):
                self.respond_file(self.data_dir / path[len('/predictions/'):], 'application/json')
            else:
                self.send_error(404)
        except FileNotFoundError:
            self.send_error(404)

    def respond_bytes(self, data, content_type):
        self.send_response(200)
        self.send_header('Content-Type', content_type)
        self.send_header('Content-Length', str(len(data)))
        self.send_header('Cache-Control', 'no-store')
        self.end_headers()
        self.wfile.write(data)

    def respond_file(self, path, content_type):
        if not path.is_file() or not path.resolve().is_relative_to(self.data_dir.resolve()):
            self.send_error(404)
            return
        self.respond_bytes(path.read_bytes(), content_type)

    def log_message(self, *args):
        pass


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--data', default=r'C:/Users/arnav/trackshift_runs/kerb960/vmax_live')
    p.add_argument('--port', type=int, default=8010)
    a = p.parse_args()
    data_dir = Path(a.data)
    if not (data_dir / 'clips_manifest.json').is_file():
        raise SystemExit(f'No clips_manifest.json in {data_dir} — run vmax_export.py first')
    server = ThreadingHTTPServer(('127.0.0.1', a.port), functools.partial(Handler, data_dir=data_dir))
    print(f"Open http://127.0.0.1:{a.port} — friend's prototype UI with live model predictions auto-loaded. Ctrl+C to stop.", flush=True)
    try:
        server.serve_forever()
    finally:
        server.server_close()
