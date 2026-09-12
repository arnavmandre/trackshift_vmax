import functools,json,tempfile,threading,unittest,urllib.request,urllib.error
from http.server import ThreadingHTTPServer
from pathlib import Path
from review_server import Session,Handler,events,inside

class ReviewTests(unittest.TestCase):
    def test_events_split_known_legal_and_missing_frames(self):
        fs=[{'frame':i,'margin_m':m} for i,m in [(0,.2),(1,.3),(2,-.1),(3,.4),(5,.5)]]
        self.assertEqual([(e['start'],e['end']) for e in events(fs)],[(0,1),(3,3),(5,5)])
    def test_no_path_escape(self):
        with tempfile.TemporaryDirectory() as d:
            with self.assertRaises(ValueError):inside(Path(d),'../secret')
    def test_empty_state_and_ranges(self):
        with tempfile.TemporaryDirectory() as d:
            s=Session(d);self.assertEqual(s.payload()['clips'],[])
            p=Path(d)/'v.mp4';p.write_bytes(b'0123456789');s.videos={'fixture':p}
            server=ThreadingHTTPServer(('127.0.0.1',0),functools.partial(Handler,session=s))
            t=threading.Thread(target=server.serve_forever,daemon=True);t.start()
            base=f'http://127.0.0.1:{server.server_port}'
            try:
                for header,expected in [('bytes=2-5',b'2345'),('bytes=-3',b'789')]:
                    req=urllib.request.Request(base+'/media/fixture',headers={'Range':header})
                    with urllib.request.urlopen(req) as r:self.assertEqual(r.status,206);self.assertEqual(r.read(),expected)
                with self.assertRaises(urllib.error.HTTPError) as cm:
                    urllib.request.urlopen(urllib.request.Request(base+'/media/fixture',headers={'Range':'bytes=20-'}))
                self.assertEqual(cm.exception.code,416)
                for path in ['/fresh_blind/sealed/labels.json','/media/../secret','/experiment.py']:
                    with self.assertRaises(urllib.error.HTTPError):urllib.request.urlopen(base+path)
            finally:server.shutdown();server.server_close();t.join()

if __name__=='__main__':unittest.main()
