"""Local evidence server with HTTP byte ranges for reliable original-video seeking."""
import argparse
import functools
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
import os
import re


class RangeHandler(SimpleHTTPRequestHandler):
    def send_head(self):
        path=self.translate_path(self.path)
        request=self.headers.get('Range')
        self.remaining=None
        if not request or not os.path.isfile(path):
            return super().send_head()
        size=os.path.getsize(path)
        match=re.fullmatch(r'bytes=(\d*)-(\d*)',request.strip())
        if not match or not any(match.groups()) or size==0:
            return self.invalid_range(size)
        first,last=match.groups()
        if first:
            start=int(first);end=min(int(last),size-1) if last else size-1
        else:
            suffix=int(last)
            if suffix==0:return self.invalid_range(size)
            start=max(0,size-suffix);end=size-1
        if start>=size or start>end:return self.invalid_range(size)
        f=open(path,'rb');f.seek(start);self.remaining=end-start+1
        self.send_response(206)
        self.send_header('Content-Type',self.guess_type(path))
        self.send_header('Content-Range',f'bytes {start}-{end}/{size}')
        self.send_header('Content-Length',str(self.remaining))
        self.send_header('Last-Modified',self.date_time_string(os.fstat(f.fileno()).st_mtime))
        self.end_headers()
        return f

    def invalid_range(self,size):
        self.send_response(416);self.send_header('Content-Range',f'bytes */{size}')
        self.send_header('Content-Length','0');self.end_headers();return None

    def end_headers(self):
        self.send_header('Accept-Ranges','bytes')
        super().end_headers()

    def copyfile(self,source,outputfile):
        remaining=self.remaining
        if remaining is None:
            return super().copyfile(source,outputfile)
        while remaining:
            data=source.read(min(65536,remaining))
            if not data:break
            outputfile.write(data);remaining-=len(data)


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--directory',required=True)
    p.add_argument('--port',type=int,default=8000);p.add_argument('--bind',default='127.0.0.1')
    a=p.parse_args(argv)
    server=ThreadingHTTPServer((a.bind,a.port),functools.partial(RangeHandler,directory=a.directory))
    print(f'Open http://{a.bind}:{a.port}',flush=True)
    try:server.serve_forever()
    except KeyboardInterrupt:pass
    finally:server.server_close()

if __name__=='__main__':main()
