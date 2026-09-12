"""Build a portable original-video review bundle, without reading ground truth."""
import argparse
import json
from pathlib import Path
import shutil
from .blind import validate_manifest,resolve
from .evidence import sha256


def build(manifest_path,out,runs=None):
    manifest=validate_manifest(manifest_path);root=Path(manifest_path).parent;out=Path(out)
    if out.exists() and any(out.iterdir()):raise ValueError('review output must be empty')
    out.mkdir(parents=True,exist_ok=True)
    payload={'clips':[]}
    receipt=None
    if runs:
        receipt=json.loads((Path(runs)/'receipt.json').read_text())
        if receipt['manifest_sha256']!=sha256(manifest_path):raise ValueError('run manifest mismatch')
    for c in manifest['clips']:
        video=resolve(root,c['video']);cid=c['id']
        if Path(cid).name!=cid or cid in ('.','..'):raise ValueError('unsafe clip id')
        dest=out/'media'/cid;dest.mkdir(parents=True)
        shutil.copyfile(video,dest/'original.mp4')
        item={**c,'video':f'media/{cid}/original.mp4','result':None}
        if receipt and f'{cid}.json' in receipt['outputs']:
            rp=resolve(runs,f'{cid}.json')
            if sha256(rp)!=receipt['outputs'][f'{cid}.json']:raise ValueError('prediction receipt mismatch')
            item['result']=json.loads(rp.read_text())
        payload['clips'].append(item)
    text=Path(__file__).with_name('review.html').read_text()
    # Prevent user-controlled JSON from terminating the script element.
    text=text.replace('__REVIEW_DATA__',json.dumps(payload).replace('<','\\u003c'))
    (out/'index.html').write_text(text)
    shutil.copyfile(Path(__file__).with_name('review.js'),out/'review.js')
    print(f'Review bundle: {out}/index.html\nServe: python -m vmax_vision.serve --port 8000 --directory {out}')


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--manifest',required=True);p.add_argument('--out',required=True);p.add_argument('--runs')
    a=p.parse_args(argv);build(a.manifest,a.out,a.runs)

if __name__=='__main__':main()
