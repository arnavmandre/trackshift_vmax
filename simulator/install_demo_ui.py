"""Add demo videos to the running VMAXPROTO queue without prediction sidecars."""
import argparse
import json
from pathlib import Path
import shutil
from .evidence import sha256


def install(source, destination, replace=False):
    source, destination = Path(source), Path(destination)
    manifest_path = destination/'clips_manifest.json'
    original = manifest_path.read_text()
    existing = json.loads(original)
    demo = json.loads((source/'manifest.json').read_text())
    entries = []
    for clip in demo['clips']:
        video = (source/clip['video']).resolve()
        if not video.is_relative_to(source.resolve()):raise ValueError('video outside source')
        if sha256(video) != clip['sha256']:raise ValueError('demo video hash mismatch')
        cid = 'demo_'+clip['sha256'][:16]
        shutil.copy2(video,destination/f'{cid}.mp4')
        entries.append(dict(id=cid,name=f"Demo - {video.stem.replace('_',' ').title()} (1080p60, playback).mp4",
                            fps=demo['fps'],playback_only=True))
    backup = destination/'clips_manifest.before_demo.json'
    if not backup.exists():backup.write_text(original)
    ids = {c['id'] for c in entries}
    if replace:
        import hashlib
        snapshot = destination/f'clips_manifest.backup_{hashlib.sha256(original.encode()).hexdigest()[:12]}.json'
        if not snapshot.exists():snapshot.write_text(original)
    existing['clips'] = entries if replace else entries+[c for c in existing['clips'] if c['id'] not in ids]
    temporary = destination/'clips_manifest.demo.tmp'
    temporary.write_text(json.dumps(existing,indent=2))
    temporary.replace(manifest_path)
    print(f'Installed {len(entries)} demo clips; queue {"replaced" if replace else "extended"}. Original files preserved.')


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source',required=True)
    parser.add_argument('--destination',required=True)
    parser.add_argument('--replace',action='store_true',help='replace the website queue, retaining old videos and a manifest backup')
    args=parser.parse_args()
    install(args.source,args.destination,args.replace)
