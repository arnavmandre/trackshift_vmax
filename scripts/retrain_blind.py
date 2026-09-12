"""Reproducible CPU fine-tuning; select only on validation, test once afterwards."""
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

ROOT = Path('retraining_out')
PLAN = dict(development_seed=2026091101, development_scenes=96,
            blind_seed=2026091102, blind_scenes=40, fps=12, seconds=2,
            width=480, height=270, stages=3, iterations_per_stage=200,
            batch=8, crop=256, learning_rates=[0.0005, 0.0002, 0.00008],
            detector_threshold=0.25,
            selection='maximum validation event F1, then minimum margin P95; no test selection',
            limitations='Synthetic same-track/assets, exact camera calibration; not real F1 accuracy')


def run(*args):
    print('+', ' '.join(map(str,args)), flush=True)
    subprocess.run([sys.executable, *map(str,args)], check=True)


def dump(path, value):
    Path(path).write_text(json.dumps(value, indent=2))


def partition(folder):
    """Training process receives a file containing training annotations only."""
    manifest=json.loads((folder/'manifest.json').read_text())
    truth=json.loads((folder/'sealed/labels.json').read_text())
    for split in ('train','validation'):
        clips=[c for c in manifest['clips'] if c['split']==split]
        if not clips:raise ValueError(f'empty {split} split')
        dump(folder/f'{split}.json', {**manifest,'clips':clips})
        dump(folder/'sealed'/f'{split}.json', {'clips':{c['id']:truth['clips'][c['id']] for c in clips}})
    print('Development split sizes:', {s:sum(c['split']==s for c in manifest['clips']) for s in ('train','validation','test')}, flush=True)


def evaluate(folder, split, weights, out, manifest='manifest.json', labels='labels.json'):
    run('-m','vmax_vision.cli','blind','infer','--manifest',folder/manifest,
        '--split',split,'--weights',weights,'--out',out/'predictions')
    run('-m','vmax_vision.cli','blind','score','--manifest',folder/manifest,
        '--runs',out/'predictions','--labels',folder/'sealed'/labels,'--out',out/'metrics')
    return json.loads((out/'metrics/evaluation.json').read_text())['summary']


def rank(summary):
    p,r=summary['event_precision'] or 0,summary['event_recall'] or 0
    f1=2*p*r/(p+r) if p+r else 0
    error=summary['margin_p95_m']
    return f1, -error if error is not None else -float('inf')


def generate(folder, count, seed):
    run('-m','vmax_vision.cli','generate','--out',folder,'--count',count,'--seed',seed,
        '--fps',PLAN['fps'],'--seconds',PLAN['seconds'],'--width',PLAN['width'],'--height',PLAN['height'])


def main():
    ROOT.mkdir(exist_ok=False)
    dump(ROOT/'protocol.json',PLAN)  # committed configuration, before any labels are scored
    dev=ROOT/'development'
    reuse=os.environ.get('VMAX_DEVELOPMENT_REUSE')
    if reuse and (Path(reuse)/'manifest.json').exists():
        from vmax_vision.blind import validate_manifest
        source=Path(reuse)
        manifest=validate_manifest(source/'manifest.json')
        if manifest['seed']!=PLAN['development_seed'] or len(manifest['clips'])!=PLAN['development_scenes']:
            raise ValueError('recovered development dataset does not match fixed protocol')
        shutil.copytree(source,dev)
        print('Reused hash-verified development videos from earlier render',flush=True)
    else:
        generate(dev,PLAN['development_scenes'],PLAN['development_seed'])
    partition(dev)
    previous=Path('pipeline_out/vmaxnet.pt')
    summaries=[]
    for stage in range(PLAN['stages']):
        weights=ROOT/f'candidate_{stage+1}.pt'
        run('-m','vmax_vision.train','--manifest',dev/'train.json','--labels',dev/'sealed/train.json',
            '--out',weights,'--resume',previous,'--iterations',PLAN['iterations_per_stage'],
            '--batch',PLAN['batch'],'--crop',PLAN['crop'],'--lr',PLAN['learning_rates'][stage],
            '--threads',2,'--seed',117+stage,'--max-frames',1200,'--frame-stride',2)
        metrics=evaluate(dev,'validation',weights,ROOT/f'validation_{stage+1}', 'validation.json','validation.json')
        summaries.append({'stage':stage+1,'weights':str(weights),'metrics':metrics})
        dump(ROOT/'validation_candidates.json',summaries)
        previous=weights
    winner=max(summaries,key=lambda row:rank(row['metrics']))
    selected=ROOT/'selected.pt'
    shutil.copyfile(winner['weights'],selected)
    from vmax_vision.evidence import sha256
    dump(ROOT/'selection.json',{'selected_stage':winner['stage'],'weights_sha256':sha256(selected),
         'validation_metrics':winner['metrics'],'rule':PLAN['selection'],'test_opened':False})
    # New scene families, never decoded by training or checkpoint selection.
    blind=ROOT/'blind_clips'
    generate(blind,PLAN['blind_scenes'],PLAN['blind_seed'])
    manifest=json.loads((blind/'manifest.json').read_text())
    for clip in manifest['clips']:clip['split']='test'
    dump(blind/'manifest.json',manifest)
    # Seal both models' predictions before looking at any blind annotations.
    for name,weights in [('baseline',Path('pipeline_out/vmaxnet.pt')),('retrained',selected)]:
        run('-m','vmax_vision.cli','blind','infer','--manifest',blind/'manifest.json',
            '--weights',weights,'--out',ROOT/name/'predictions')
    results={}
    for name in ('baseline','retrained'):
        run('-m','vmax_vision.cli','blind','score','--manifest',blind/'manifest.json',
            '--runs',ROOT/name/'predictions','--labels',blind/'sealed/labels.json','--out',ROOT/name/'metrics')
        results[name]=json.loads((ROOT/name/'metrics/evaluation.json').read_text())['summary']
    dump(ROOT/'comparison.json',results)
    run('-m','vmax_vision.cli','review','--manifest',blind/'manifest.json',
        '--runs',ROOT/'retrained/predictions','--out',ROOT/'steward_review')
    print('FINAL BLIND COMPARISON',json.dumps(results,indent=2),flush=True)
    summary=os.environ.get('GITHUB_STEP_SUMMARY')
    if summary:
        with open(summary,'a') as f:
            f.write('## Fresh synthetic blind test\n\nSame track and assets; exact synthetic camera calibration. Not real F1 accuracy.\n\n')
            f.write('| Model | Event precision | Event recall | False reports | Misses | Margin P95 (m) |\n|---|---|---|---|---|---|\n')
            for name,m in results.items():
                f.write(f"| {name} | {m['event_precision']} | {m['event_recall']} | {m['false_reports']} | {m['missed_events']} | {m['margin_p95_m']} |\n")

if __name__=='__main__':main()
