"""Cheap deterministic before/after gallery: python -m simulator.preview --out PATH."""
import argparse
import json
from pathlib import Path
import time
import numpy as np
from PIL import Image, ImageDraw
import geometry as g
from .appearance import surface, shadows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', required=True)
    args = parser.parse_args()
    dest = Path(args.out)
    dest.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    g.W, g.H = 960, 540
    track = g.track_mesh()
    tiles, evidence = [], []
    for name, margin in [('legal', -.05), ('crossing', .05), ('outside', .30)]:
        positions, headings, contacts, excess = g.solve(g.profile(margin))
        frame = 48
        position, heading = positions[frame], headings[frame]
        c, s = np.cos(heading), np.sin(heading)
        rotation = np.array([[c,-s,0],[s,c,0],[0,0,1]])
        mesh = [(v@rotation.T + [*position,0], col) for v,col in g.car_mesh(wheel_dz=g.kerb_height(excess[frame]))]
        rows = [dict(world_position=position, heading_rad=heading)]
        for angle, offset in [('low', [10,-13,5]), ('high', [-12,-9,10])]:
            target = np.array([*position, .2])
            cam = g.camera(name, target+offset, target, 48)
            base = g.render(track, cam)
            enhanced, world, valid = surface(base, cam)
            before = g.render(mesh, cam, base)[0]
            after = g.render(mesh, cam, shadows(enhanced,world,valid,rows))[0]
            pair = Image.new('RGB', (1920, 572), '#17202b')
            pair.paste(Image.fromarray(before),(0,32)); pair.paste(Image.fromarray(after),(960,32))
            draw = ImageDraw.Draw(pair)
            draw.text((12,10), f'{name} / {angle} - CLASSIC', fill='white')
            draw.text((972,10), 'ENHANCED', fill='white')
            pair.save(dest/f'{name}_{angle}.jpg', quality=94)
            tiles.append(pair.resize((960,286)))
            evidence.append(dict(case=name, angle=angle, point_margin_m=float(excess[frame].min()), camera=cam))
    gallery = Image.new('RGB',(960,286*len(tiles)))
    for i,tile in enumerate(tiles):gallery.paste(tile,(0,286*i))
    gallery.save(dest/'comparison.jpg',quality=94)
    (dest/'preview.json').write_text(json.dumps(dict(seconds=time.perf_counter()-started,cases=evidence),indent=2))
    print(dest/'comparison.jpg')


if __name__ == '__main__':main()
