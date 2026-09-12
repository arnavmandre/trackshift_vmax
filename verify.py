"""Validate v2 encoded outputs and unchanged canonical world labels; export mesh."""
import json, pathlib, subprocess
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import render as r

def export_car():
    chunks=[r.body((218,27,39)).array()]
    for x,y in r.g.CONTACT:
        w=r.wheel(int(y>0)).array().copy();w[:,:3]+=[x,y,.36];chunks.append(w)
    mesh=np.concatenate(chunks);root=r.OUT/'assets';root.mkdir(exist_ok=True)
    mats={};indices=[]
    for tri in mesh.reshape(-1,3,12):
        key=tuple(float(x) for x in tri[0,6:10])
        if key not in mats:mats[key]=f'material_{len(mats)}'
        indices.append(mats[key])
    with (root/'formula_car.obj').open('w') as f:
        f.write('mtllib formula_car.mtl\n')
        for v in mesh:f.write('v '+' '.join(f'{x:.7f}' for x in v[:3])+'\n')
        for v in mesh:f.write('vn '+' '.join(f'{x:.7f}' for x in v[3:6])+'\n')
        for v in mesh:f.write('vt '+' '.join(f'{x:.7f}' for x in v[10:12])+'\n')
        previous=None
        for i,mat in enumerate(indices):
            if mat!=previous:f.write('usemtl '+mat+'\n');previous=mat
            f.write('f '+' '.join(f'{j}/{j}/{j}' for j in range(i*3+1,i*3+4))+'\n')
    with (root/'formula_car.mtl').open('w') as f:
        for key,name in mats.items():
            f.write(f'newmtl {name}\nKd {key[0]} {key[1]} {key[2]}\nKs 0.15 0.15 0.15\nNs 50\n')
            if key[3]==10:f.write('map_Kd livery.png\n')
    atlas=Image.new('RGB',(1024,256),(10,24,33));d=ImageDraw.Draw(atlas);font='/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf'
    d.text((50,8),'VMAX',font=ImageFont.truetype(font,170),fill=(242,247,244));d.polygon([(715,35),(810,35),(720,206),(625,206)],fill=(245,64,45));d.text((825,75),'RACING',font=ImageFont.truetype(font,30),fill=(242,247,244));atlas.save(root/'livery.png')
    return {'triangles':len(mesh)//3,'bounds_min_m':mesh[:,:3].min(0).tolist(),'bounds_max_m':mesh[:,:3].max(0).tolist()}

def probe(path,n,fps):
    meta=json.loads(subprocess.check_output(['ffprobe','-v','error','-count_frames','-select_streams','v:0','-show_entries','stream=width,height,nb_read_frames,r_frame_rate','-of','json',str(path)]))['streams'][0]
    assert meta['width']==1280 and meta['height']==720 and int(meta['nb_read_frames'])==n and meta['r_frame_rate']==f'{fps}/1', (path,meta)

def independent(rows):
    worst=0.
    for row in rows:
        p=np.array(list(row['contact_points_world'].values()));x,y=p.T
        # Generated paths stay in these arc, entry, and exit nearest-point domains.
        assert np.all((x>=-30)&(y<=70))
        dist=np.where(x<0,abs(y),np.where(y>40,abs(x-40),abs(np.hypot(x,y-40)-40)))-7
        error=float(abs(dist-np.array(list(row['corner_excess_m'].values()))).max());worst=max(worst,error)
        assert error<1e-9 and bool(dist.min()>0)==row['is_violation']
    return worst

def main():
    report={'fixed_camera_clips':24,'fixed_camera_resolution':[1280,720],'world_labels_preserved':True,'distance_error_m':{},'scenarios':{}}
    for source in sorted((r.ROOT/'data').glob('*/ground_truth.json')):
        truth=json.loads(source.read_text());name=source.parent.name;folder=r.OUT/name
        report['distance_error_m'][name]=independent(truth['frames']);report['scenarios'][name]=truth['events']
        for cam in r.CAMS:
            enriched=json.loads((folder/(cam['name']+'.json')).read_text())
            for expected,actual in zip(truth['frames'],enriched['frames']):
                assert all(actual[k]==v for k,v in expected.items())
            assert len(truth['frames'])==len(enriched['frames'])
            probe(folder/(cam['name']+'.mp4'),96,24)
    showcase=json.loads((r.OUT/'race_pace/ground_truth.json').read_text());rows=showcase['frames']
    probe(r.OUT/'race_pace/race_pace.mp4',240,60)
    report['distance_error_m']['race_pace']=independent(rows)
    speed=np.array([row['speed_mps'] for row in rows]);lat=np.array([row['lateral_acceleration_g'] for row in rows]);pos=np.array([row['world_position'] for row in rows])
    velocity=np.gradient(pos,1/60,axis=0,edge_order=2)
    assert abs(np.linalg.norm(velocity,axis=1)-speed).max()<1e-10
    assert abs(lat).max()<3 and abs(max(row['min_excess_m'] for row in rows)-.15)<1e-8
    report['race_pace']={'speed_range_kmh':(np.array([speed.min(),speed.max()])*3.6).tolist(),'peak_lateral_g':float(abs(lat).max()),'peak_margin_m':max(row['min_excess_m'] for row in rows),'events':showcase['events'],'opengl_projection_error_px':showcase['opengl_projection_error_px']}
    report['cameras']=r.calibrate()
    report['exported_car']=export_car()
    (r.OUT/'validation.json').write_text(json.dumps(report,indent=2))
    lines=['# VMAX v2 validation','', 'All 25 clips passed frame-count, resolution and fps checks. All 24 controlled-scenario label files preserve every original ground-truth field exactly.','', '| Check | Result |','|---|---|',f"| Independent tyre-distance agreement | {max(report['distance_error_m'].values()):.3g} m maximum error |",f"| Race-pace speed range | {speed.min()*3.6:.2f}–{speed.max()*3.6:.2f} km/h |",f"| Race-pace peak lateral acceleration | {abs(lat).max():.3f} g |",'| Race-pace peak violation margin | 0.150000 m |',f"| Race-pace camera matrix agreement | {showcase['opengl_projection_error_px']:.3g} pixels |",'', '| Camera | Curved-boundary coverage | Matrix agreement (px) | Float32 in-frame error (px) |','|---|---:|---:|---:|']
    for name,cam in report['cameras'].items():lines.append(f"| {name} | {cam['coverage_percent']:.3f}% | {cam['opengl_vs_pinhole_error_px']:.3g} | {cam['float32_in_frame_error_px']:.3g} |")
    lines+=['','The race-pace animation is kinematic, not a validated tyre-force, aerodynamic, or suspension simulation. Lateral acceleration is calculated from the trajectory and constrained to the chosen 3 g budget.','', 'See output/validation.json for full scenario event counts, camera checks and mesh bounds.']
    (r.ROOT/'VALIDATION.md').write_text('\n'.join(lines)+'\n');print('\n'.join(lines))

if __name__=='__main__':main()
