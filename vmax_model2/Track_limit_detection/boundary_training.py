"""Image-only car crops and dense boundary-endpoint training for the fixed-camera VMAX dataset.

The Swin backbone and Mask2Former pixel decoder are retained. A spatial heatmap
head replaces the object-query mask/class output for this fixed set of landmarks.
This is not standard instance-segmentation Mask2Former fine-tuning.
"""
from dataclasses import asdict, dataclass
from pathlib import Path
import ctypes
import json
import math
import random
import shutil
import sys
import sysconfig

import cv2
import numpy as np
from PIL import Image
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import Dataset
from transformers import Mask2FormerForUniversalSegmentation, Mask2FormerImageProcessor

WHEELS = ['front_left', 'front_right', 'rear_left', 'rear_right']
ENDPOINTS = [f'{wheel}_{side}' for wheel in WHEELS for side in ['wheel_right', 'wheel_left']]
SCHEMA = 'vmax.tyre_boundaries.v1'

@dataclass
class Config:
    checkpoint: str = 'facebook/mask2former-swin-tiny-coco-instance'
    image_size: int = 384
    heatmap_size: int = 192
    sigma_px: float = 2.0  # heatmap pixels, equivalent to 4 crop pixels by default
    frame_stride: int = 2
    epochs: int = 40
    patience: int = 8
    batch_size: int = 2
    accumulation_steps: int = 2
    head_lr: float = 1e-3
    decoder_lr: float = 1e-4
    backbone_lr: float = 1e-5
    unfreeze_last_stage: bool = True
    seed: int = 42
    crop_padding: float = 1.65
    difference_threshold: int = 28
    min_component_area: int = 8
    background_samples_per_video: int = 12
    min_tread_width_m: float = 0.08
    max_tread_width_m: float = 0.24
    min_peak_mass: float = 0.05  # local heatmap mass; not probability of correctness
    boundary_samples: int = 161
    local_peak_radius: int = 3


def read_rgb(path):
    with Image.open(path) as image:
        return np.asarray(image.convert('RGB'))


def save_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')


def preload_nvrtc():
    if sys.platform.startswith('linux') and torch.version.cuda:
        version = torch.version.cuda
        base = Path(sysconfig.get_path('purelib'))/'nvidia'
        for folder in [base/f'cu{version.split(".")[0]}'/'lib', base/'cuda_nvrtc'/'lib']:
            path = folder/f'libnvrtc-builtins.so.{version}'
            if path.is_file():
                handle = ctypes.CDLL(str(path), mode=ctypes.RTLD_GLOBAL)
                print(f'Loaded NVRTC support: {path.name}', flush=True)
                return handle
    return None


def build_backgrounds(data_dir, train_scenarios, config, output_dir):
    """Median background from TRAIN images only. No labels or validation/test pixels."""
    data_dir, output_dir = Path(data_dir), Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    cameras = sorted({p.stem for name in train_scenarios for p in (data_dir/name).glob('*.mp4')})
    backgrounds, provenance = {}, {}
    for camera in cameras:
        frames, used = [], []
        for scenario in train_scenarios:
            video = data_dir/scenario/f'{camera}.mp4'
            if not video.is_file(): continue
            cap = cv2.VideoCapture(str(video))
            if not cap.isOpened(): raise RuntimeError(f'Cannot read {video}')
            count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            ids = np.unique(np.linspace(0, count-1, min(count,config.background_samples_per_video),dtype=int))
            try:
                for frame_id in ids:
                    cap.set(cv2.CAP_PROP_POS_FRAMES,int(frame_id));ok,bgr=cap.read()
                    if not ok: raise RuntimeError(f'Cannot decode {video}:{frame_id}')
                    frames.append(cv2.cvtColor(bgr,cv2.COLOR_BGR2RGB))
                    used.append({'video':str(video.resolve()),'frame_idx':int(frame_id)})
            finally: cap.release()
        if not frames: raise ValueError(f'No training images for {camera}')
        backgrounds[camera]=np.median(np.stack(frames),axis=0).astype(np.uint8)
        Image.fromarray(backgrounds[camera]).save(output_dir/f'{camera}.png')
        provenance[camera]=used
        print(f'Background {camera}: {len(frames)} training frames',flush=True)
    save_json(output_dir/'provenance.json',provenance)
    return backgrounds


def find_car_crop(image, background, config):
    """Largest moving foreground component in a known fixed camera. NO label inputs.

    Static-camera synthetic baseline, not a general car detector. Scene changes,
    multiple cars, camera motion, or foreground occluders can defeat this method.
    """
    if image.shape!=background.shape: raise ValueError('Camera/background resolution mismatch')
    difference=np.abs(image.astype(np.int16)-background.astype(np.int16)).max(axis=2)
    mask=(difference>config.difference_threshold).astype(np.uint8)
    mask=cv2.morphologyEx(mask,cv2.MORPH_CLOSE,np.ones((5,5),np.uint8))
    count,labels,stats,_=cv2.connectedComponentsWithStats(mask,connectivity=8)
    if count<2: return None
    idx=1+int(stats[1:,cv2.CC_STAT_AREA].argmax())
    x,y,w,h,area=map(int,stats[idx])
    if area<config.min_component_area: return None
    # Camera motion or lighting changes are not a valid car proposal.
    if w*h>image.shape[0]*image.shape[1]*.5: return None
    side=max(64,int(math.ceil(max(w,h)*config.crop_padding+16)))
    return (int(round(x+w/2-side/2)),int(round(y+h/2-side/2)),side)


def crop_image(image, crop, size):
    x,y,side=map(int,crop)
    if side<1: raise ValueError('Invalid crop size')
    canvas=np.zeros((side,side,3),np.uint8)
    height,width=image.shape[:2]
    x0,y0=max(x,0),max(y,0);x1,y1=min(x+side,width),min(y+side,height)
    if x1>x0 and y1>y0: canvas[y0-y:y1-y,x0-x:x1-x]=image[y0:y1,x0:x1]
    return cv2.resize(canvas,(size,size),interpolation=cv2.INTER_LINEAR)


def to_crop(points,crop,size):
    return (np.asarray(points)-np.asarray(crop[:2])+.5)*size/crop[2]-.5


def from_crop(points,crop,size):
    return (np.asarray(points)+.5)*crop[2]/size-.5+np.asarray(crop[:2])


def to_world(points,camera):
    xy=np.asarray(points,dtype=float)
    hom=np.column_stack([xy,np.ones(len(xy))])@np.linalg.inv(np.asarray(camera['ground_plane_homography'])).T
    if not np.isfinite(hom).all() or np.any(np.abs(hom[:,2])<1e-9): raise ValueError('Projection near horizon')
    return hom[:,:2]/hom[:,2,None]


def signed_excess(world):
    world=np.asarray(world);x,y=world[...,0],world[...,1]
    entry=np.hypot(x-np.clip(x,-30,0),y)
    exit_distance=np.hypot(x-40,y-np.clip(y,40,70))
    angle=np.clip(np.arctan2(x,40-y),0,np.pi/2)
    arc=np.hypot(x-40*np.sin(angle),y-(40-40*np.cos(angle)))
    return np.minimum(np.minimum(entry,exit_distance),arc)-7.0


def boundary_interval(world,count=161):
    pairs=np.asarray(world).reshape(4,2,2)
    fractions=np.linspace(0,1,count)
    samples=pairs[:,0,None]+fractions[None,:,None]*(pairs[:,1]-pairs[:,0])[:,None]
    upper=signed_excess(samples).min(1)
    error=np.linalg.norm(pairs[:,1]-pairs[:,0],axis=1)/(2*(count-1))
    return np.array([(upper-error).min(),upper.min()])


def verdict(interval):
    return True if interval[0]>0 else (False if interval[1]<=0 else None)


def prepare_records(data_dir, split_scenarios, backgrounds, config, cache_dir):
    """Crop proposals use pixels only. Truth is used solely to check target eligibility."""
    cache_dir=Path(cache_dir);cache_dir.mkdir(parents=True,exist_ok=True)
    from collections import Counter
    records=[];audit={}
    for split,scenarios in split_scenarios.items():
        counts=Counter()
        for scenario in scenarios:
            for labels_path in sorted((Path(data_dir)/scenario).glob('*.json')):
                doc=json.loads(labels_path.read_text())
                if doc.get('label_contract',{}).get('schema_version')!=SCHEMA: raise ValueError(f'Wrong schema: {labels_path}')
                camera=doc.get('camera')
                if camera is None: raise ValueError('Moving-camera showcase is excluded')
                rows=doc['frames'];assert len({r['car_id'] for r in rows})==1,'Multi-car association not implemented'
                by_frame={r['frame_idx']:r for r in rows};assert len(by_frame)==len(rows)
                video=labels_path.with_suffix('.mp4');cap=cv2.VideoCapture(str(video))
                if not cap.isOpened(): raise RuntimeError(f'Cannot read {video}')
                assert int(cap.get(cv2.CAP_PROP_FRAME_COUNT))==len(rows)
                try:
                    for frame_idx in range(len(rows)):
                        ok,bgr=cap.read()
                        if not ok: raise RuntimeError(f'Cannot decode {video}:{frame_idx}')
                        if frame_idx%config.frame_stride:continue
                        counts['sampled']+=1
                        image=cv2.cvtColor(bgr,cv2.COLOR_BGR2RGB)
                        row=by_frame[frame_idx]
                        xyz=np.array([p for wheel in WHEELS for p in row['tyre_boundaries'][wheel]['ground_tread_endpoints_world_xyz']])
                        cam_xyz=(xyz-np.asarray(camera['position_m']))@np.asarray(camera['R_world_to_camera']).T
                        projected=cam_xyz@np.asarray(camera['K']).T
                        uv=projected[:,:2]/projected[:,2,None]
                        if not (np.isfinite(uv).all() and np.all(cam_xyz[:,2]>.2)
                                and np.all(uv>=0) and np.all(uv<np.asarray(camera['resolution']))):
                            counts['out_of_view']+=1;continue
                        np.testing.assert_allclose(to_world(uv,camera),xyz[:,:2],atol=1e-7)
                        interval=boundary_interval(xyz[:,:2],config.boundary_samples)
                        np.testing.assert_allclose(interval,row['boundary_margin_m_interval'],atol=1e-8)
                        assert verdict(interval) is row['is_violation']
                        crop=find_car_crop(image,backgrounds[camera['name']],config)
                        if crop is None:counts['no_crop']+=1;continue
                        points=to_crop(uv,crop,config.image_size)
                        if np.any(points<2) or np.any(points>=config.image_size-2):
                            counts['crop_missed_endpoint']+=1;continue
                        # Preserve source pixels; any augmentation is applied only to training samples.
                        path=cache_dir/f'{scenario}_{camera["name"]}_{frame_idx:04d}.png'
                        Image.fromarray(image).save(path)
                        records.append({'image_path':str(path.resolve()),'video':str(video.resolve()),
                            'split':split,'scenario':scenario,'frame_idx':frame_idx,'camera':camera,
                            'crop':list(crop),'points_original_px':uv.tolist(),'world':xyz[:,:2].tolist(),
                            'is_violation':row['is_violation'],'true_margin_m_interval':row['boundary_margin_m_interval']})
                        counts['kept']+=1
                        counts['review' if row['is_violation'] is None else ('positive' if row['is_violation'] else 'negative')]+=1
                finally:cap.release()
        audit[split]=dict(counts);print(split,dict(counts),flush=True)
        subset=[r for r in records if r['split']==split]
        assert subset and {r['is_violation'] for r in subset if r['is_violation'] is not None}=={False,True},f'{split} requires positive and negative examples'
    return records,audit


class BoundaryDataset(Dataset):
    def __init__(self,records,config,augment=False):self.records=records;self.config=config;self.augment=augment
    def __len__(self):return len(self.records)
    def __getitem__(self,index):
        record=self.records[index];image=read_rgb(record['image_path']);crop=record['crop']
        if self.augment:
            x,y,side=crop;new_side=max(16,int(round(side*random.uniform(.90,1.15))))
            candidate=[int(x+side/2-new_side/2+random.uniform(-.04,.04)*side),
                       int(y+side/2-new_side/2+random.uniform(-.04,.04)*side),new_side]
            points=to_crop(record['points_original_px'],candidate,self.config.image_size)
            if np.all(points>=4) and np.all(points<self.config.image_size-4):crop=candidate
        cropped=crop_image(image,crop,self.config.image_size)
        if self.augment:
            contrast=random.uniform(.85,1.15);brightness=random.uniform(-12,12)
            gains=np.random.uniform(.93,1.07,(1,1,3))
            cropped=np.clip((cropped.astype(float)-127.5)*contrast+127.5+brightness,0,255)
            cropped=np.clip(cropped*gains,0,255).astype(np.uint8)
        points=to_crop(record['points_original_px'],crop,self.config.image_size)
        return cropped,points.astype(np.float32)


def collate_batch(samples,processor):
    images,points=zip(*samples)
    batch=processor(images=list(images),return_tensors='pt')
    batch['points']=torch.from_numpy(np.stack(points))
    return batch


class BoundaryHeatmapModel(nn.Module):
    """Swin + Mask2Former pixel decoder, followed by a dedicated eight-channel head."""
    def __init__(self,pixel_level,config):
        super().__init__();self.pixel_level=pixel_level;self.config=config
        channels=pixel_level.decoder.mask_projection.out_channels
        self.head=nn.Sequential(nn.Conv2d(channels,128,3,padding=1),nn.GroupNorm(8,128),nn.GELU(),
                                nn.Upsample(scale_factor=2,mode='bilinear',align_corners=False),
                                nn.Conv2d(128,64,3,padding=1),nn.GroupNorm(8,64),nn.GELU(),nn.Conv2d(64,8,1))
        for p in self.pixel_level.encoder.parameters():p.requires_grad=False
        if config.unfreeze_last_stage:
            for p in self.pixel_level.encoder.encoder.layers[-1].parameters():p.requires_grad=True
            # Output feature norms are small and adapt with the last backbone stage.
            for name,p in self.pixel_level.encoder.named_parameters():
                if 'hidden_states_norms' in name:p.requires_grad=True

    @classmethod
    def from_pretrained(cls,path,config):
        base=Mask2FormerForUniversalSegmentation.from_pretrained(path,local_files_only=True)
        model=cls(base.model.pixel_level_module,config)
        model.feature_config=base.config
        return model

    def train(self,mode=True):
        super().train(mode)
        self.pixel_level.encoder.eval()  # keep frozen stages deterministic
        if mode and self.config.unfreeze_last_stage:self.pixel_level.encoder.encoder.layers[-1].train()
        return self

    def forward(self,pixel_values):
        features=self.pixel_level(pixel_values).decoder_last_hidden_state
        return F.interpolate(self.head(features),size=(self.config.heatmap_size,self.config.heatmap_size),
                             mode='bilinear',align_corners=False)

    def optimizer_groups(self):
        return [{'params':self.head.parameters(),'lr':self.config.head_lr},
                {'params':self.pixel_level.decoder.parameters(),'lr':self.config.decoder_lr},
                {'params':[p for p in self.pixel_level.encoder.parameters() if p.requires_grad],
                 'lr':self.config.backbone_lr}]


def gaussian_targets(points,config):
    size=config.heatmap_size
    axis=torch.arange(size,device=points.device,dtype=torch.float32)
    yy,xx=torch.meshgrid(axis,axis,indexing='ij')
    # align_corners=False uses pixel-centre coordinates for scaling.
    centers=(points.float()+.5)*(size/config.image_size)-.5
    squared=(xx[None,None]-centers[:,:,0,None,None])**2+(yy[None,None]-centers[:,:,1,None,None])**2
    heatmaps=torch.exp(-squared/(2*config.sigma_px**2))
    return heatmaps/heatmaps.sum((-2,-1),keepdim=True).clamp_min(1e-12)


def heatmap_loss(logits,points,config):
    logits=logits.float();target=gaussian_targets(points,config)
    log_prob=logits.flatten(2).log_softmax(-1).reshape_as(logits)
    spatial_ce=-(target*log_prob).sum((-2,-1)).mean()
    # A small coordinate term provides a useful direction while early maps are diffuse.
    prob=log_prob.exp();axis=torch.arange(config.heatmap_size,device=logits.device,dtype=torch.float32)
    x=(prob.sum(-2)*axis).sum(-1);y=(prob.sum(-1)*axis).sum(-1)
    expected=torch.stack([x,y],-1)/config.heatmap_size
    desired=((points.float()+.5)*(config.heatmap_size/config.image_size)-.5)/config.heatmap_size
    return spatial_ce+10*F.smooth_l1_loss(expected,desired)


def decode_heatmaps(logits,config):
    """Always expose raw locations. Concentration is used for decisions, not hiding plots."""
    probs=logits.float().flatten(2).softmax(-1).reshape_as(logits)
    batch,channels,h,w=probs.shape
    flat=probs.flatten(2).argmax(-1);cx=flat%w;cy=flat//w
    axis_y=torch.arange(h,device=logits.device)[None,None,:,None]
    axis_x=torch.arange(w,device=logits.device)[None,None,None,:]
    local=(abs(axis_x-cx[:,:,None,None])<=config.local_peak_radius)&(abs(axis_y-cy[:,:,None,None])<=config.local_peak_radius)
    weights=probs*local;mass=weights.sum((-2,-1))
    x=(weights*axis_x).sum((-2,-1))/mass.clamp_min(1e-12)
    y=(weights*axis_y).sum((-2,-1))/mass.clamp_min(1e-12)
    points=(torch.stack([x,y],-1)+.5)*(config.image_size/config.heatmap_size)-.5
    return points,mass


def decide(points_original,scores,camera,config):
    if not np.isfinite(points_original).all():return None,None,'missing_endpoint'
    if np.any(points_original<0) or np.any(points_original>=np.asarray(camera['resolution'])):
        return None,None,'endpoint_outside_image'
    if np.any(np.asarray(scores)<config.min_peak_mass):return None,None,'diffuse_heatmap'
    try:world=to_world(points_original,camera)
    except ValueError:return None,None,'projection_near_horizon'
    pairs=world.reshape(4,2,2);widths=np.linalg.norm(pairs[:,1]-pairs[:,0],axis=1)
    if np.any((widths<config.min_tread_width_m)|(widths>config.max_tread_width_m)):
        return None,None,'implausible_tread_width'
    interval=boundary_interval(world,config.boundary_samples);result=verdict(interval)
    return result,interval,'sampling_boundary_uncertainty' if result is None else None


@torch.no_grad()
def predict_frame(model,processor,image,camera,backgrounds,config,device):
    if camera['name'] not in backgrounds:return {'reason':'unknown_camera','points_original_px':None,'crop':None}
    crop=find_car_crop(image,backgrounds[camera['name']],config)
    if crop is None:return {'reason':'no_car_crop','points_original_px':None,'crop':None}
    cropped=crop_image(image,crop,config.image_size)
    inputs=processor(images=cropped,return_tensors='pt')['pixel_values'].to(device)
    # Inference uses FP32 to match validation; batch size 1 keeps this affordable.
    points,scores=decode_heatmaps(model(inputs),config)
    points=from_crop(points[0].cpu().numpy(),crop,config.image_size);scores=scores[0].cpu().numpy()
    prediction,interval,reason=decide(points,scores,camera,config)
    return {'points_original_px':points,'scores':scores,'prediction':prediction,'interval':interval,'reason':reason,'crop':crop}


@torch.no_grad()
def localization_metrics(model,loader,device,config):
    model.eval();errors=[];losses=[]
    for batch in loader:
        pixels=batch['pixel_values'].to(device);truth=batch['points'].to(device)
        logits=model(pixels);losses.append(float(heatmap_loss(logits,truth,config)))
        points,_=decode_heatmaps(logits,config)
        errors.extend(torch.linalg.vector_norm(points-truth,dim=-1).cpu().numpy().ravel().tolist())
    return {'loss':float(np.mean(losses)),'endpoint_mean_crop_px':float(np.mean(errors)),
            'endpoint_median_crop_px':float(np.median(errors)),'pck_5_crop_px':float(np.mean(np.asarray(errors)<=5))}


def save_checkpoint(model,processor,config,path,background_dir):
    path=Path(path);path.mkdir(parents=True,exist_ok=True)
    torch.save(model.state_dict(),path/'boundary_heatmaps.pt')
    # Save the feature-extractor configuration to reconstruct offline without COCO weights.
    model.feature_config.save_pretrained(path/'feature_config')
    save_json(path/'training_config.json',asdict(config))
    processor.save_pretrained(path)
    shutil.copytree(background_dir,path/'backgrounds',dirs_exist_ok=True)


def load_checkpoint(path,device='cpu'):
    """Offline reload. Original source checkpoint is not required after saving."""
    from transformers import Mask2FormerConfig
    from transformers.models.mask2former.modeling_mask2former import Mask2FormerPixelLevelModule
    path=Path(path);config=Config(**json.loads((path/'training_config.json').read_text()))
    feature_config=Mask2FormerConfig.from_pretrained(path/'feature_config',local_files_only=True)
    model=BoundaryHeatmapModel(Mask2FormerPixelLevelModule(feature_config),config)
    model.feature_config=feature_config
    model.load_state_dict(torch.load(path/'boundary_heatmaps.pt',map_location='cpu',weights_only=True))
    model.to(device).eval()
    processor=Mask2FormerImageProcessor.from_pretrained(path,local_files_only=True)
    backgrounds={p.stem:read_rgb(p) for p in (path/'backgrounds').glob('*.png')}
    return model,processor,config,backgrounds


def error_stats(values):
    if not values:return {'mean':None,'median':None,'p95':None}
    return {'mean':float(np.mean(values)),'median':float(np.median(values)),
            'p95':float(np.percentile(values,95))}


@torch.no_grad()
def evaluate_records(model,processor,records,backgrounds,config,device,output_dir,name):
    """Report unfiltered localisation and rejected-decision coverage separately."""
    from collections import Counter
    from tqdm import tqdm
    model.eval();pixel_errors=[];world_errors=[];margin_errors=[];concentrations=[]
    decisions=np.zeros((3,3),np.int64);reasons=Counter();results=[]
    endpoint_count=0
    def category(value):return 2 if value is None else int(value)
    for record in tqdm(records,desc=f'Evaluate {name}'):
        result=predict_frame(model,processor,read_rgb(record['image_path']),record['camera'],backgrounds,config,device)
        points=result['points_original_px'];prediction=result.get('prediction');interval=result.get('interval')
        reason=result['reason'];truth=record['is_violation']
        if points is not None:
            valid=np.isfinite(points).all(1);endpoint_count+=int(valid.sum())
            pixel_errors.extend(np.linalg.norm(points[valid]-np.asarray(record['points_original_px'])[valid],axis=1).tolist())
            concentrations.extend(result['scores'].tolist())
            try:
                predicted_world=to_world(points[valid],record['camera'])
                world_errors.extend((100*np.linalg.norm(predicted_world-np.asarray(record['world'])[valid],axis=1)).tolist())
            except ValueError:pass
        if interval is not None:
            margin_errors.append(float(100*abs(np.mean(interval)-np.mean(record['true_margin_m_interval']))))
        if reason:reasons[reason]+=1
        decisions[category(truth),category(prediction)]+=1
        results.append({'video':record['video'],'frame_idx':record['frame_idx'],'truth':truth,'prediction':prediction,
                        'review_reason':reason,'crop':list(result['crop']) if result['crop'] is not None else None,
                        'points_original_px':points.tolist() if points is not None else None,
                        'peak_mass_uncalibrated':result['scores'].tolist() if points is not None else None,
                        'predicted_margin_m_interval':interval.tolist() if interval is not None else None})
    known=int(decisions[:2].sum());decided=int(decisions[:2,:2].sum());tn,fp,abstain_legal=decisions[0];fn,tp,abstain_positive=decisions[1]
    def div(a,b):return float(a/b) if b else None
    metrics={'frames':len(records),'endpoint_output_coverage':div(endpoint_count,len(records)*8),
        'endpoint_error_original_px_all_raw':error_stats(pixel_errors),
        'endpoint_error_simulated_cm_all_raw':error_stats(world_errors),
        'pck_5_original_px':float(np.mean(np.asarray(pixel_errors)<=5)) if pixel_errors else None,
        'peak_mass_uncalibrated':error_stats(concentrations),
        'margin_error_cm_accepted_geometry_only':error_stats(margin_errors),
        'margin_measurement_coverage':div(len(margin_errors),len(records)),
        'decision_coverage_known_truth':div(decided,known),'violation_accuracy_decided_known_only':div(tn+tp,decided),
        'violation_precision_known_truth':div(tp,tp+fp),'violation_recall_abstentions_count_as_misses':div(tp,tp+fn+abstain_positive),
        'correct_decisions_over_known_frames':div(tn+tp,known),'always_legal_accuracy_known_truth':div(tn+fp+abstain_legal,known),
        'ground_truth_review_frames':int(decisions[2].sum()),'review_reasons':dict(reasons),
        'confusion_rows_and_columns_legal_violation_review':decisions.tolist()}
    save_json(Path(output_dir)/f'{name}_metrics.json',metrics);save_json(Path(output_dir)/f'{name}_predictions.json',results)
    print(json.dumps(metrics,indent=2));return metrics,results
