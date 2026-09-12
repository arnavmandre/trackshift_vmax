"""Conservative decision-level fusion for synchronized views of the same car.
Association and synchronization must be supplied by the caller. No truth labels
or heatmap mass ranking are used to choose the verdict.
"""
from collections import Counter, defaultdict

CAMERAS = ('broadcast', 'exit', 'trackside')


def fuse_views(views, expected_cameras=CAMERAS):
    if not views:raise ValueError('No views')
    identities={(v['scenario'],v['car_id'],v['frame_idx']) for v in views}
    if len(identities)!=1:raise ValueError('Views must refer to the same scenario, car and frame')
    names=[v['camera_name'] for v in views]
    if len(names)!=len(set(names)):raise ValueError('Duplicate camera view')
    if set(names)!=set(expected_cameras):raise ValueError('Missing or unexpected camera view')
    times=[v['time_s'] for v in views]
    if max(times)-min(times)>1e-6:raise ValueError('Camera timestamps are not synchronized')
    accepted=[]
    for v in views:
        if v['prediction'] is not None and type(v['prediction']) is not bool:
            raise ValueError('Verdicts must be bool or None')
        if v['prediction'] is not None and not v.get('review_reason'):
            accepted.append(v)
    votes={v['prediction'] for v in accepted}
    prediction=next(iter(votes)) if len(votes)==1 else None
    reason=None if prediction is not None else ('camera_disagreement' if votes else 'no_reliable_view')
    scenario,car,frame=next(iter(identities))
    return dict(scenario=scenario,car_id=car,frame_idx=frame,time_s=times[0],
        prediction=prediction,review_reason=reason,accepted_view_count=len(accepted),
        supporting_cameras=[v['camera_name'] for v in accepted] if prediction is not None else [],
        views=[dict(camera_name=v['camera_name'],prediction=v['prediction'],
                    review_reason=v.get('review_reason'),
                    predicted_margin_m_interval=v.get('predicted_margin_m_interval')) for v in views])


def fuse_records(records, expected_cameras=CAMERAS):
    groups=defaultdict(list)
    for v in records:groups[(v['scenario'],v['car_id'],v['frame_idx'])].append(v)
    fused=[]
    for key,views in sorted(groups.items()):
        result=fuse_views(views,expected_cameras)
        # Ground truth is attached only AFTER computing the verdict, for scoring.
        truths={v['truth'] for v in views}
        if len(truths)!=1:raise ValueError(f'Inconsistent truth across cameras: {key}')
        result['truth']=next(iter(truths));fused.append(result)
    return fused


def predict_multiview(model,processor,frames,backgrounds,config,device,expected_cameras=CAMERAS):
    """frames: one {image,camera,scenario,car_id,frame_idx,time_s} per camera.
    Returns one verdict; real footage needs upstream car matching/synchronization.
    """
    import boundary_training as bt
    views=[]
    for frame in frames:
        result=bt.predict_frame(model,processor,frame['image'],frame['camera'],backgrounds,config,device)
        interval=result.get('interval')
        views.append(dict(scenario=frame['scenario'],car_id=frame['car_id'],
            frame_idx=frame['frame_idx'],time_s=frame['time_s'],camera_name=frame['camera']['name'],
            prediction=result.get('prediction'),review_reason=result.get('reason'),
            predicted_margin_m_interval=interval.tolist() if interval is not None else None))
    return fuse_views(views,expected_cameras)
