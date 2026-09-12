"""World-anchored procedural materials and approximate soft contact shadows.

Only RGB changes: geometry, depth, camera calibration and labels are untouched.
This is an inexpensive preview renderer, not physically based lighting.
"""
import numpy as np


def _noise(x, y, scale):
    x, y = x*scale, y*scale
    ix, iy = np.floor(x), np.floor(y)
    fx, fy = x-ix, y-iy
    fx, fy = fx*fx*(3-2*fx), fy*fy*(3-2*fy)
    def sample(a,b):
        value = np.sin(a*127.1+b*311.7)*43758.5453
        return (value-np.floor(value))*2-1
    return ((1-fx)*sample(ix,iy)+fx*sample(ix+1,iy))*(1-fy) + ((1-fx)*sample(ix,iy+1)+fx*sample(ix+1,iy+1))*fy


def surface(base, cam):
    rgb, depth = base
    height, width = depth.shape
    yy, xx = np.mgrid[:height, :width]
    rays = np.stack([xx + .5, yy + .5, np.ones_like(xx)], -1) @ np.linalg.inv(cam['K']).T
    valid = np.isfinite(depth)
    world = (rays * np.where(valid, depth, 0)[..., None]) @ np.asarray(cam['R_world_to_camera']) + cam['position_m']
    x, y = world[..., 0], world[..., 1]
    # Continuous world-space detail stays fixed when the camera moves.
    footprint = np.maximum(np.linalg.norm(np.gradient(world,axis=0),axis=-1), np.linalg.norm(np.gradient(world,axis=1),axis=-1))
    grain = _noise(x,y,24)*np.clip(1-footprint*24,0,1)
    broad = _noise(x,y,1.8)
    grass = (rgb[..., 1] > rgb[..., 0]*1.1) & valid
    asphalt = (rgb[..., 2] > rgb[..., 0]) & valid
    amount = np.where(grass, 13., np.where(asphalt, 5., 3.))
    out = rgb.astype(float) + (grain + broad*.65)[..., None]*amount[..., None]
    # Subtle rubber-darkened racing band, avoiding the painted boundary.
    import geometry as g
    distance = g.distance(world[..., :2])
    rubber = np.exp(-((distance-4.8)/.65)**2)*asphalt
    out *= (1-.13*rubber)[..., None]
    sky = np.stack([125+yy/height*45, 166+yy/height*37, 202+yy/height*22], -1)
    out = np.where(valid[..., None], out, sky)
    return (np.clip(out, 0, 255).astype(np.uint8), depth), world, valid


def shadows(base, world, valid, rows):
    strength = np.zeros(valid.shape, np.float32)
    for row in rows:
        delta = world[..., :2] - row['world_position']
        c, s = np.cos(row['heading_rad']), np.sin(row['heading_rad'])
        x = delta[..., 0]*c + delta[..., 1]*s
        y = -delta[..., 0]*s + delta[..., 1]*c
        strength = np.maximum(strength, .38*np.exp(-((x/2.35)**4+(y/.95)**4)*1.5))
    rgb = base[0].astype(float)*(1-strength*valid)[..., None]
    return np.clip(rgb, 0, 255).astype(np.uint8), base[1]
