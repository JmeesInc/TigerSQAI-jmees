"""Rigid scope geometry and calibrated nonuniform priors, in patient RAS mm."""
import math
import numpy as np


class RejectPose(ValueError):
    pass


def unit(v):
    v = np.asarray(v, dtype=float)
    n = np.linalg.norm(v)
    if n < 1e-10:
        raise RejectPose("zero direction")
    return v / n


def basis(axis):
    axis = unit(axis)
    ref = np.array([0., 0., 1.])
    if abs(axis @ ref) > 0.95:
        ref = np.array([0., 1., 0.])
    a = unit(ref - (ref @ axis) * axis)
    return a, np.cross(axis, a)


def normal(rng, spec):
    if spec['sd'] <= 0:
        return float(np.clip(spec['mean'], spec['min'], spec['max']))
    for _ in range(1000):
        x = rng.normal(spec['mean'], spec['sd'])
        if spec['min'] <= x <= spec['max']:
            return float(x)
    raise ValueError("Invalid or vanishing truncated-normal prior")


def choice(rng, spec):
    weights = np.asarray(spec['weights'], dtype=float)
    if np.any(weights < 0) or weights.sum() <= 0:
        raise ValueError("Invalid categorical weights")
    return rng.choice(spec['values'], p=weights / weights.sum()).item()


def optical_axis(shaft, oblique_deg, axial_rotation_rad):
    """At fixed shaft, axial rotation traces a cone of half-angle oblique_deg."""
    shaft = unit(shaft)
    a, b = basis(shaft)
    radial = np.cos(axial_rotation_rad) * a + np.sin(axial_rotation_rad) * b
    theta = np.deg2rad(oblique_deg)
    return unit(np.cos(theta) * shaft + np.sin(theta) * radial)


def solve_scope(port, target, working_mm, oblique_deg, azimuth):
    """Solve T=P+insertion*S+working*V, subject to S dot V = cos(theta)."""
    port, target = np.asarray(port), np.asarray(target)
    delta = target - port
    length = np.linalg.norm(delta)
    d = unit(delta)
    theta = np.deg2rad(oblique_deg)
    if length <= working_mm or length <= working_mm * np.sin(theta):
        raise RejectPose("working distance exceeds port-target distance")
    alpha = np.arcsin(working_mm * np.sin(theta) / length)
    a, b = basis(d)
    meridian = np.cos(azimuth) * a + np.sin(azimuth) * b
    shaft = np.cos(alpha) * d - np.sin(alpha) * meridian
    radial = np.sin(alpha) * d + np.cos(alpha) * meridian
    sa, sb = basis(shaft)
    rotation = math.atan2(radial @ sb, radial @ sa)
    view = optical_axis(shaft, oblique_deg, rotation)
    insertion = np.sqrt(length**2 - (working_mm * np.sin(theta))**2) - working_mm * np.cos(theta)
    tip = port + insertion * shaft
    if not np.allclose(tip + working_mm * view, target, atol=1e-6):
        raise RuntimeError("Rigid-scope solution inconsistent")
    return tip, shaft, view, float(insertion), rotation


def sample_ports(rng, ribs, config):
    cfg = config['ports']
    result = []
    count = choice(rng, cfg['count'])
    center=np.asarray(cfg['thorax_center_xy_mm'])
    for _ in range(cfg['samples_per_layout']):
        level = int(choice(rng, cfg['interspaces']))
        angle = np.deg2rad(normal(rng, cfg['azimuth_deg']))
        pair = []
        for k in (level, level + 1):
            vertices = ribs[str(k)]
            vertices = vertices[vertices[:, 0] > center[0]+cfg['minimum_right_offset_mm']]
            if not len(vertices):
                raise ValueError(f"Right rib {k}: invalid RAS transform")
            angles = np.arctan2(vertices[:, 1]-center[1], vertices[:, 0]-center[0])
            near = np.argsort(np.abs(angles - angle))[:min(cfg['rib_surface_neighbors'], len(vertices))]
            pair.append(np.median(vertices[near], axis=0))
        fraction = normal(rng, cfg['interspace_fraction'])
        point = pair[0] * (1 - fraction) + pair[1] * fraction
        if all(np.linalg.norm(point - np.array(p['position_mm'])) >= cfg['minimum_spacing_mm'] for p in result):
            result.append({'position_mm': point.tolist(), 'interspace': level,
                           'azimuth_deg': float(np.rad2deg(angle))})
        if len(result) == count:
            return result
    raise ValueError("Cannot sample separated right intercostal ports; check atlas coordinates")


def sample_camera(rng, ports, landmarks, config):
    target_cfg = config['targets']
    name = choice(rng, {'values': target_cfg['names'], 'weights': target_cfg['weights']})
    jitter = rng.normal(size=3)
    if np.any(np.abs(jitter) > target_cfg['jitter_limit_sd']):
        raise RejectPose("target jitter tail")
    target = np.array(landmarks[name]) + jitter * target_cfg['jitter_sd_mm']
    # Prefer scope ports close to the target's craniocaudal level.
    positions = np.array([p['position_mm'] for p in ports])
    weights = np.exp(-0.5 * ((positions[:, 2] - target[2]) / config['ports']['scope_port_height_sd_mm'])**2)
    lengths = np.linalg.norm(positions-target,axis=1)
    lo,hi=config['ports']['port_to_target_mm']
    weights *= (lengths>=lo)&(lengths<=hi)
    if weights.sum()<=0:raise RejectPose('no reachable scope port')
    port_index = int(rng.choice(len(ports), p=weights / weights.sum()))
    port = positions[port_index]
    length = np.linalg.norm(target - port)
    lo, hi = config['ports']['port_to_target_mm']
    if not lo <= length <= hi:
        raise RejectPose("port-target distance")
    s = config['scope']
    w = s['working_distance_mm']
    distance = float(rng.lognormal(w['log_mean'], w['log_sd']))
    if not w['min'] <= distance <= w['max']:
        raise RejectPose("working distance prior tail")
    theta = float(choice(rng, s['oblique_angle_deg']))
    a = s['shaft_azimuth_rad']
    azimuth = float(rng.vonmises(a['mean'], a['kappa']))
    tip, shaft, view, insertion, rotation = solve_scope(port, target, distance, theta, azimuth)
    if not s['insertion_mm'][0] <= insertion <= s['insertion_mm'][1]:
        raise RejectPose("insertion")
    if shaft[0] >= -s['minimum_medial_component'] or view[0] >= -s['minimum_medial_component']:
        raise RejectPose("scope points away from mediastinum")
    # Scope-head sensor orientation: optical meridian + independent head roll.
    up0, up1 = basis(shaft)
    up = np.cos(rotation) * up0 + np.sin(rotation) * up1
    up = unit(up - (up @ view) * view)
    right = unit(np.cross(view, up))
    roll = normal(rng, s['roll_deg'])
    r = np.deg2rad(roll)
    right, up = np.cos(r)*right + np.sin(r)*up, -np.sin(r)*right + np.cos(r)*up
    c2w = np.eye(4)
    c2w[:3, :3] = np.column_stack([right, up, -view])  # Blender looks along -Z
    c2w[:3, 3] = tip
    cv_to_world = c2w @ np.diag([1., -1., -1., 1.])
    width, height = config['resolution']
    fov = normal(rng, s['horizontal_fov_deg'])
    focal = width / (2 * np.tan(np.deg2rad(fov)/2))
    intr=config['intrinsics']
    if intr['focal_x_px'] is not None:
        focal=normal(rng,intr['focal_x_px'])
        fov=float(np.rad2deg(2*np.arctan(width/(2*focal))))
    fy=focal*normal(rng,intr['fy_over_fx'])
    cx=(width-1)/2+normal(rng,intr['principal_x_offset_fraction'])*width
    cy=(height-1)/2+normal(rng,intr['principal_y_offset_fraction'])*height
    k = [[focal, 0., cx], [0., fy, cy], [0., 0., 1.]]
    return {'target_name': name, 'target_mm': target.tolist(), 'scope_port_index': port_index,
            'port_mm': port.tolist(), 'tip_mm': tip.tolist(), 'shaft_axis_ras': shaft.tolist(),
            'optical_axis_ras': view.tolist(), 'oblique_angle_deg': theta,
            'shaft_rotation_rad': rotation, 'sensor_roll_deg': roll,
            'insertion_mm': insertion, 'working_distance_mm': distance,
            'port_to_target_mm': float(length), 'horizontal_fov_deg': fov,
            'K': k, 'camera_to_world_blender_mm': c2w.tolist(),
            'camera_to_world_cv_mm': cv_to_world.tolist(),
            'world_to_camera_cv_mm': np.linalg.inv(cv_to_world).tolist(),
            'pixel_convention': 'integer pixel centers; origin top-left; CV +Z forward; RAS mm'}


def distortion_grid(rng, camera, config):
    """Inverse Brown radial warp; outputs nearest source indices into overscan render."""
    w, h = config['resolution']
    cfg = config['distortion']
    k1 = normal(rng, cfg['k1']) if cfg['enabled'] else 0.
    k2 = normal(rng, cfg['k2']) if cfg['enabled'] else 0.
    scale = cfg['overscan'] if cfg['enabled'] else 1.
    sw, sh = int(np.ceil(w*scale)), int(np.ceil(h*scale))
    focal,fy = camera['K'][0][0],camera['K'][1][1]
    cx,cy = camera['K'][0][2],camera['K'][1][2]
    scx,scy=cx+(sw-w)/2,cy+(sh-h)/2
    yy, xx = np.mgrid[:h, :w]
    xd, yd = (xx-cx)/focal, (yy-cy)/fy
    rd = np.sqrt(xd*xd+yd*yd)
    ru = rd.copy()
    # Reject priors whose radial mapping folds inside the source image.
    max_r = np.hypot(max(abs(scx),abs(sw-1-scx))/focal,max(abs(scy),abs(sh-1-scy))/fy)
    r = np.linspace(0, max_r, 256)
    if np.min(1+3*k1*r*r+5*k2*r**4) <= 0:
        raise RejectPose("noninvertible radial distortion prior")
    for _ in range(cfg['inverse_iterations']):
        ru -= (ru*(1+k1*ru**2+k2*ru**4)-rd)/(1+3*k1*ru**2+5*k2*ru**4)
    if np.max(np.abs(ru*(1+k1*ru**2+k2*ru**4)-rd)) > 1e-7:
        raise RejectPose("distortion inverse did not converge")
    ratio = np.divide(ru, rd, out=np.ones_like(rd), where=rd>0)
    sx = np.rint(xd*ratio*focal+scx).astype(int)
    sy = np.rint(yd*ratio*fy+scy).astype(int)
    valid = (sx>=0)&(sx<sw)&(sy>=0)&(sy<sh)
    camera['distortion'] = {'model':'Brown radial forward normalized coordinates',
                            'k1':k1,'k2':k2,'p1':0.,'p2':0.,'interpolation':'nearest',
                            'render_size':[sw,sh], 'render_K':[[focal,0,scx],[0,fy,scy],[0,0,1]]}
    return sx.clip(0,sw-1), sy.clip(0,sh-1), valid
