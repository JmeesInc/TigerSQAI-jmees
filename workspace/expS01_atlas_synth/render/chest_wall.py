"""Provisional parietal pleura fitted to the existing atlas rib cage, not pixel fill."""
import numpy as np
import trimesh
from scipy.spatial import ConvexHull


def build_chest_wall(objects,ports,cfg):
    if not cfg.get('enabled',False):return None
    ribs=[o['v'] for o in objects if o.get('role') in ['rib','chest_support']]
    if not ribs:raise ValueError('Parietal pleura needs exported atlas ribs')
    points=np.concatenate(ribs);hull=ConvexHull(points)
    mesh=trimesh.Trimesh(points,hull.simplices,process=True).convex_hull
    # Inset the fitted rib envelope; convex closure bridges intercostal gaps.
    center=mesh.vertices.mean(0);radial=mesh.vertices-center
    mesh.vertices-=cfg['inset_mm']*radial/np.maximum(np.linalg.norm(radial,axis=1,keepdims=True),1e-9)
    v,f=trimesh.remesh.subdivide_to_size(mesh.vertices,mesh.faces,max_edge=cfg['mesh_edge_mm'],max_iter=8)
    keep=np.ones(len(f),bool)
    # Real port openings are local cylindrical tunnels, never a global collision exemption.
    for port in ports:
        p=np.asarray(port['position_mm']);axis=center-p;axis/=np.linalg.norm(axis)
        d=v-p;along=d@axis;perp=np.linalg.norm(d-along[:,None]*axis,axis=1)
        hole=(perp<cfg['port_radius_mm'])&(np.abs(along)<cfg['port_tunnel_length_mm'])
        keep &= ~hole[f].any(1)
    return {'name':'parietal_pleura_rib_envelope','role':'parietal_pleura','fine_id':10,'v':v,'f':f[keep]}
