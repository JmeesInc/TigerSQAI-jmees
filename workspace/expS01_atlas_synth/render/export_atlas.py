"""Blender entry point: evaluate ONLY an explicit atlas object manifest."""
import json
import sys
from pathlib import Path
import bpy
import numpy as np
from mathutils import Vector


def main():
    config_path, output = sys.argv[sys.argv.index('--')+1:]
    cfg = json.loads(Path(config_path).read_text())
    transform = np.asarray(cfg['atlas_to_ras_mm'], dtype=float)
    depsgraph = bpy.context.evaluated_depsgraph_get()
    arrays, records, ribs = {}, [], {}
    names = [(int(k), n, 'anatomy') for k, values in cfg['objects'].items() for n in values]
    proxy=cfg.get('provisional',{})
    if proxy.get('enabled',False) and proxy['pericardium']['enabled']:
        names += [(0,n,'proxy_source') for n in proxy['pericardium']['source_objects']]
    words = {3:'Third',4:'Fourth',5:'Fifth',6:'Sixth',7:'Seventh',8:'Eighth',9:'Ninth',10:'Tenth'}
    names += [(i, f'{word} rib.r', 'rib') for i, word in words.items()]
    for i, (fine_id, name, role) in enumerate(names):
        obj = bpy.data.objects.get(name)
        if obj is None:
            raise ValueError(f'Missing manifest object: {name}')
        evaluated = obj.evaluated_get(depsgraph)
        mesh = evaluated.to_mesh()
        if mesh is None or not mesh.polygons:
            raise ValueError(f'No evaluated faces: {name}')
        mesh.calc_loop_triangles()
        vertices = np.array([evaluated.matrix_world @ v.co for v in mesh.vertices], dtype=np.float64)
        vertices = vertices @ transform[:3,:3].T + transform[:3,3]
        faces = np.array([t.vertices[:] for t in mesh.loop_triangles], dtype=np.int32)
        evaluated.to_mesh_clear()
        if not np.isfinite(vertices).all():
            raise ValueError(f'Nonfinite geometry: {name}')
        if role == 'rib':
            ribs[str(fine_id)] = vertices.tolist()
            # Existing rib surfaces also constrain the port/shaft path. Label 0:
            # contextual anatomy outside the challenge foreground taxonomy.
            fine_id=int(cfg['context_class_id'])
        key = f'o{i}'
        arrays[key+'_v'], arrays[key+'_f'] = vertices.astype(np.float32), faces
        records.append({'key':key,'name':name,'fine_id':fine_id,'role':role})
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output, **arrays)
    output.with_suffix('.json').write_text(json.dumps({'objects':records,'ribs':ribs,
        'landmarks_mm':cfg['landmarks_mm'],'blender_version':bpy.app.version_string}, indent=2))
    print('EXPORTED',len(records),'atlas objects including right-rib context')


if __name__ == '__main__':
    main()
