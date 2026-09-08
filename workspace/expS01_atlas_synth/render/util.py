import hashlib
import json
from pathlib import Path
import numpy as np


def sha256(path):
    digest=hashlib.sha256()
    with open(path,'rb') as stream:
        for chunk in iter(lambda:stream.read(1024*1024),b''):digest.update(chunk)
    return digest.hexdigest()


def save_meshes(path,objects):
    arrays={}
    records=[]
    for i,o in enumerate(objects):
        key=f'o{i}'
        arrays[key+'_v']=np.asarray(o['v'],dtype=np.float32)
        arrays[key+'_f']=np.asarray(o['f'],dtype=np.int32)
        records.append({k:v for k,v in o.items() if k not in ['v','f'] }|{'key':key})
    arrays['records']=np.array(json.dumps(records))
    np.savez(path,**arrays)


def load_meshes(path):
    with np.load(path,allow_pickle=False) as archive:
        records=json.loads(str(archive['records']))
        return [r|{'v':archive[r['key']+'_v'].copy(),'f':archive[r['key']+'_f'].copy()} for r in records]


def atomic_json(path,value):
    path=Path(path)
    tmp=path.with_suffix('.tmp')
    tmp.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
    tmp.replace(path)
