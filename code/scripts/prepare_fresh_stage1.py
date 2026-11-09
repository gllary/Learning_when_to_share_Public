"""Prepare a new label-free workspace for current frozen Stage1 forwards.

This only creates a new workspace. Expert jobs are explicit separate commands
because MatterVial, JMP and Glass require different pinned environments.
"""
import argparse
import hashlib
import json
from pathlib import Path
import shutil

ROOT=Path(__file__).resolve().parents[2]
DATASETS={'Core-23':'core23','DS2-248':'ds2_248','HEA-95':'hea95','B2-18':'b2_18'}
def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
    return h.hexdigest()
def save(p,x):
    p.parent.mkdir(parents=True,exist_ok=True)
    with p.open('x') as f:json.dump(x,f,indent=2)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',required=True);a=p.parse_args()
    work=Path(a.output).resolve();assert work.is_relative_to(ROOT/'runs');work.mkdir(parents=True,exist_ok=False)
    jobs=json.loads((ROOT/'checkpoint/matbench/stage1/manifest.json').read_text())['jobs'];protected={}
    for j in jobs:
        for k in ('checkpoint','preprocessor','preprocessing'):
            if k in j:
                path=ROOT/j[k];j[k]=str(path);protected[str(path)]=sha(path)
        assert protected[j['checkpoint']]==j['checkpoint_sha256']
    runtime=work/'runtime/external';runtime.mkdir(parents=True)
    (runtime/'JMP').symlink_to(ROOT/'code/vendor/jmp',target_is_directory=True)
    protocol={'datasets':{},'materials':{}}
    for dataset,slug in DATASETS.items():
        base=ROOT/'results/external'/slug;cfg=json.loads((base/'config.json').read_text())
        protocol['datasets'][dataset]={'n':cfg['n']};protocol['materials'][dataset]={'sample_ids':cfg['sample_ids']}
        folder=work/'inputs'/dataset;folder.mkdir(parents=True)
        shutil.copy2(base/'data/materials.csv',folder/'materials.csv')
        features=work/'data_features';features.mkdir(exist_ok=True)
        (features/dataset).symlink_to(base/'data/features',target_is_directory=True)
    save(work/'protocol.json',protocol)
    save(work/'stage1_sources.json',{'jobs':jobs,'protected_files':protected,'project':str(ROOT),'jmp_runtime':str(work/'runtime')})
    commands=[]
    script=ROOT/'code/hpsafemoe/external/forward_stage1.py'
    for outer in range(5):
        for kind in ('mattervial','glass'):
            commands.append({'environment':'hpsafe-mattervial' if kind=='mattervial' else 'hpsafemoe','argv':['python','-B',str(script),kind,str(outer)],'env':{'HPSAFEMOE_EXTERNAL_WORKSPACE':str(work)}})
        for task in ('jdft2d','perovskites','phonons','mp_gap','mp_e_form'):
            commands.append({'environment':'the pinned JMP runtime','argv':['python','-B',str(script),'jmp',str(outer),'--task',task],'env':{'HPSAFEMOE_EXTERNAL_WORKSPACE':str(work)}})
    save(work/'job_commands.json',commands)
    print(json.dumps({'workspace':str(work),'jobs':len(commands),'launched':False,'external_labels_copied':False}))
