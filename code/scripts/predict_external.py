"""Run label-free Stage2 inference with frozen Stage1 exports."""
import argparse
import json
from pathlib import Path
import sys
import numpy as np

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'code/hpsafemoe/external'))
from inference import load_checkpoint,predict_member,combine_task_members,combine_dataset
from evaluation import sha256,write_json_new

def run(slug,output,device,stage1_root=None):
    output=Path(output).resolve()
    assert output.is_relative_to(ROOT/'runs'), 'Write a new directory under runs/'
    output.mkdir(parents=True,exist_ok=False)
    base=ROOT/'results/external'/slug
    cfg=json.loads((base/'config.json').read_text());refs=json.loads((base/'model_refs.json').read_text())
    graph=json.loads((base/'input_contract.json').read_text())['graph']
    exports_root=(Path(stage1_root).resolve()/cfg['dataset']) if stage1_root else base/'predictions/stage1'
    manifest=json.loads((ROOT/refs['proposal']).read_text())
    models=[r for r in manifest['models'] if '/proposal_full/' in r['path']]
    assert len(models)==50
    outers={};checks=[]
    for outer in range(5):
        outers[outer]={}
        paths=[r for r in models if f'/outer_{outer}/' in r['path']]
        task_records={t:[] for t in cfg['target_tasks']}
        for record in paths:
            p=Path(record['path']);inner=int(p.parent.name.split('_')[-1]);seed=int(p.stem.split('_')[-1])
            r={**record,'path':str(ROOT/p)}
            model,config,norm=load_checkpoint(r,outer,inner,seed,device)
            exports={}
            for expert in config['expert_ids']:
                folder=exports_root/f'outer_{outer}'/expert
                if not folder.exists():exports[expert]=None;continue
                ids=np.load(folder/'sample_ids.npy',allow_pickle=False).astype(str).tolist()
                assert ids==cfg['sample_ids']
                exports[expert]={k:np.load(folder/(k+'.npy'),allow_pickle=False) for k in ['hidden','prediction','uncertainty','available']}
            for task in cfg['target_tasks']:
                task_exports={name:(exports[name] if graph[task][str(outer)][name]['present'] else None) for name in config['expert_ids']}
                for name,slot in graph[task][str(outer)].items():
                    if not slot['present']:continue
                    folder=exports_root/f'outer_{outer}'/name
                    receipt=json.loads((folder/'receipt.json').read_text())
                    assert receipt['checkpoint_sha256']==slot['checkpoint_sha256']
                    assert not receipt['labels_accessed'] and not receipt['weights_updated']
                    for file,digest in receipt['files'].items():assert sha256(folder/file)==digest
                values=predict_member(model,config,norm,task,task_exports,cfg['n'],device)
                task_records[task].append(values)
                dest=output/'members'/f'outer_{outer}'/task;dest.mkdir(parents=True,exist_ok=True)
                np.savez_compressed(dest/f'inner_{inner}_seed_{seed}.npz',**values)
            del model
        for task,records in task_records.items():
            cert=refs['frozen_beta'][str(outer)][task];path=ROOT/cert['file']
            assert sha256(path)==cert['sha256']
            beta=json.loads(path.read_text())['beta']
            outers[outer][task]=combine_task_members(records,beta)
            checks.append({'outer':outer,'task':task,'beta':beta,'models':len(records)})
    values=combine_dataset(cfg['dataset'],outers,cfg['sample_ids'])
    np.savez_compressed(output/'predictions.npz',**values)
    write_json_new(output/'predictions_committed.json',{'dataset':cfg['dataset'],'prediction_sha256':sha256(output/'predictions.npz'),'checks':checks,'labels_read':False,'weights_updated':False,'beta_refitted':False,'stage1_input':str(exports_root)})
    print(output)
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('dataset',choices=['core23','ds2_248','hea95','b2_18']);p.add_argument('--output',required=True);p.add_argument('--device',default='cpu');p.add_argument('--stage1-root',help='Optional new workspace/stage1_exports directory');a=p.parse_args()
    run(a.dataset,a.output,a.device,a.stage1_root)
