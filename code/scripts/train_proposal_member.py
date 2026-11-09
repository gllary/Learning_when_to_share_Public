"""Explicit new member training. Never resumes or modifies published experiments."""
import argparse
import importlib
import json
import os
from dataclasses import asdict
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'code'))

def main():
    p=argparse.ArgumentParser();p.add_argument('--dataset',choices=['jarvis','matbench'],required=True)
    p.add_argument('--architecture',choices=['model'],default='model');p.add_argument('--no-cross',action='store_true')
    p.add_argument('--outer',type=int,default=0,choices=range(5));p.add_argument('--inner',type=int,required=True,choices=range(5))
    p.add_argument('--seed',type=int,required=True,choices=[20261071,20261072]);p.add_argument('--output',required=True)
    args=p.parse_args()
    if args.dataset=='jarvis' and args.no_cross:p.error('JARVIS exposes only the Full main model')
    output=Path(args.output).resolve()
    assert output.is_relative_to(ROOT/'runs');output.mkdir(parents=True,exist_ok=False)
    os.environ['HPSAFEMOE_RUN_DIR']=str(output)
    import torch
    torch.set_num_threads(4)
    module=importlib.import_module(f'hpsafemoe.stage2_proposal.{args.dataset}.{args.architecture}.training')
    cfg=json.loads((ROOT/f'checkpoint/{args.dataset}/stage2_proposal/{args.architecture}/training_config.json').read_text())
    fitseed=args.seed+args.inner*1000003
    if args.dataset=='jarvis':
        inputs_module=importlib.import_module(f'hpsafemoe.stage2_proposal.jarvis.{args.architecture}.input_statistics')
        inputs=inputs_module.Inputs();normalization=inputs.set_inner(args.inner)
        model,history=module.fit(inputs,'full',args.seed,args.inner,1400)
        torch.save({'model':model.state_dict(),'config':asdict(model.config),'normalization':normalization,'step':1400,'seed':args.seed,'fit_seed':fitseed,'inner':args.inner},output/'final.pt')
        (output/'history.json').write_text(json.dumps(history,indent=2))
    else:
        from hpsafe_sota.expert_inputs.protocol import load_expert_pool
        from hpsafe_sota.expert_inputs.data import FoldInputs
        pool=load_expert_pool(ROOT/'data/matbench/stage1_pool/pool/experts.yaml')
        fold=FoldInputs(project_root=ROOT/'results/matbench',pool=pool,outer_fold=args.outer,roles_to_load=('train','val'))
        assignment=module._inner_partitions(fold,5,seed=cfg['inner_partition_seed']+args.outer)
        eligible=module._partition_indices(fold,assignment,args.inner,training=True)
        fold.stats=module._masked_stats(fold,eligible)
        model=module._fit_one_model(fold=fold,candidate='ablation_no_cross_expert' if args.no_cross else 'proposal_full',config=cfg,seed=fitseed,eligible=eligible,device=torch.device('cuda:0'),progress_path=output/'progress.json',progress_offset=0,progress_total=1400,phase='proposal_learning')
        torch.save({'state_dict':model.state_dict(),'model_config':asdict(model.config),'normalization':{k:v.to_dict() for k,v in fold.stats.items()},'training_steps':1400,'outer_fold':args.outer,'inner_fold':args.inner,'seed':args.seed,'fit_seed':fitseed},output/'final.pt')
    (output/'run_config.json').write_text(json.dumps(vars(args),indent=2))
    print(output)
if __name__=='__main__':main()
