"""Read author CLI/YAML recipes without running their downloader or trainer."""
import argparse
import ast
import shlex
import sys
from common import ROOT

PROPS={'formation':'formation_energy_peratom','total_energy':'optb88vdw_total_energy','opt_bandgap':'optb88vdw_bandgap','ehull':'ehull','mbj_bandgap':'mbj_bandgap'}

def pyg_configuration(recipe):
    import torch
    from torch_geometric.graphgym.config import cfg,set_cfg
    if recipe=='ehull':
        directory=ROOT/'vendor_ext/prism'
        args=['--seed','0','--figshare_target','ehull','--model','prism','--dataset','jarvis','--batch','64','--batch_accumulation','1','--lr','0.001','--epochs','500','--augment']
    else:
        directory=ROOT/'vendor_ext/atomnet/AtomNet'
        lines=(directory.parent/'README.md').read_text().splitlines()
        # First task-specific Linux command is the author's current training
        # recipe, unlike commented exploratory seeds in code/scripts/.
        command=next(s for s in lines if s.startswith('python main.py ') and '--figshare_target '+PROPS[recipe]+' ' in s and '--inference' not in s and '--ig' not in s)
        args=shlex.split(command)[2:]
    source=directory/'main.py';tree=ast.parse(source.read_text())
    block=next(n for n in tree.body if isinstance(n,ast.If) and '__main__' in ast.unparse(n.test))
    prefix=[]
    for node in block.body:
        rendered=ast.unparse(node)
        if rendered=='set_printing()':break
        # Configuration parsing skips execution-only assertions and cleanup.
        if isinstance(node,ast.Assert) and 'torch.cuda.is_available()' in rendered:continue
        if rendered=='torch.cuda.empty_cache()':continue
        prefix.append(node)
    namespace=dict(argparse=argparse,torch=torch,cfg=cfg,set_cfg=set_cfg)
    previous=sys.argv
    try:
        sys.argv=[str(source),*args]
        exec(compile(ast.Module(body=prefix,type_ignores=[]),str(source),'exec'),namespace)
    finally:sys.argv=previous
    return cfg,namespace['args'],source,args

def gsm_configuration():
    import yaml
    sys.path.insert(0,str(ROOT/'vendor_ext/gsmnet'))
    from models.config import TrainingConfig
    value=yaml.safe_load((ROOT/'vendor_ext/gsmnet/configs/gsmnet.yaml').read_text())
    value['target']='mbj_bandgap'
    config=TrainingConfig(**value)
    config.model.euclidean=config.euclidean
    return config
