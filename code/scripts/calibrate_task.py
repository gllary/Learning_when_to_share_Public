"""Fit only the fixed task-level liberal rule from a proposal's OOF records."""
import argparse
import importlib
import json
from pathlib import Path
import sys
import numpy as np
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'code'))

if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--dataset',required=True,choices=['matbench','jarvis'])
    p.add_argument('--oof',required=True,help='NPZ with anchor, raw and y_train_oof (JARVIS) or y_oof (Matbench); never test labels')
    p.add_argument('--classification',action='store_true');p.add_argument('--seed',required=True,type=int)
    p.add_argument('--output',required=True);args=p.parse_args()
    output=Path(args.output).resolve();assert output.is_relative_to(ROOT/'runs')
    output.mkdir(parents=True,exist_ok=False)
    with np.load(args.oof,allow_pickle=False) as z:
        anchor=z['anchor'];raw=z['raw'];y=z['y_train_oof' if args.dataset=='jarvis' else 'y_oof']
    module=importlib.import_module('hpsafemoe.stage2_calibration.'+args.dataset)
    result,arrays=module.calibrate(anchor,raw,y,args.classification,args.seed,resamples=5000)
    with (output/'calibration.json').open('x') as f:json.dump(result,f,indent=2)
    label_key='y_train_oof' if args.dataset=='jarvis' else 'y_oof'
    np.savez_compressed(output/'calibration_records.npz',anchor=anchor,raw=raw,**{label_key:y},**arrays)
    print(json.dumps({'beta':result['beta'],'certificate':result['certificate']}))
