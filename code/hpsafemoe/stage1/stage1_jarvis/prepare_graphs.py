import argparse
import json
import multiprocessing as mp
import os
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parent/'src'))
import numpy as np
from common import protocol_hash
from common import ROOT,atomic_json,sha256
from hpsafe_cogn_stage1.graph_store import initialize_store,append_graphs,read_ids
from cache_contract import contract,verify

def one(item):
    jid,atoms,kind=item
    from official_preprocessing import kgcnn_structure
    from kgcnn.crystal.preprocessor import KNNAsymmetricUnitCell,VoronoiAsymmetricUnitCell
    pre=KNNAsymmetricUnitCell(24) if kind=='cogn' else VoronoiAsymmetricUnitCell(min_ridge_area=1e-6)
    structure=kgcnn_structure(atoms)
    return jid,pre(structure)

def main():
    p=argparse.ArgumentParser();p.add_argument('--kind',choices=['cogn','congn'],required=True);p.add_argument('--workers',type=int,default=4);a=p.parse_args()
    target=ROOT/'data'/(a.kind+'_graphs.h5');receipt=target.with_suffix('.receipt.json')
    ids=np.load(ROOT/'data'/'universe_ids.npy').astype(str)
    if target.is_file() and receipt.is_file():
        prior=json.loads(receipt.read_text());verify(prior,a.kind,[(target,prior['sha256'])])
        if set(ids)<=set(read_ids(target)):
            print('CACHE_READY',a.kind,'reused',flush=True);return
        raise RuntimeError('Existing cache has incomplete material coverage')
    structures=json.loads((ROOT/'data'/'structures.json').read_text())
    attrs=['offset'] if a.kind=='cogn' else ['offset','voronoi_ridge_area']
    tmp=target.with_suffix('.building.h5')
    if tmp.exists():raise RuntimeError('An incomplete cache exists; inspect before restarting')
    with initialize_store(tmp,model_name=a.kind) as h:
        ctx=mp.get_context('spawn')
        with ctx.Pool(a.workers) as pool:
            batch=[];completed=0
            for jid,g in pool.imap(one,((j,structures[j],a.kind) for j in ids),chunksize=8):
                batch.append((jid,g))
                if len(batch)>=64:
                    append_graphs(h,[x[1] for x in batch],[x[0] for x in batch],node_attributes=['atomic_number','multiplicity'],edge_attributes=attrs)
                    completed+=len(batch);batch=[];h.flush()
                    atomic_json(ROOT/'status'/(a.kind+'_cache.json'),{'completed':completed,'total':len(ids),'status':'building'})
                    print('CACHE_PROGRESS',a.kind,completed,len(ids),flush=True)
            if batch:append_graphs(h,[x[1] for x in batch],[x[0] for x in batch],node_attributes=['atomic_number','multiplicity'],edge_attributes=attrs)
    assert np.array_equal(read_ids(tmp),ids)
    os.replace(tmp,target);atomic_json(receipt,{'cache_contract':contract(a.kind),'sha256':sha256(target),'labels_read':False,'materials':len(ids),'kind':a.kind,'preprocessing':'official_POSCAR_CIF_primitive_True','source_sha256':sha256(ROOT/'official_preprocessing.py'),'protocol_sha256':protocol_hash()})
    atomic_json(ROOT/'status'/(a.kind+'_cache.json'),{'completed':len(ids),'total':len(ids),'status':'complete'})
    print('CACHE_READY',a.kind,flush=True)

if __name__=='__main__':main()
