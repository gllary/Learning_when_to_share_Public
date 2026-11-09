"""Build one immutable, target-free graph cache for a strong expert recipe."""
import argparse
import json
import os
from common import ROOT
from strong_backend import load_graph_dataset,receipt_path,universe,write_receipt,verify_receipt


def main():
    p=argparse.ArgumentParser();p.add_argument("--recipe",required=True);p.add_argument("--force-check",action="store_true");a=p.parse_args()
    ids,_=universe();receipt=receipt_path(a.recipe)
    if receipt.exists() and not a.force_check:
        verify_receipt(a.recipe)
        print("CACHE_ALREADY_READY",receipt,flush=True);return
    ds=load_graph_dataset(a.recipe,build=True)
    if len(ds)!=len(ids):raise RuntimeError((len(ds),len(ids)))
    # Cache samples contain only a constant dummy y; real labels live in task manifests.
    probes=[0,len(ds)//2,len(ds)-1]
    assert all(float(ds[i].y)==0.0 for i in probes)
    path=write_receipt(a.recipe,ds,ids)
    print(json.dumps({"status":"ready","receipt":str(path),"rows":len(ds)}),flush=True)


if __name__=="__main__":main()
