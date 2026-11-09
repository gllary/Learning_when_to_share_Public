"""Read at most four target-free graph shards per worker."""
from collections import OrderedDict
import json
import numpy as np
from common import ROOT

class GraphStore:
    def __init__(self):
        self.ids=np.load(ROOT/'data/universe_ids.npy').astype(str)
        self.index={j:i for i,j in enumerate(self.ids)};self.cache=OrderedDict();self.preloaded={}
        self.receipt=json.loads((ROOT/'data/alignn_graphs.receipt.json').read_text())
        from cache_contract import verify
        verify(self.receipt,'alignn',[(ROOT/'data/alignn_shards'/s['file'],s['sha256']) for s in self.receipt['shards']])
    def get(self,jid):
        import dgl
        if jid in self.preloaded:return self.preloaded[jid]
        i=self.index[jid];size=self.receipt['shard_size'];shard=i//size
        if shard not in self.cache:
            graphs,_=dgl.load_graphs(str(ROOT/'data/alignn_shards'/self.receipt['shards'][shard]['file']))
            self.cache[shard]=graphs
            if len(self.cache)>4:self.cache.popitem(last=False)
        self.cache.move_to_end(shard);graphs=self.cache[shard];local=i%size
        return graphs[2*local],graphs[2*local+1]

    def preload(self,jids):
        """Load the task's training graphs once, avoiding random whole-shard I/O."""
        import dgl
        groups={};size=self.receipt['shard_size'];estimate=0
        for j in jids:groups.setdefault(self.index[j]//size,[]).append(j)
        for shard,selected in groups.items():
            idx=[]
            for j in selected:
                local=self.index[j]%size;idx.extend([2*local,2*local+1])
            graphs,_=dgl.load_graphs(str(ROOT/'data/alignn_shards'/self.receipt['shards'][shard]['file']),idx_list=idx)
            for n,j in enumerate(selected):
                pair=(graphs[2*n],graphs[2*n+1]);self.preloaded[j]=pair
                for g in pair:
                    estimate+=4096+g.num_edges()*16+g.num_nodes()*8
                    # DGL 1.x exposes `DataView.values()` as Column wrappers
                    # on some builds; graph[key] reliably returns the tensor.
                    for view in (g.ndata,g.edata):
                        for key in view.keys():
                            value=view[key]
                            estimate+=value.numel()*value.element_size()
            # A hard cap bounds this optimization even on pathological cells.
            if estimate>40*1024**3:raise MemoryError('Task graph preload exceeds 40 GiB safety budget')
        self.cache.clear()
        return estimate
