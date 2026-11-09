import importlib
from pathlib import Path
import sys
import unittest
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'code'))

class ProposalContract(unittest.TestCase):
    def make(self,dataset,cross):
        mod=importlib.import_module(f'hpsafemoe.stage2_proposal.{dataset}.model.model')
        names=('a','b','c','d')
        cfg=mod.ProposalConfig(task_names=('a','b'),expert_ids=names,hidden_dims={n:8 for n in names},anchor_indices=(0,1),token_dim=16,task_dim=8,residual_hidden=16,residual_rank=4,dropout=0.,attention_top_k=4,residual_limit=1.25,route_temperature=.75,use_cross_expert_input=cross,use_data_branch=True,use_physics_branch=True,use_learned_task_source_relation=cross,use_shared_low_rank_basis=False,use_task_private_correction=True,use_sample_gate=True)
        x=dict(task_index=torch.zeros(3,dtype=torch.long),anchor_standard=torch.tensor([.1,.2,.3]),hidden={n:torch.randn(3,8) for n in names},prediction_standard={n:torch.randn(3) for n in names},uncertainty_standard={n:torch.rand(3) for n in names},available=torch.ones(3,4,dtype=torch.bool))
        return mod,cfg,x
    def test_matbench_internal_mask(self):
        torch.set_num_threads(1)
        mod,cfg,x=self.make('matbench',False);model=mod.CrossExpertProposalModel(cfg).eval()
        with torch.no_grad():
            before=model(**x).raw_standard
            for n in cfg.expert_ids[1:]:
                x['hidden'][n].fill_(float('nan'));x['prediction_standard'][n].fill_(float('inf'));x['uncertainty_standard'][n].fill_(float('nan'))
            after=model(**x).raw_standard
        self.assertTrue(torch.isfinite(after).all());torch.testing.assert_close(before,after,rtol=0,atol=0)
    def test_full_soft_mixture(self):
        torch.set_num_threads(1)
        for ds in ('jarvis','matbench'):
            mod,cfg,x=self.make(ds,True);out=mod.CrossExpertProposalModel(cfg).eval()(**x)
            expected=x['anchor_standard']+out.route_weights[:,1]*out.data_residual+out.route_weights[:,2]*out.physics_residual
            torch.testing.assert_close(out.raw_standard,expected)
    def test_jarvis_scope(self):
        mod,cfg,_=self.make('jarvis',False)
        with self.assertRaises(ValueError):mod.CrossExpertProposalModel(cfg)
if __name__=='__main__':unittest.main()
