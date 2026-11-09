import math
import numpy as np
import torch

def train_step(inputs,variant,model,opt,rng,trainids,policy,device):
    if variant != 'full':raise ValueError('Only the Full main model is released')
    model.train();opt.zero_grad(set_to_none=True);losses=[];anchors=[];aux=[];entropy=[]
    for task in inputs.tasks:
        classification=inputs.kinds[task]=='classification'
        size=policy['classification_batch_size'] if classification else policy['regression_batch_size']
        ids=trainids[task];idx=rng.choice(len(ids),size,replace=len(ids)<size);selected=[ids[i] for i in idx]
        batch=inputs.batch(task,selected,'train',device,policy['source_dropout'],rng);out=model(**batch)
        y=np.asarray([inputs.labels[task]['train'][j] for j in selected],dtype=np.float64)
        if not classification:y=(y-inputs.stats[task][0])/inputs.stats[task][1]
        target=torch.as_tensor(y,dtype=torch.float32,device=device);criterion=torch.nn.functional.binary_cross_entropy_with_logits if classification else torch.nn.functional.l1_loss
        losses.append(criterion(out.raw_standard,target));anchors.append(criterion(batch['anchor_standard'],target).detach());aux.append((criterion(out.data_standard,target)+criterion(out.physics_standard,target))/2)
        probs=out.route_probabilities[:,1:].clamp_min(1e-8);entropy.append(-(probs*probs.log()).sum(1).mean())
    v=torch.stack(losses);regret=torch.relu(v-torch.stack(anchors));cvar=torch.topk(regret,k=math.ceil(.25*len(inputs.tasks))).values.mean();h=torch.stack(entropy).mean()
    objective=v.mean()+.75*regret.mean()+.25*cvar+.20*torch.stack(aux).mean()+.003*(h-.45).square()
    assert torch.isfinite(objective);objective.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),2);opt.step()
    return float(objective.detach()), float(regret.mean().detach())
