import numpy as np
def predict(inputs,model,task,role,device,jids=None):
    import torch
    keys=['raw_prediction','data_prediction','physics_prediction','route_probabilities','route_index','data_attention','physics_attention','reliability','anchor_prediction','route_weights','routing_components','restore_parameters']
    parts={k:[] for k in keys};jids=inputs.roles[task][role] if jids is None else jids;model.eval()
    with torch.inference_mode():
        for start in range(0,len(jids),128):
            subset=jids[start:start+128];batch=inputs.batch(task,subset,role,device);out=model(**batch)
            for key,value in [('raw_prediction',out.raw_standard),('data_prediction',out.data_standard),('physics_prediction',out.physics_standard)]:parts[key].append(inputs.restore(task,value.cpu().numpy()))
            parts['anchor_prediction'].append(inputs.anchor(task,subset,role))
            parts['route_weights'].append(out.route_weights.cpu().numpy())
            parts['routing_components'].append(torch.stack([batch['anchor_standard'],out.data_residual,out.physics_residual,out.raw_standard],1).cpu().numpy())
            mean,scale=inputs.stats[task]
            parts['restore_parameters'].append(np.tile([float(inputs.kinds[task]=='classification'),mean,scale],(len(subset),1)))
            for key,value in [('route_probabilities',out.route_probabilities),('route_index',out.route_one_hot.argmax(1)),('data_attention',out.data_attention),('physics_attention',out.physics_attention),('reliability',out.reliability)]:parts[key].append(value.cpu().numpy())
    result={'sample_ids':np.asarray(jids,dtype=str),**{k:np.concatenate(v) for k,v in parts.items()}}
    assert all(np.isfinite(v).all() for k,v in result.items() if k!='sample_ids')
    from .soft_contract import validate_record
    validate_record(result)
    return result
