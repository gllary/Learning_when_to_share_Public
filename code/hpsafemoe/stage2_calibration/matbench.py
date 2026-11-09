"""Single fixed task-level Stage2 safety calibration rule. No model, feature gate or profile choice."""
import numpy as np

ALPHAS=(0.,.25,.5,.75,1.)
EPSILON_GAIN_PP=1.
SCHEMA='matbench-safety-calibration-v1'

def validate(a,r,y=None,classification=False):
    a,r=np.asarray(a,dtype=np.float64),np.asarray(r,dtype=np.float64)
    assert a.ndim==r.ndim==1 and a.shape==r.shape and len(a)
    assert np.isfinite(a).all() and np.isfinite(r).all()
    if classification:assert ((a>=0)&(a<=1)).all() and ((r>=0)&(r<=1)).all()
    if y is not None:
        y=np.asarray(y,dtype=np.float64);assert y.shape==a.shape and np.isfinite(y).all()
        if classification:assert set(np.unique(y))=={0,1}
    return a,r,y

def apply_beta(a,r,beta):
    a,r=np.asarray(a,dtype=np.float64),np.asarray(r,dtype=np.float64)
    assert a.shape==r.shape and beta in ALPHAS and np.isfinite(a).all()
    if beta==0:return a.copy()
    assert np.isfinite(r).all()
    return r.copy() if beta==1 else a+beta*(r-a)

class AUCPlan:
    """Exact grouped pair-count AUC; weighted multiplicities equal resampling rows."""
    def __init__(self,y,p):
        y,p=np.asarray(y),np.asarray(p,dtype=np.float64)
        assert y.ndim==p.ndim==1 and y.shape==p.shape and np.isfinite(p).all()
        assert set(np.unique(y))=={0,1}
        _,inverse=np.unique(p,return_inverse=True)
        self.pos=inverse[y==1];self.neg=inverse[y==0];self.groups=int(inverse.max())+1
    def score(self,pos_counts=None,neg_counts=None):
        pc=np.bincount(self.pos,weights=pos_counts,minlength=self.groups)
        nc=np.bincount(self.neg,weights=neg_counts,minlength=self.groups)
        assert pc.sum()>0 and nc.sum()>0
        # Equal-score pairs contribute exactly one half.
        return float(np.sum(pc*(np.cumsum(nc)-.5*nc))/(pc.sum()*nc.sum()))

def score(y,p,classification):
    return AUCPlan(y,p).score() if classification else float(np.abs(np.asarray(y)-np.asarray(p)).mean())

def normalized_gain(base,new,classification):
    return 100.*(new-base)/max(1.-base,1e-12) if classification else 100.*(base-new)/max(abs(base),1e-12)

def select_alpha(a,r,y,classification):
    a,r,y=validate(a,r,y,classification)
    scores=[score(y,apply_beta(a,r,x),classification) for x in ALPHAS]
    i=min(range(len(ALPHAS)),key=lambda j:(-scores[j] if classification else scores[j],ALPHAS[j]))
    return ALPHAS[i],scores

def meta_partitions(y,classification,seed):
    y=np.asarray(y);rng=np.random.default_rng(seed);out=np.empty(len(y),dtype=np.int8)
    groups=[np.flatnonzero(y==k) for k in (0,1)] if classification else [np.arange(len(y))]
    for index in groups:
        assert len(index)>=5
        rng.shuffle(index);out[index]=np.arange(len(index),dtype=np.int64)%5
    return out

def paired_bootstrap(y,a,q,classification,seed,resamples=5000):
    y,a,q=map(np.asarray,(y,a,q));base=score(y,a,classification);new=score(y,q,classification)
    rng=np.random.default_rng(seed);values=np.empty(resamples,dtype=np.float64)
    if classification:
        plan_a,plan_q=AUCPlan(y,a),AUCPlan(y,q)
        pos=int((y==1).sum());neg=int((y==0).sum())
        for i in range(resamples):
            pc=np.bincount(rng.integers(0,pos,size=pos),minlength=pos)
            nc=np.bincount(rng.integers(0,neg,size=neg),minlength=neg)
            values[i]=normalized_gain(plan_a.score(pc,nc),plan_q.score(pc,nc),True)
    else:
        error_a,error_q=np.abs(a-y),np.abs(q-y)
        if np.array_equal(error_a,error_q):values.fill(0.)
        else:
            for start in range(0,resamples,16):
                idx=rng.integers(0,len(y),size=(min(16,resamples-start),len(y)))
                ea,eq=error_a[idx].mean(1),error_q[idx].mean(1)
                values[start:start+len(idx)]=100.*(ea-eq)/np.maximum(ea,1e-12)
    assert np.isfinite(values).all()
    return dict(G=normalized_gain(base,new,classification),L=float(np.quantile(values,.05)),
                upper_diagnostic=float(np.quantile(values,.95)),confidence=.95,resamples=resamples,
                resample_size=len(y),class_counts_preserved=bool(classification),seed=int(seed),
                method='paired_full_n_percentile',metric='roc_auc' if classification else 'mae',
                baseline_oof_score=base,crossfit_oof_score=new,denominator_floor=1e-12),values

def accept_liberal(G,L):
    assert np.isfinite(G) and np.isfinite(L)
    return bool(G>0 and L>=-EPSILON_GAIN_PP)

def calibrate(a,r,y,classification,seed,resamples=5000):
    a,r,y=validate(a,r,y,classification);fold=meta_partitions(y,classification,seed)
    q=np.empty_like(a);rows=[]
    for k in range(5):
        fit,held=fold!=k,fold==k;alpha,candidates=select_alpha(a[fit],r[fit],y[fit],classification)
        q[held]=apply_beta(a[held],r[held],alpha)
        rows.append(dict(held_fold=k,fit_rows=int(fit.sum()),held_rows=int(held.sum()),alpha=alpha,
                         fit_candidate_scores=candidates,
                         held_gain_diagnostic=normalized_gain(score(y[held],a[held],classification),score(y[held],q[held],classification),classification)))
    cert,b=paired_bootstrap(y,a,q,classification,seed+1777,resamples)
    alpha,candidates=select_alpha(a,r,y,classification);passed=accept_liberal(cert['G'],cert['L']);beta=alpha if passed else 0.
    result=dict(schema=SCHEMA,rule='fixed_liberal',epsilon_gain_pp=EPSILON_GAIN_PP,alpha_grid=list(ALPHAS),
                alpha_all_oof=alpha,beta=beta,task_rule_passed=passed,correction_enabled=bool(beta>0),
                metric='roc_auc' if classification else 'mae',threshold_classification_scores=False,
                certificate=cert,meta_folds=rows,meta_seed=seed,candidate_scores_all_oof=candidates,
                positive_fold_count_diagnostic=sum(x['held_gain_diagnostic']>0 for x in rows),
                worst_fold_gain_diagnostic=min(x['held_gain_diagnostic'] for x in rows),
                fold_diagnostics_used_for_acceptance=False,final_alpha_method='all_matching_outer_train_oof_grid_search',
                independent_risk_guarantee=False)
    return result,dict(meta_fold=fold,crossfit_prediction=q,bootstrap_gains=b)
