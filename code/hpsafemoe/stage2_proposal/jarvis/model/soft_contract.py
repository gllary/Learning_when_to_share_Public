import numpy as np
def validate_record(a):
    n=len(a['sample_ids']);w=np.asarray(a['route_weights']);c=np.asarray(a['routing_components'])
    rp=np.asarray(a['restore_parameters'])
    assert w.shape==(n,3) and c.shape==(n,4) and rp.shape==(n,3)
    assert np.isfinite(w).all() and np.isfinite(c).all() and np.isfinite(rp).all()
    assert (w>=0).all() and (w<=1).all()
    np.testing.assert_allclose(w.sum(1),1,rtol=0,atol=2e-7)
    assert np.array_equal(w,a['route_probabilities'])
    assert np.array_equal(w.argmax(1),a['route_index'])
    assert np.isin(rp[:,0],[0,1]).all() and (rp[:,2]>0).all()
    anchor,sc,rc,raw=c.T
    expected=(w*np.stack([anchor,anchor+sc,anchor+rc],1)).sum(1)
    np.testing.assert_allclose(raw,expected,rtol=1e-6,atol=2e-6)
    lo=np.minimum(anchor+sc,anchor+rc);hi=np.maximum(anchor+sc,anchor+rc)
    normal=w[:,0]==0
    tol=8*np.finfo(c.dtype).eps*(1+np.maximum(np.abs(lo),np.abs(hi)))
    assert np.all(raw[normal]>=lo[normal]-tol[normal]) and np.all(raw[normal]<=hi[normal]+tol[normal])
    fallback=w[:,0]>0
    assert np.all(w[fallback,0]==1) and np.all(w[fallback,1:]==0)
    assert np.array_equal(raw[fallback],anchor[fallback])
    restored=raw.astype(np.float64)*rp[:,2]+rp[:,1]
    clf=rp[:,0]==1
    restored[clf]=1/(1+np.exp(-np.clip(raw[clf].astype(np.float64),-40,40)))
    np.testing.assert_array_equal(restored,a['raw_prediction'])
    return True
