"""Small endpoint energy-score adapter; independent draws are never trajectories."""
import numpy as np
from temporal_eval.metrics import align_future


def features(coordinates, observed, representation):
    x=np.asarray(coordinates,dtype=float)
    if x.ndim==2:x=x[None]
    ca=np.asarray(representation['ca_indices'],dtype=int)
    fit=np.asarray(representation['align_indices'],dtype=int)
    pairs=np.asarray(representation['internal_pairs_ca_indices'],dtype=int)
    aligned=align_future(x,np.asarray(observed,dtype=float),fit)[:,ca]
    internal=np.linalg.norm(aligned[:,pairs[:,0]]-aligned[:,pairs[:,1]],axis=-1)
    return {'CA':aligned.reshape(len(x),-1)/np.sqrt(len(ca)),
            'internal':internal/np.sqrt(len(pairs))}


def nonself_distance(x):
    x=np.asarray(x,dtype=float)
    if len(x)<2:return None
    a,b=np.triu_indices(len(x),1)
    return float(np.linalg.norm(x[a]-x[b],axis=1).mean())


def energy_score(x,target,requested=4):
    x=np.asarray(x,dtype=float);target=np.asarray(target,dtype=float).reshape(-1)
    if x.ndim!=2 or x.shape[1:]!=target.shape or not np.isfinite(x).all() or not np.isfinite(target).all():
        raise ValueError('Finite fixed-dimensional energy features required')
    n=len(x);pair=nonself_distance(x)
    error=float(np.linalg.norm(x-target,axis=1).mean()) if n else None
    return dict(draws=n,requested_draws=requested,nonself_unordered_draw_pairs=n*(n-1)//2,
                target_distance_mean_A=error,nonself_pair_distance_mean_A=pair,
                energy_score_A=None if pair is None else error-.5*pair,
                status='ok' if n>=requested else 'partial',
                reason=None if n>=requested else 'Fewer than four authorized independent draws; actual n retained',
                estimator='mean norm to single heldout MD - half mean nonself draw-pair norm; unbiased iid-draw estimate; finite value may be negative')


def distribution_distance(a,b,seeds_a,seeds_b):
    a,b=np.asarray(a,dtype=float),np.asarray(b,dtype=float)
    if len(a)<2 or len(b)<2:return dict(status='NA',reason='At least two independent draws from each ensemble',draws_a=len(a),draws_b=len(b))
    if len(seeds_a)!=len(a) or len(seeds_b)!=len(b) or len(set(seeds_a))!=len(a) or len(set(seeds_b))!=len(b):
        raise ValueError('Unique actual seed identities required for cross-ensemble pairs')
    nonshared=np.asarray(seeds_a)[:,None]!=np.asarray(seeds_b)[None]
    cross_n=int(nonshared.sum())
    if not cross_n:return dict(status='NA',reason='No cross-ensemble pairs with different seeds',cross_n=0)
    cross=float(np.linalg.norm(a[:,None]-b[None],axis=-1)[nonshared].mean())
    aa,bb=nonself_distance(a),nonself_distance(b)
    return dict(status='ok',draws_a=len(a),draws_b=len(b),cross_distance_A=cross,cross_n=cross_n,
                excluded_same_seed_pairs=int(nonshared.size-cross_n),
                nonself_a_A=aa,nonself_b_A=bb,energy_distance_A=2*cross-aa-bb,
                interpretation='Same-seed cross pairs excluded because reused normal streams can correlate paths; finite U estimate may be negative; no hard coordinate-marginal CK constraint')


def distance_curve(xyz,observed,representation,dt_ps,order):
    """Existing G numerator/platform expanded by lag; normalized features match history helper."""
    result=[]
    for observable,feat in features(xyz,observed,representation).items():
        n=len(feat)
        variance=float(np.square(feat-feat.mean(0)).sum(1).mean())
        platform=2*n/(n-1)*variance if n>1 else None
        count=representation['ca_count'] if observable=='CA' else representation['pair_count']
        for lag in range(1,min(24,n-1)+1):
            value=float(np.square(feat[lag:]-feat[:-lag]).sum(1).mean())
            ratio=value/platform if platform is not None and platform*count>1e-16 else None
            key=('ca' if observable=='CA' else 'distance')+f'_lag{lag}_order_gain'
            gain=order.get(key);error=abs(ratio-(1-gain)) if ratio is not None and gain is not None else None
            if error is not None and error>1e-10:raise ValueError('Original D/platform=1-G identity failed')
            result.append(dict(observable=observable,lag_frames=lag,lag_ps=float(lag*dt_ps),n_pairs=n-lag,
                               D_A2=value,shuffle_platform_A2=platform,D_over_shuffle=ratio,G_identity_abs_error=error))
    return result
