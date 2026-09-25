"""Full two-channel inverted rank-four states with inherited radial boundaries."""
from pathlib import Path
from functools import lru_cache
import json
import numpy as np
from scipy.linalg import eigh_tridiagonal
from scipy.sparse import diags
from scipy.sparse.linalg import eigsh

checks=[]
def ck(name,condition):
    passed=bool(condition)
    checks.append({"name":name,"passed":passed})
    if not passed:
        raise AssertionError(name)

def scalar_form(u,r,h,ell):
    return np.sum(np.diff(np.r_[0.,u,0.])**2)/(2*h*h)+np.dot(u*u,ell*(ell+1)/(2*r*r)-r*r/2)

@lru_cache(None)
def scalar(ell,cutoff,size):
    h=cutoff/(size+1)
    r=h*np.arange(1,size+1)
    d=1/h**2+ell*(ell+1)/(2*r*r)-r*r/2
    off=np.full(size-1,-.5/h**2)
    _,vec=eigh_tridiagonal(d,off,select='i',select_range=(0,0),tol=1e-12)
    u=vec[:,0]
    if np.sum(u)<0:
        u=-u
    energy=scalar_form(u,r,h,ell)
    return energy,u

def coupled(L,v,cutoff,size):
    h=cutoff/(size+1)
    r=h*np.arange(1,size+1)
    d=np.empty(2*size)
    d[0::2]=1/h**2+L*(L-1)/(2*r*r)-r*r/2
    d[1::2]=1/h**2+L*(L+1)/(2*r*r)-r*r/2
    adjacent=np.zeros(2*size-1)
    adjacent[0::2]=-v*r
    spatial=np.full(2*size-2,-.5/h**2)
    matrix=diags([spatial,adjacent,d,adjacent,spatial],[-2,-1,0,1,2],format='csc')
    # This shift is below the exact quadratic-form lower bound, so the nearest
    # shift-invert eigenpair is the ground, not a selected excited eigenvalue.
    shift=-cutoff*cutoff/2-abs(v)*cutoff-1
    _,vec=eigsh(matrix,k=1,sigma=shift,which='LM',tol=2e-12,
                v0=np.exp(-2*(cutoff-np.repeat(r,2))))
    u=vec[:,0].reshape(size,2)
    if np.sum(u)<0:
        u=-u
    energy=scalar_form(u[:,0],r,h,L-1)+scalar_form(u[:,1],r,h,L)-2*v*np.dot(r,u[:,0]*u[:,1])
    residual=np.linalg.norm(matrix@u.ravel()-energy*u.ravel())
    return energy,u,residual

def state(cutoff,lam,size):
    v=np.sqrt(2)*lam
    r=cutoff*np.arange(1,size+1)/(size+1)
    e0,u0=scalar(0,cutoff,size)
    e11,u11=scalar(11,cutoff,size)
    e1,u1,res1=coupled(1,v,cutoff,size)
    e12,u12,res12=coupled(12,v,cutoff,size)
    es,ev=e12+e0,e1+e11
    raw_overlap=np.dot(u11,u12[:,0])*np.dot(u1[:,0],u0)
    return {"cutoff":cutoff,"lambda":lam,"size":size,
            "E_S":es,"E_V":ev,"E_V_minus_E_S":ev-es,
            "scaled_gap":cutoff**2*(ev-es),
            "radial_overlap_squared":raw_overlap**2,
            "J_ground_weight":(48/23)*raw_overlap**2,
            "S_upper_heavy_probability":(1-2*np.dot(u12[:,0],u12[:,1]))/2,
            "V_upper_heavy_probability":(1-2*np.dot(u1[:,0],u1[:,1]))/2,
            "S_mean_outer_distance":cutoff-np.dot(r,np.sum(u12*u12,axis=1)),
            "V_mean_outer_distance":cutoff-np.dot(r,np.sum(u1*u1,axis=1)),
            "maximum_eigen_residual":max(res1,res12)}

def extrapolate(rows,key):
    # h=Lambda/(M+1), so use this exact ratio rather than an assumed factor4.
    a,b=rows[-2:]
    ratio=((b['size']+1)/(a['size']+1))**2
    return (ratio*b[key]-a[key])/(ratio-1)

records=[]
parameter_pairs=[(cutoff,lam) for cutoff in [2.,4.,8.,16.] for lam in [0.,.7,2.]]+[(2.,8.),(2.,16.)]
for cutoff,lam in parameter_pairs:
    rows=[state(cutoff,lam,m) for m in [400,800,1600]]
    previous=extrapolate(rows[:2],'E_V_minus_E_S')
    final=extrapolate(rows,'E_V_minus_E_S')
    ck(f"Lambda={cutoff},lambda={lam}: Richardson gap convergence",abs(final-previous)<3e-5)
    ck(f"Lambda={cutoff},lambda={lam}: nonzero normalized ground overlap",rows[-1]['radial_overlap_squared']>1e-4)
    if lam==0:
        ck(f"Lambda={cutoff}: exact zero-coupling degeneracy",max(abs(row['E_V_minus_E_S']) for row in rows)<2e-8)
        ck(f"Lambda={cutoff}: conserved-charge ground weight",abs(rows[-1]['J_ground_weight']-48/23)<2e-8)
    records.append({"cutoff":cutoff,"lambda":lam,"grids":rows,
                    "extrapolated_gap":final,"extrapolated_scaled_gap":cutoff**2*final,
                    "Richardson_change":abs(final-previous),
                    "extrapolated_ground_weight":extrapolate(rows,'J_ground_weight')})

# Independent nonuniform-coupling crossover: v=xi/Lambda^3.
crossover=[]
for xi in [.5,2.,6.,20.]:
    limit=-5.5+np.sqrt(xi*xi+36)-np.sqrt(xi*xi+.25)
    s12,s1=np.sqrt(xi*xi+36),np.sqrt(xi*xi+.25)
    weight_limit=(48/23)*(1+6/s12)*(1+.5/s1)/4
    seq=[]
    for cutoff in [8.,16.,32.]:
        lam=xi/(np.sqrt(2)*cutoff**3)
        rows=[state(cutoff,lam,m) for m in [800,1600]]
        gap=extrapolate(rows,'E_V_minus_E_S')
        seq.append({"cutoff":cutoff,"extrapolated_scaled_gap":cutoff**2*gap,
                    "extrapolated_ground_weight":extrapolate(rows,'J_ground_weight')})
    ck(f"xi={xi}: negative crossover gap",all(row['extrapolated_scaled_gap']<0 for row in seq))
    ck(f"xi={xi}: scaled-gap approach to analytic limit",abs(seq[-1]['extrapolated_scaled_gap']-limit)<abs(seq[0]['extrapolated_scaled_gap']-limit))
    ck(f"xi={xi}: charge-weight approach to analytic limit",abs(seq[-1]['extrapolated_ground_weight']-weight_limit)<abs(seq[0]['extrapolated_ground_weight']-weight_limit))
    crossover.append({"xi":xi,"limit_scaled_gap":limit,"limit_ground_weight":weight_limit,"sequence":seq})

result={"passed":True,"number_of_checks":len(checks),"checks":checks,
        "fixed_coupling":records,"crossover":crossover,
        "scope":"Normalizable finite-rank, finite-Dirichlet-cutoff states; no sea or string limit."}
(Path(__file__).resolve().parents[1] / 'data_exports/mqm/mqm_majorana_rank4_cutoff_state_results.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps({"passed":True,"checks":len(checks),
                  "fixed_coupling":[{k:r[k] for k in ['cutoff','lambda','extrapolated_gap','extrapolated_scaled_gap','Richardson_change','extrapolated_ground_weight']} for r in records],
                  "crossover":crossover},indent=2))
