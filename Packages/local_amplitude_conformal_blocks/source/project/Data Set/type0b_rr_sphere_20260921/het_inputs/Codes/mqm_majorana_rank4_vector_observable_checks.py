"""Fixed-normalization broken-flavor observable and exact radial sum rules.

Representation/Clifford tests are exact. Jacobi truncations test spectral
convergence and exact finite-matrix sum rules; they are not rigorous error
enclosures for the infinite operator.
"""
from pathlib import Path
import json

import numpy as np
import sympy as s
from scipy.linalg import eigh_tridiagonal

tests=[]


def check(name, condition):
    if not bool(condition):
        raise AssertionError(name)
    tests.append(name)


d=23
S=s.zeros(d+1)
S[d,d]=s.sqrt(s.Rational(d,d+1))
for a in range(d):
    S[a,a]=-1/s.sqrt(d*(d+1))
check("zonal_norm",s.trace(S.conjugate().T*S)==1)
check("zonal_trace",s.trace(S)==0)
total=s.Integer(0)
for a in range(d):
    J=s.zeros(d+1)
    J[a,d],J[d,a]=s.I,-s.I
    V=s.zeros(d+1)
    V[a,d]=V[d,a]=1/s.sqrt(2)
    JS=J*S+S*J.T
    check(f"vector_coefficient_{a}",JS==s.I*s.sqrt(s.Rational(48,23))*V)
    total+=s.trace(JS.conjugate().T*JS)
check("ST2_Casimir48",total==48)


def clifford(even_dimension):
    m=even_dimension//2
    p1=s.Matrix([[0,1],[1,0]])
    p2=s.Matrix([[0,-s.I],[s.I,0]])
    p3=s.diag(1,-1)
    result=[]
    for j in range(m):
        for p in (p1,p2):
            factors=[p3]*j+[p]+[s.eye(2)]*(m-j-1)
            result.append(s.kronecker_product(*factors))
    return result


gam=clifford(8)
chi=[[gam[4*i+a]/s.sqrt(2) for a in range(4)] for i in range(2)]
X=s.Matrix([[0,1],[-1,0]])
Hs=[]
for a in range(4):
    Hs.append(s.I*sum((chi[i][a]*X[i,j]*chi[j][a]
                      for i in range(2) for j in range(2)),s.zeros(16)))
for a in range(3):
    J=s.I*sum((chi[i][a]*chi[i][3] for i in range(2)),s.zeros(16))
    first=Hs[3]*J-J*Hs[3]
    double=J*first-first*J
    check(f"Clifford_double_commutator_{a}",double==2*(Hs[a]-Hs[3]))
Q=-s.I*sum((chi[0][a]*chi[1][a] for a in range(4)),s.zeros(16))
Htrace=sum(Hs,s.zeros(16))
check("trace_Yukawa_is_Gauss",Htrace==-2*Q)
for idx,v in enumerate(Q.nullspace()):
    check(f"trace_Yukawa_on_Gauss_{idx}",Htrace*v==s.zeros(16,1))


def radial(L, coupling, Omega, dimension):
    a=L+0.5
    diag=Omega*(a+np.arange(dimension))
    n=np.arange(dimension-1)
    q=np.where(n%2==0,n//2+a,(n+1)//2)
    edge=-abs(coupling)/np.sqrt(Omega)*np.sqrt(q)
    return eigh_tridiagonal(diag,edge)


def data(lam,Omega,dimension):
    coupling=np.sqrt(2)*lam
    ES,VS=radial(12,coupling,Omega,dimension)
    EV,VV=radial(1,coupling,Omega,dimension)
    cs=VS[:,0]
    if cs[0]<0:cs=-cs
    a2=np.abs(cs[::2])**2
    b2=np.abs(VV[0,:])**2
    k=np.arange(len(a2))
    gaps=EV[None,:]-ES[0]+Omega*(11+2*k[:,None])
    weights=(48/23)*a2[:,None]*b2[None,:]
    p=a2.sum()
    zeroth=weights.sum()
    first=(weights*gaps).sum()
    direct_first=(48/23)*(np.dot(a2,Omega*(12.5+2*k))-ES[0]*p)
    check(f"lambda{lam}_D{dimension}_positive_gaps",gaps.min()>0 if lam else abs(gaps.min())<1e-12)
    check(f"lambda{lam}_D{dimension}_sum0",abs(zeroth-(48/23)*p)<1e-12)
    check(f"lambda{lam}_D{dimension}_sum1",abs(first-direct_first)<2e-11)
    # Full nonlinear Yukawa is linear in the Jacobi off-diagonal entries.
    n=np.arange(dimension-1)
    q=np.where(n%2==0,n//2+12.5,(n+1)//2)
    HY=-2*abs(coupling)/np.sqrt(Omega)*np.dot(np.sqrt(q),cs[:-1]*cs[1:])
    check(f"lambda{lam}_D{dimension}_double_commutator_sum1",
          abs(first+24*HY/23)<2e-11)
    return {"lambda":lam,"dimension":dimension,"singlet_energy":float(ES[0]+1.5*Omega),
            "vector_lowest_energy":float(EV[0]+12.5*Omega),
            "lowest_gap":float(gaps[0,0]),"lower_probability":float(p),
            "total_weight_per_flavor":float(zeroth),
            "lowest_line_weight_per_flavor":float(weights[0,0]),
            "first_moment_per_flavor":float(first)}


samples=[]
for lam in (0.0,0.1,0.7,2.0,4.0):
    small=data(lam,1.0,128)
    large=data(lam,1.0,192)
    for key in ("singlet_energy","vector_lowest_energy","lowest_gap","lower_probability",
                "total_weight_per_flavor","lowest_line_weight_per_flavor","first_moment_per_flavor"):
        check(f"lambda{lam}_convergence_{key}",abs(small[key]-large[key])<5e-11)
    samples.append(large)
zero=samples[0]
check("zero_coupling_single_zero_frequency_line",abs(zero["total_weight_per_flavor"]-48/23)<1e-14
      and zero["lowest_line_weight_per_flavor"]==zero["total_weight_per_flavor"])

small=data(1e-4,1.,128)
check("weak_lower_probability",abs(small["lower_probability"]-(1-25e-8))<1e-12)
check("weak_lowest_line",abs(small["lowest_line_weight_per_flavor"]/(48/23)-(1-28e-8))<1e-12)
check("weak_first_moment",abs(small["first_moment_per_flavor"]/(48/23)-25e-8)<1e-12)

output={"status":"passed","checks":len(tests),"samples":samples,"named_checks":tests}
(Path(__file__).resolve().parents[1] / 'data_exports/mqm/mqm_majorana_rank4_vector_observable_results.json').write_text(json.dumps(output,indent=2)+"\n")
print(json.dumps({"status":"passed","checks":len(tests),"samples":samples},indent=2))
