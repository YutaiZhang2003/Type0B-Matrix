"""Exact sparse-tensor test of rational SO(n)_1 null annihilators."""
from collections import defaultdict
from fractions import Fraction as F
from itertools import product
from pathlib import Path
import json
import sympy as s

def clean(v): return {k:t for k,t in v.items() if t}
def add(*terms):
    out=defaultdict(F)
    for scale,v in terms:
        for k,t in v.items():out[k]+=scale*t
    return clean(out)
def J(v,i,a=0,b=1):
    out=defaultdict(F)
    for k,t in v.items():
        if k[i]==b:
            q=list(k);q[i]=a;out[tuple(q)]+=t
        elif k[i]==a:
            q=list(k);q[i]=b;out[tuple(q)]-=t
    return clean(out)
def Omega(v,i,j,n):
    out=defaultdict(F)
    for k,t in v.items():
        q=list(k);q[i],q[j]=q[j],q[i];out[tuple(q)]+=t
        if k[i]==k[j]:
            for a in range(n):
                q=list(k);q[i]=q[j]=a;out[tuple(q)]-=t
    return clean(out)
def bracket(v,i,j,n,a=0,b=1):
    return add((F(2,3),J(v,j,a,b)),
               (-F(1,n-1),J(Omega(v,i,j,n),i,a,b)),
               (F(1,3),Omega(J(v,i,a,b),i,j,n)))
def null(v,i,n,x,a=0,b=1):
    return add(*[(F(1,x[i]-x[j]),bracket(v,i,j,n,a,b))
                 for j in range(len(x)) if j!=i])
def basis4(n):
    pairs=[((0,1),(2,3)),((0,2),(1,3)),((0,3),(1,2))]
    out=[]
    for pairing in pairs:
        v={}
        for a,b in product(range(n),repeat=2):
            q=[None]*4
            for color,pair in zip((a,b),pairing):
                for j in pair:q[j]=color
            v[tuple(q)]=F(1)
        out.append(v)
    return out
def dot(u,v):return sum((t*v.get(k,F(0)) for k,t in u.items()),F(0))
checks={}; data={}
for n in (3,4,23,24):
    data[str(n)]={}
    trace={(a,a):F(1) for a in range(n)}
    st={(0,0):F(1),(1,1):-F(1)}
    anti={(0,1):F(1),(1,0):-F(1)}
    n2={}
    for label,v in (("trace",trace),("ST",st),("A",anti)):
        norm=F(0)
        for i in range(2):
            for a in range(n):
                for b in range(a+1,n):
                    z=null(v,i,n,(0,1),a,b)
                    norm+=dot(z,z)
        n2[label]=str(norm)
        checks[f"n{n}_N2_{label}"]=(norm==0 if label=="trace" else norm>0)
    data[str(n)]["N2_null_norms"]=n2
    B=basis4(n)
    for x in ((0,1,3,6),(-4,-1,2,8)):
        images=[[null(v,i,n,x) for v in B] for i in range(4)]
        gram=s.Matrix(3,3,lambda a,b:sum(dot(images[i][a],images[i][b]) for i in range(4)))
        pf=s.Matrix([F(1,(x[0]-x[1])*(x[2]-x[3])),
                     -F(1,(x[0]-x[2])*(x[1]-x[3])),
                     F(1,(x[0]-x[3])*(x[1]-x[2]))])
        checks[f"n{n}_N4_pf_{x}"]=gram*pf==s.zeros(3,1)
        checks[f"n{n}_N4_rank2_{x}"]=gram.rank()==2
        # The unweighted sum of brackets vanishes on total singlets.
        checks[f"n{n}_unweighted_bracket_singlet_{x}"]=all(
            not add(*[(F(1),bracket(v,i,j,n)) for j in range(4) if j!=i])
            for i in range(4) for v in B)
        data[str(n)][str(x)]={"null_gram":[[str(z) for z in gram.row(i)] for i in range(3)],
                              "rank":gram.rank(),"pfaffian":list(map(str,pf))}
assert all(checks.values()),checks
out={"all_pass":True,"checks":checks,"data":data,
     "scope":"N4 Gram uses the three pair-contraction singlets (full SO(n) singlet space at n=23,24; n=4 also has an epsilon singlet). One adjoint component suffices on the tested invariant space by covariance.",
     "norm_convention":"J=L/i; common i cancels in positive norms. No numerical rank threshold used."}
Path('data_exports/mqm/mqm_kz_null_checks_results.json').write_text(json.dumps(out,indent=2))
print(json.dumps({"all_pass":True,"count":len(checks),"data":data},indent=2))
