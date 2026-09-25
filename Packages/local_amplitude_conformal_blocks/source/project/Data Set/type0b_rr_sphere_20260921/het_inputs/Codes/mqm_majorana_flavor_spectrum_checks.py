"""Exact free-oscillator representation tests of the canonical Majorana MQM.

The fermionic orthogonal complement rule is sourced in Neergard
arXiv:2005.13843, Theorem 2 and equation 49. Character checks compare it
with the direct Clifford-product character at smaller flavor ranks.
No stable-oscillator test here is a scattering or large-rank no-go.
"""
from fractions import Fraction as F
from functools import lru_cache
from itertools import product
from math import comb
from pathlib import Path
import json
import sympy as S

checks=[]
def ck(name,condition):
    assert condition,name
    checks.append(name)

def dimD(w):
    r=len(w); z=[F(w[i])+r-i-1 for i in range(r)]
    rho=[F(r-i-1) for i in range(r)]
    ans=F(1)
    for i in range(r):
        for j in range(i+1,r): ans*= (z[i]**2-z[j]**2)/(rho[i]**2-rho[j]**2)
    assert ans.denominator==1
    return ans.numerator

def dimB(w):
    r=len(w);rho=[F(2*(r-i)-1,2) for i in range(r)]
    z=[F(w[i])+rho[i] for i in range(r)]
    ans=F(1)
    for i in range(r):
        ans*=z[i]/rho[i]
        for j in range(i+1,r): ans*= (z[i]**2-z[j]**2)/(rho[i]**2-rho[j]**2)
    assert ans.denominator==1
    return ans.numerator

def branchDToB(w):
    choices=[]
    for i in range(len(w)-1):
        lo=abs(w[-1]) if i==len(w)-2 else w[i+1]
        hi=w[i]
        count=int(hi-lo)
        choices.append([lo+k for k in range(count+1)])
    return tuple(product(*choices))

def complement4(a,b,s):
    return tuple(2-int(a>=s-i)-int(b>=s-i) for i in range(s))

@lru_cache(None)
def charDplus(w,ys):
    """Average of the two last-sign characters, equal to either if w[-1]=0."""
    r=len(w)
    top=S.Matrix([[y**(S.Rational(wi)+r-i-1)+y**(-S.Rational(wi)-r+i+1)
                   for y in ys] for i,wi in enumerate(w)])
    bot=S.Matrix([[y**(r-i-1)+y**(-r+i+1) for y in ys] for i in range(r)])
    return S.cancel(top.det()/bot.det())

def su2(twice_j,u):
    return sum(u**m for m in range(-twice_j,twice_j+1,2))

def char4plus(a,b,u,v):
    return (su2(a+b,u)*su2(a-b,v)+su2(a-b,u)*su2(a+b,v))/2

# Direct Fock dimensions and nontrivial torus characters test the full
# complement sum, not just the particular representations of interest.
for s in range(2,13):
    total=0
    for a in range(s+1):
        for b in range(a+1):
            w=complement4(a,b,s)
            dc=(a+b+1)*(a-b+1)
            total+=2*dc*dimD(w)
            ck(f'D{s} to B{s-1} dimension at color ({a},{b})',
               sum(dimB(mu) for mu in branchDToB(w))==dimD(w))
    ck(f'SO4 x SO{2*s} full Fock dimension',total==2**(4*s))

for s in [2,3,4]:
    u,v=S.Rational(2),S.Rational(3)
    xs=(u*v,u/v)
    ys=tuple(S.Integer(p) for p in [5,7,11,13][:s])
    direct=S.prod(x+1/x+y+1/y for x in xs for y in ys)
    dual=sum(2*char4plus(a,b,u,v)*charDplus(complement4(a,b,s),ys)
             for a in range(s+1) for b in range(a+1))
    ck(f'rank4 direct Clifford torus character for {2*s} flavors',S.cancel(direct-dual)==0)

# Full connected-group oscillator decomposition at N=4. Sym^n of spin one
# contains spins n,n-2,... once each; left and right oscillators are independent.
s=12
muS=(0,)*11
muV=(1,)+(0,)*10
levels=[]
for m in range(13):
    counts={muS:0,muV:0}
    contributors={muS:[],muV:[]}
    for nl in range(m+1):
        nr=m-nl
        for jl in range(nl,-1,-2):
            for jr in range(nr,-1,-2):
                a,b=jl+jr,jl-jr
                if a>s: continue
                w=complement4(a,abs(b),s)
                multiplicity=2 if b==0 else 1
                for mu in branchDToB(w):
                    if mu in counts:
                        counts[mu]+=multiplicity
                        contributors[mu].append([nl,nr,jl,jr,list(w),multiplicity])
    ck(f'rank4 O-character halves at level {m}',all(v%2==0 for v in counts.values()))
    levels.append({'boson_degree':m,'energy_over_Omega':m+3,
                   'singlet_multiplicity_SO4':counts[muS],
                   'vector_multiplicity_SO4':counts[muV],
                   'singlet_multiplicity_O4_det':counts[muS]//2,
                   'vector_multiplicity_O4_det':counts[muV]//2,
                   'contributors':{'singlet':contributors[muS],'vector':contributors[muV]}})
ck('no rank4 singlet/vector below degree 11',
   all(x['singlet_multiplicity_SO4']==x['vector_multiplicity_SO4']==0 for x in levels[:11]))
ck('first rank4 singlet/vector multiplicities',
   levels[11]['singlet_multiplicity_SO4']==levels[11]['vector_multiplicity_SO4']==2)
ck('Spin24 symmetric traceless rank2 branches as 275+23+1',
   dimD((2,)+(0,)*11)==299 and dimB((2,)+(0,)*10)==275)

# N=3 independently checks the source reviewer's fundamental-spinor threshold.
fundS=(F(1,2),)*11
rank3=[]
dim3=0
for j in range(13):
    w=(F(3,2),)*(12-j)+(F(1,2),)*j
    dim3+=2*(2*j+1)*dimD(w)
    ck(f'rank3 flavor branching dimension j={j}',sum(dimB(mu) for mu in branchDToB(w))==dimD(w))
    if fundS in branchDToB(w): rank3.append(j)
ck('rank3 full Clifford dimension',dim3==2**36)
ck('fundamental Spin23 spinor needs color j=11 or12 at rank3',rank3==[11,12])

ground=[]
for N in range(1,7):
    D=dimB((F(N,2),)*11)
    ground.append({'N':N,'energy_over_Omega':str(F(N*(N-1),4)),
                   'highest_weight_Spin23':[str(F(N,2))]*11,
                   'ground_dimension_SO':2*D,'ground_dimension_O_det':D,
                   'O_det_parity':(-1)**N})
ck('rank1 ground dimension',ground[0]['ground_dimension_SO']==2**12)
ck('rank2 ground middle exterior dimension',ground[1]['ground_dimension_SO']==comb(24,12))
ck('rank2 Spin23 fiber dimension',ground[1]['ground_dimension_O_det']==comb(23,11))

out={'passed':True,'check_count':len(checks),'checks':checks,'ground_levels':ground,
     'rank4_levels':levels,'rank3_fundamental_spinor_first_degree':11,
     'scope':'Stable lambda=0 oscillator and finite-rank representation statements only; no continuum/scattering no-go.'}
(Path(__file__).resolve().parents[1] / 'data_exports/mqm/mqm_majorana_flavor_spectrum_results.json').write_text(json.dumps(out,indent=2)+'\n')
print(json.dumps({'passed':True,'check_count':len(checks),'ground_levels':ground,
                  'rank4_first':levels[11],'rank3_spinor_threshold':rank3},indent=2))
