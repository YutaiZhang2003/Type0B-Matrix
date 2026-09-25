"""Independent sine-Ritz cutoff energies and exact crossover algebra."""
from pathlib import Path
import json
import numpy as np
from scipy.special import sici
from scipy.linalg import eigh
import sympy as s

exact=[]
xi=s.symbols('xi',real=True)
L=s.symbols('L',positive=True)
m=lambda ll:s.Rational(ll*ll,2)-s.sqrt(s.Rational(ll*ll,4)+xi*xi)
F=-s.Rational(11,2)+s.sqrt(xi*xi+36)-s.sqrt(xi*xi+s.Rational(1,4))
assert s.simplify(m(1)-m(12)+66-F)==0
exact.append('fixed-xi crossover coefficient')
assert F.subs(xi,0)==0
exact.append('zero-coupling crossover')
assert s.limit(F,xi,s.oo)==-s.Rational(11,2)
exact.append('large-xi crossover limit, not interchange of limits')
assert s.expand(s.diff(F,xi)/xi-(1/s.sqrt(xi*xi+36)-1/s.sqrt(xi*xi+s.Rational(1,4))))==0
exact.append('strict negative derivative for positive xi')
r,k=s.symbols('r k',positive=True)
wm=L**2/(2*r*r)-s.sqrt(L**2/(4*r**4)+k*k*r*r)
derivative=-L**2+(L**2/s.Integer(2)-k*k*r**6)/s.sqrt(L**2/s.Integer(4)+k*k*r**6)
assert s.simplify(r**3*s.diff(wm,r)-derivative)==0
exact.append('monotone local matrix minimum derivative')

def matrices(n,cut):
    j=np.arange(1,n+1,dtype=float)
    minus=j[:,None]-j[None,:]
    plus=j[:,None]+j[None,:]
    def first(a):
        ans=np.full_like(a,0.5)
        mask=a!=0
        ans[mask]=(np.cos(np.pi*a[mask])-1)/(np.pi*a[mask])**2
        return ans
    def second(a):
        ans=np.full_like(a,1/3)
        mask=a!=0
        ans[mask]=2*np.cos(np.pi*a[mask])/(np.pi*a[mask])**2
        return ans
    R=cut*(first(minus)-first(plus))
    R2=cut*cut*(second(minus)-second(plus))
    Inv=np.pi/(cut*cut)*(plus*sici(np.pi*plus)[0]-minus*sici(np.pi*minus)[0])
    base=np.diag((np.pi*j/cut)**2/2)-R2/2
    return base,R,Inv
def ground(a):return float(eigh(a,subset_by_index=[0,0],check_finite=False,driver='evr')[0][0])
results=[]
for cut,reference in ((4.,-.250526500381530),(16.,-.022473451884913)):
    seq=[]
    for n in (64,128,224,352):
        base,R,inv=matrices(n,cut)
        eps0=ground(base)
        eps11=ground(base+66*inv)
        def coupled(LL):
            lo=base+LL*(LL-1)*inv/2
            hi=base+LL*(LL+1)*inv/2
            return ground(np.block([[lo,-np.sqrt(2)*.7*R],[-np.sqrt(2)*.7*R,hi]]))
        e1,e12=coupled(1),coupled(12)
        gap=e1+eps11-e12-eps0
        seq.append({'sine_basis_per_channel':n,'ES':e12+eps0,'EV':e1+eps11,'gap':gap})
    discrepancy=seq[-1]['gap']-reference
    assert abs(discrepancy)<2e-8,(cut,seq,discrepancy)
    results.append({'cutoff':cut,'lambda':.7,'omega':1,'sequence':seq,
                    'independent_FD_reference':reference,'last_minus_FD':discrepancy})

out={'passed':True,'exact_checks':exact,'sine_Ritz_cases':results,
     'method':'Dirichlet sine Ritz, exact r, r² and r^-2 matrix integrals; no finite-difference code imported.',
     'scope':'Numerical convergence evidence plus exact local crossover algebra; no large-rank sea conclusion.'}
(Path(__file__).resolve().parents[1] / 'data_exports/mqm/mqm_majorana_singlet_integration_results.json').write_text(json.dumps(out,indent=2)+'\n')
print(json.dumps(out,indent=2))
