"""Four-fermion bootstrap from tree CAR and exact GS supercharge modes.

No heterotic-worldsheet/screenshot amplitude enters this computation.
This certifies the minimal local/resolvent cubic-spinor completion and its
amplitude; it does not prove absence of higher-derivative/nonlocal contacts
without the parent GS derivative/locality assumptions.
"""
import itertools
import json
from pathlib import Path
import numpy as np
import sympy as s

I=s.I
w,v,k,x,y,z=s.symbols('w v k x y z',real=True)
D=lambda u:1+I*u
H=lambda u:1+2*I*u
E=x+y+z
checks=[]
def ck(name,expr):
    residual=s.simplify(s.cancel(expr))
    assert residual==0,(name,residual)
    checks.append(name)

# The all-frequency CAR at O(chi^2) fixes the coefficients of R(T_F)chi
# and T_F chi in the minimal ansatz, independently of scattering.
p=w-k;r=v+k
source=(w-v-k)*(1+k*k)-k*(w+p)*(v+r)
AR=(H(v)*(w+p)*D(k)-H(w)*(v+r)*D(-k))/2
AL=(H(v)*(w+p)-H(w)*(v+r))*(1+k*k)/2
a,b=s.symbols('a b')
sol=s.solve(s.Poly(s.expand(source+a*AR+b*AL),w,v,k).coeffs(),[a,b])
assert sol=={a:-1,b:0},sol
checks.append('CAR uniquely fixes minimal stress structures a=-1,b=0')
ck('CAR different fermion flavors',source-AR)
ck('CAR equal-flavor stress source',2-(H(w)+H(v))/D(w+v))

# Fierz source in the Cayley tensor channel is -(w-v)/2. CAR determines
# F(w)-F(v)=(w-v)/6; the exact opposite-chart supercharge fixes F(i/2)=0.
F=lambda u:u/6-I/12
ck('Cayley CAR all frequencies',3*(F(w)-F(v))-(w-v)/2)
ck('exact GS supercharge mode removes homogeneous constant',F(I/2))
ck('Cayley coefficient is -i H/12',F(w)+I*H(w)/12)

# Build real 8x8 Spin(7) Clifford matrices using octonion multiplication,
# and derive Omega by the Fierz identity (not by assuming its sign).
triples=[(1,2,3),(1,4,5),(1,7,6),(2,4,6),(2,5,7),(3,4,7),(3,6,5)]
C=np.zeros((8,8,8),dtype=int)
for aa,bb,cc in triples:
    for ii,jj,kk in ((aa,bb,cc),(bb,cc,aa),(cc,aa,bb)):
        C[ii,jj,kk]=1;C[jj,ii,kk]=-1
G=[]
for aa in range(1,8):
    M=np.zeros((8,8),dtype=int);M[aa,0]=1;M[0,aa]=-1
    for jj in range(1,8):
        for kk in range(1,8):M[kk,jj]=C[aa,jj,kk]
    G.append(M)
eye=np.eye(8,dtype=int)
assert all(np.array_equal(G[aa]@G[bb]+G[bb]@G[aa],-2*(aa==bb)*eye)
           for aa in range(7) for bb in range(7))
checks.append('all 49 Spin7 Clifford relations')
Omega=(np.einsum('aij,akl->ijkl',np.array(G),np.array(G))
       -np.einsum('ik,jl->ijkl',eye,eye)+np.einsum('il,jk->ijkl',eye,eye))
assert all(np.array_equal(Omega,-np.swapaxes(Omega,ii,ii+1)) for ii in range(3))
checks.append('Fierz residual is fully antisymmetric Cayley form')

# Standard outgoing ket b_beta^dagger(x)b_gamma^dagger(y)b_delta^dagger(z),
# bra beta_delta(z) beta_gamma(y) beta_beta(x). Coefficients below strip
# lambda^2=gamma^2/2, not gamma^2.
m=y+z
direct=[-H(x)*(y-z)/(2*D(-m)),
         H(x)*(E+y)/(2*D(x+z)),
        -H(x)*(E+z)/(2*D(x+y)),
         I*H(x)/2]
sequential=[m*(z-y)*(E+x)/(1+m*m),-m,m,-m]
target=[-H(E)*(y-z)/(2*D(m)),
         H(E)*(x-z)/(2*D(x+z)),
        -H(E)*(x-y)/(2*D(x+y)),
         I*H(E)/2]
for n in range(4):
    ck('direct plus sequential tensor '+str(n),direct[n]+sequential[n]-target[n])

def parity(perm):
    return (-1)**sum(perm[i]>perm[j] for i in range(len(perm)) for j in range(i+1,len(perm)))
def pair_key(a,b,c,d):
    return tuple(sorted((tuple(sorted((a,b))),tuple(sorted((c,d))))))
P=[pair_key(0,1,2,3),pair_key(0,2,1,3),pair_key(0,3,1,2)]
for perm in itertools.permutations((1,2,3)):
    en=[None,x,y,z]
    xx,yy,zz=[en[j] for j in perm]
    permuted=[-H(E)*(yy-zz)/(2*D(yy+zz)),
               H(E)*(xx-zz)/(2*D(xx+zz)),
              -H(E)*(xx-yy)/(2*D(xx+yy)),I*H(E)/2]
    inds=[0,*perm]
    keys=[pair_key(inds[0],inds[1],inds[2],inds[3]),
          pair_key(inds[0],inds[2],inds[1],inds[3]),
          pair_key(inds[0],inds[3],inds[1],inds[2])]
    for key,coef in zip(keys,permuted[:3]):
        ck('graded exchange '+str(perm)+' '+str(key),coef-parity(perm)*target[P.index(key)])
    ck('graded exchange '+str(perm)+' Cayley',parity(perm)*permuted[3]-parity(perm)*target[3])

# All four coefficients vanish for an incoming exact Goldstino mode E=i/2.
for n,coef in enumerate(target):
    ck('incoming Goldstino Ward tensor '+str(n),coef.subs(z,I/2-x-y))

# Minimal local uniqueness is not arbitrary full functional uniqueness.
# A homogeneous quartic kernel with >=4 derivatives can obey the same
# special-mode zero. Its existence is documented rather than hidden.
k0,k1,k2=s.symbols('k0 k1 k2')
k3=-k0-k1-k2
higher=s.prod(s.Rational(1,2)-I*kk for kk in (k0,k1,k2,k3))
ck('higher-derivative loophole explicitly has supercharge zero',higher.subs(k0,-I/2))

result={
    'status':'passed',
    'scope':'minimal local/resolvent GS completion fixed by all-frequency CAR and exact GS supercharge mode; full finite-kappa or higher-derivative exclusion proof still outstanding',
    'checks':checks,
    'minimal_chi3_map':'delta beta_F(w)|chi3= -lambda^2 H(w) [R(T_F)chi + i Omega chi^3/12]; lambda^2=gamma^2/2 after Fourier factors',
    'amplitude_in_gamma_squared_units':'('+str(-H(E)/4)+') * [(y-z)/D(y+z) P1 -(x-z)/D(x+z) P2 +(x-y)/D(x+y) P3 - i Omega]',
    'delta_normalization':'fermions already have {b(w),b†(wprime)}=delta(w-wprime); no sqrt(E*x*y*z) external factor',
    'direct_in_lambda_squared_units':[str(c) for c in direct],
    'sequential_in_lambda_squared_units':[str(c) for c in sequential],
    'Cayley0123':int(Omega[0,1,2,3]),
    'higher_derivative_loophole':'An additional fully symmetric quartic generator kernel product_j(1/2-i*k_j) vanishes at the supercharge momentum but has four derivatives; excluding it requires the GS derivative/locality/pole assumptions, not CAR plus one special-frequency Ward identity alone.'
}
(Path(__file__).resolve().parents[1]/'data_exports/heterotic_long_string_trees_20260912/four_fermion_results.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps({k:v for k,v in result.items() if k not in ('checks','direct_in_lambda_squared_units','sequential_in_lambda_squared_units')},indent=2))
print('checks passed:',len(checks))
