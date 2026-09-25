"""Independent differential-operator checks for the singlet-state sea audit.

The script checks exact finite algebra. It constructs no thermodynamic sea.
"""
from pathlib import Path
from itertools import combinations_with_replacement
import json
import sympy as s

checks=[]
def eq(name,a,b=0):
    diff=a-b
    entries=list(diff) if isinstance(diff,s.MatrixBase) else [diff]
    assert all(s.simplify(v)==0 for v in entries),(name,diff)
    checks.append(name)

x=s.symbols('x0:3',real=True)
w=s.symbols('omega',positive=True)
lam=s.symbols('lambda',real=True)
sx=s.Matrix([[0,1],[1,0]])
sy=s.Matrix([[0,-s.I],[s.I,0]])
sz=s.diag(1,-1)
T=[-lam*m for m in (sx,sy,sz)]
V=sum((x[i]*T[i] for i in range(3)),s.zeros(2))
psi=s.Matrix([s.Function('f')(*x),s.Function('g')(*x)])
def H(v):
    return -sum((v.diff(y,2) for y in x),s.zeros(2,1))/2-w*w*sum(y*y for y in x)*v/2+V*v
def A(v,i):return (-s.I*v.diff(x[i])+w*x[i]*v)/s.sqrt(2*w)
def B(v,i):return (-s.I*v.diff(x[i])-w*x[i]*v)/s.sqrt(2*w)
def Ah(v,i):return A(v,i)-T[i]*v/(w*s.sqrt(2*w))
def comm(a,b):return a*b-b*a
for i in range(3):
    eq(f'Heisenberg A {i}',s.I*(H(A(psi,i))-A(H(psi),i)),w*A(psi,i)-T[i]*psi/s.sqrt(2*w))
    eq(f'Heisenberg B {i}',s.I*(H(B(psi,i))-B(H(psi),i)),-w*B(psi,i)-T[i]*psi/s.sqrt(2*w))
    tdot=s.I*comm(V,T[i])
    eq(f'Heisenberg heavy generator {i}',s.I*(H(T[i]*psi)-T[i]*H(psi)),tdot*psi)
    eq(f'shifted dilation defect {i}',s.I*(H(Ah(psi,i))-Ah(H(psi),i)),w*Ah(psi,i)-tdot*psi/(w*s.sqrt(2*w)))
    assert tdot!=s.zeros(2)
for i,j in ((0,1),(1,2),(2,0)):
    eq(f'noncommuting shifted coordinate {i}{j}',Ah(Ah(psi,j),i)-Ah(Ah(psi,i),j),comm(T[i],T[j])*psi/(2*w**3))

# The standard SO(N) tensor Casimir is >=N-1 for every nonzero dominant
# integral highest weight. This finite scan checks signs/normalization in
# the analytic minimum proof, including the last signed D-type weight.
counts={}
for N in range(3,13):
    r=N//2
    vals=[]
    for rev in combinations_with_replacement(range(4),r):
        ell=tuple(reversed(rev))
        if not any(ell):continue
        possibilities=[ell]
        if N%2==0 and ell[-1]:possibilities.append(ell[:-1]+(-ell[-1],))
        for mu in possibilities:
            c=sum(mu[i]*(mu[i]+N-2*(i+1)) for i in range(r))
            assert c>=N-1,(N,mu,c)
            vals.append(c)
    assert min(vals)==N-1
    checks.append(f'nontrivial SO({N}) tensor Casimir minimum')
    counts[N]=len(vals)

# The complete flavor-singlet color list and its exact Casimir minimum.
r,l=s.symbols('r l',integer=True,nonnegative=True)
for rr in range(2,15):
    cl=[]
    for ll in range(0,rr+1,2):
        kappa=[12]*(rr-ll)+[11]*ll
        c=sum(kk*(kk+2*rr-2*(i+1)) for i,kk in enumerate(kappa))
        eq(f'singlet color Casimir r={rr},l={ll}',s.Integer(c),11*rr*(rr+10)+(rr-ll)*(rr+ll+22))
        cl.append(c)
    eq(f'minimal singlet color Casimir r={rr}',s.Integer(min(cl)),11*rr*(rr+10)+(2*rr+21 if rr%2 else 0))
eq('SO4 singlet lower Casimir radial factor two',s.Integer(264),2*11*12)
eq('SO4 singlet upper Casimir radial factor two',s.Integer(312),2*12*13)

# Quartic stabilizer really is the displayed one-body Cartan regulator:
# every skew block contributes twice to each trace.
z,L=s.symbols('z Lambda',positive=True)
J=s.Matrix([[0,1],[-1,0]])
X=z*J
eq('trace quartic normalization',w*w*s.trace((X.T*X)**2)/(8*L*L),w*w*z**4/(4*L*L))
eq('Cartan regulator minimum location',s.diff(-w*w*z*z/2+w*w*z**4/(4*L*L),z).subs(z,L))
eq('Cartan regulator well value',(-w*w*z*z/2+w*w*z**4/(4*L*L)).subs(z,L),-w*w*L*L/4)

# Complete scalar/vector radial towers at N4 and lambda=0.
q=s.symbols('q')
singlet_towers=(q**(s.Rational(25,2))+q**(s.Rational(27,2)))*q**s.Rational(3,2)/(1-q*q)**2
vector_per_component=q**s.Rational(25,2)*(q**s.Rational(3,2)+q**s.Rational(5,2))/(1-q*q)**2
eq('N4 full singlet oscillator character',s.factor(singlet_towers),q**14*(1+q)/(1-q*q)**2)
eq('N4 vector versus singlet full character',s.factor(vector_per_component),s.factor(singlet_towers))

# A polynomial example of the proposed probe family is real symmetric and
# color covariant, including under a disconnected reflection.
X3=s.Matrix([[0,x[2],-x[1]],[-x[2],0,x[0]],[x[1],-x[0],0]])
c1,c2=s.symbols('c1 c2',real=True)
def probe_f(M):
    Z=-M*M
    return s.eye(3)+c1*Z+c2*Z*Z
O=s.Matrix([[s.Rational(3,5),-s.Rational(4,5),0],[s.Rational(4,5),s.Rational(3,5),0],[0,0,-1]])
eq('probe matrix orthogonal color covariance',probe_f(O*X3*O.T),O*probe_f(X3)*O.T)
eq('probe matrix real symmetry',probe_f(X3).T,probe_f(X3))
eq('density probe color invariance',s.trace(probe_f(O*X3*O.T)),s.trace(probe_f(X3)))

out={'passed':True,'checks':checks,'dominant_weights_scanned':counts,
     'scope':'Exact differential identities and Casimir/regulator normalization only; no claim of nonintegrability or a sea limit.'}
(Path(__file__).resolve().parents[1] / 'data_exports/mqm/mqm_majorana_singlet_sea_results.json').write_text(json.dumps(out,indent=2)+'\n')
print(json.dumps(out,indent=2))
