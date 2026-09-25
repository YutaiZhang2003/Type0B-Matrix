"""Exact radial shift identities and the N2 source boundary spectrum.

These tests use the known reducing radial sectors and independently derive
the regular boundary data from the two-site KZ matrices. They do not fit
heterotic amplitudes or infer a full MQM charge algebra.
"""
from pathlib import Path
import json
import sympy as s
import scipy.optimize as so
import numpy as np

passed=[]
def check(name,x):
    ok=all(s.simplify(z)==0 for z in x) if isinstance(x,s.MatrixBase) else s.simplify(x)==0
    assert ok,(name,x)
    passed.append(name)

r,nu,w=s.symbols('r nu w',positive=True)
f=s.Function('f')(r)
def h(u,v=nu,frequency=w):
    return -s.diff(u,r,2)/2+(v*v-s.Rational(1,4))*u/(2*r*r)+frequency**2*r*r*u/2
def shift(u,v=nu,sign=1,frequency=w):
    return (s.diff(u,r)-(v+s.Rational(1,2))*u/r+sign*frequency*r*u)/s.sqrt(2)
for sign in (-1,1):
    Df=shift(f,sign=sign)
    check(f'radial first-order intertwining sign{sign}',
          h(Df,nu+1)-shift(h(f),sign=sign)+sign*w*Df)
    check(f'inverted radial intertwining sign{sign}',
          (h(Df,nu+1)-shift(h(f),sign=sign)+sign*w*Df).subs(w,s.I*w))

N=s.symbols('N',positive=True,integer=True)
At=N*(N-1)/2
As=N*N/2
check('source minus tensor radial index',As-At-N/2)
for nn in (2,4,6):
    v=s.Rational(nn*(nn-1),2)-1
    K=nn//2
    for sign in (-1,1):
        state=r**(v+s.Rational(1,2))*s.exp(-w*r*r/2)*s.assoc_laguerre(K+1,v,w*r*r)
        image=state
        for j in range(K):image=s.simplify(shift(image,v+j,sign))
        E=w*(v+1+2*(K+1)-sign*K)
        check(f'N{nn} order{K} excited radial image sign{sign}',h(image,v+K)-E*image)
        assert image!=0

n,x,y=s.symbols('n x y',positive=True)
a=1/(n-1)
O12=s.diag(-(n-1),-1)
O1S=s.Matrix([[0,n-1],[1,-(n-2)]])/2
O2S=s.Matrix([[0,-(n-1)],[-1,-(n-2)]])/2
Bpair=s.eye(2)+a*O12
B1=s.eye(2)/2+a*O1S
B2=s.eye(2)/2+a*O2S
Ax=Bpair/(x-y)+B1/x
Ay=Bpair/(y-x)+B2/y
metric=s.diag(1,n-1)
m=s.Matrix([x+y,x-y])
check('full-coordinate source horizontal x',m.diff(x)-Ax*m)
check('full-coordinate source horizontal y',m.diff(y)-Ay*m)
v=s.Matrix([1,1])
P=v*(v.T*metric)/n
check('origin spinor projector idempotent',P*P-P)
check('origin singular connection annihilates surviving branch',B2*P)
check('origin regular connection gives Robin coefficient',P*Bpair*P-(n-2)*P/n)
check('ground origin Robin condition',P*m.diff(y).subs(y,0)+(n-2)*P*m.subs(y,0)/(n*x))
Ptrace=s.diag(1,0)
check('pair regular normal connection has zero surviving compression',Ptrace*(B2-B1)*Ptrace)
check('pair ground trace normal derivative',Ptrace*(m.diff(y)-m.diff(x)))

# Angular endpoint matrices, determinant, and exact lowest solution.
theta,lam=s.symbols('theta lam',real=True)
L=s.pi/4
c=(n-2)/n
angular=s.Matrix([s.cos(theta)+s.sin(theta),s.sqrt(n-1)*(s.cos(theta)-s.sin(theta))])
vflat=s.Matrix([1,s.sqrt(n-1)])
Pflat=vflat*vflat.T/n
check('angular degree-one equation',angular.diff(theta,2)+angular)
check('angular origin Dirichlet complement',(s.eye(2)-Pflat)*angular.subs(theta,0))
check('angular origin Robin component',Pflat*(angular.diff(theta)+c*angular).subs(theta,0))
check('angular pair Dirichlet component',angular[1].subs(theta,L))
check('angular pair Neumann component',angular[0].diff(theta).subs(theta,L))
F=lam*(s.tan(lam*L)-(n-1)/s.tan(lam*L))+n-2
check('angular spectral equation has lambda1',F.subs(lam,1))
for j in range(1,12,2):
    wanted=(n-2)*(1-j) if j%4==1 else (n-2)*(1+j)
    check(f'integer angular frequency{j}',s.trigsimp(F.subs(lam,j))-wanted)

# The determinant form avoids confusing tangent poles with roots.
def determinant(z,nn=23):
    return z*(np.sin(z*np.pi/4)**2-(nn-1)*np.cos(z*np.pi/4)**2)+(nn-2)*np.sin(z*np.pi/4)*np.cos(z*np.pi/4)
roots=[]
for j in range(8):
    left=2*j+1e-8
    right=2*(j+1)-1e-8
    # Exactly one root in each open tangent/cotangent interval is proved
    # in the memo; numerical bracketing only locates these explicit roots.
    grid=np.linspace(left,right,1001)
    intervals=[(a,b) for a,b in zip(grid[:-1],grid[1:]) if determinant(a)*determinant(b)<0]
    for aa,bb in intervals:
        root=so.brentq(determinant,aa,bb,xtol=1e-13)
        roots.append(root)
        assert abs(determinant(root))<1e-9

result={'exact_checks':len(passed),'checks':passed,'N2_n23_first_angular_roots':roots,
        'radial_shift':'delta E = -sign * K * Omega for K=N/2 repeated equal-sign steps',
        'scope':'Restricted radial intertwiners and independently derived N2 boundary data; no full heterotic charge realization.'}
(Path(__file__).resolve().parents[1] / 'data_exports/mqm/mqm_spinor_radial_intertwiner_results.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result,indent=2))
