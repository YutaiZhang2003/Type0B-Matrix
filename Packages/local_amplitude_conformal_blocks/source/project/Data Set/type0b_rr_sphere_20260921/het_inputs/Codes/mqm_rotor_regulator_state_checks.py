"""Finite rotor-regulator diagnostics; no quantum sea or duality claim.

Tests the exact centered-constraint null direction, its scalar stiffness,
a nearest-gap replacement, and a high-precision classical critical point.
"""
import json
from pathlib import Path
import numpy as np
import mpmath as mp
import sympy as s
from scipy.linalg import eigh
from scipy.optimize import brentq


def wall_matrices(N, mu=20., a=1., cstar=1.):
    count=np.arange(N+2)+cstar
    tau=np.array([brentq(lambda t:mu/a*(np.sinh(t)*np.cosh(t)-t)-ci,1e-9,8) for ci in count])
    x=np.sqrt(2*mu)*np.cosh(tau)
    profile=np.sqrt(x*x-2*mu)/a
    d=x[2:]-x[:-2]; ell=tau[2:]-tau[:-2]; rho=2/d
    D=np.zeros((N,N)); J=np.zeros((N,N))
    for i in range(N):
        if i>0: D[i,i-1]=-1; J[i,i-1]=np.sqrt(a)*profile[i]/ell[i]
        if i<N-1: D[i,i+1]=1; J[i,i+1]=-np.sqrt(a)*profile[i+2]/ell[i]
    E=np.diff(np.eye(N),axis=0)
    B=np.diff(np.vstack([np.zeros((1,N)),np.eye(N),np.zeros((1,N))]),axis=0)
    L=E.T@np.diag(1/np.diff(x[1:-1])**2)@E
    Hrot=a*J.T@L@J
    Hcenter=D.T@np.diag(4*a*a/d**4)@D-np.eye(N)+Hrot
    Hnearest=B.T@np.diag(a*a/np.diff(x)**4)@B-np.eye(N)+Hrot
    G=np.eye(N)+J.T@np.diag(1/(a*rho*rho))@J
    z=np.array([1/profile[i+1] if i%2==0 else 0 for i in range(N)])
    assert np.linalg.norm(J@z)<1e-9
    centered=eigh(Hcenter,G,eigvals_only=True)
    nearest=eigh(Hnearest,G,eigvals_only=True)
    assert nearest[0]>0
    return {'N':N,'centered_low_eigenvalues':centered[:4].tolist(),
            'nearest_low_eigenvalues':nearest[:4].tolist(),
            'null_constraint_error':float(np.linalg.norm(J@z)),
            'centered_parity_Rayleigh':float(z@Hcenter@z/(z@G@z)),
            'scope':'Hessian at reference profile, which is not an exact critical point'}


def critical_point(N=15):
    mp.mp.dps=60
    a=mp.mpf(1); mu=mp.mpf(20); beta=mp.mpf(50)
    c=[mp.mpf(i+1) for i in range(N+2)]
    ts=[mp.findroot(lambda t:mu/a*(mp.sinh(t)*mp.cosh(t)-t)-ci,
                    (mp.mpf('.1'),mp.mpf(2)),maxsteps=100) for ci in c]
    ref=[mp.sqrt(2*mu)*mp.cosh(t) for t in ts]
    def value_gradient(xx):
        X=[ref[0]]+list(xx)+[ref[-1]]
        v=[mp.sqrt(t*t-2*mu) for t in X]
        tau=[mp.acosh(t/mp.sqrt(2*mu)) for t in X]
        eta=[mp.sqrt(a)*(c[i]-(X[i]*v[i]-2*mu*tau[i])/(2*a)) for i in range(N+2)]
        eta[0]=eta[-1]=mp.mpf(0)
        ell=[tau[i+2]-tau[i] for i in range(N)]
        F=[(eta[i+2]-eta[i])/ell[i] for i in range(N)]
        J=mp.matrix(N,N)
        for i in range(N):
            if i>0:J[i,i-1]=(v[i]/mp.sqrt(a)+F[i]/v[i])/ell[i]
            if i<N-1:J[i,i+1]=-(v[i+2]/mp.sqrt(a)+F[i]/v[i+2])/ell[i]
        theta=[mp.asin(f/beta) for f in F]
        r=[X[i+1]-X[i] for i in range(N+1)]
        grad=mp.matrix([a*a/3*(r[i+1]**-3-r[i]**-3)-xx[i] for i in range(N)])
        val=a*a/6*sum(t**-2 for t in r)-sum(t*t for t in xx)/2
        for i in range(N-1):
            dt=theta[i+1]-theta[i]; ri=r[i+1]
            b=a*beta*mp.sin(dt)/ri**2
            for j in range(N):
                grad[j]+=b*(J[i+1,j]/mp.sqrt(1-F[i+1]**2/beta**2)-J[i,j]/mp.sqrt(1-F[i]**2/beta**2))
            vpair=2*a*beta**2*mp.sin(dt/2)**2/ri**2
            val+=vpair
            grad[i]+=2*vpair/ri; grad[i+1]-=2*vpair/ri
        rho=[2/(X[i+2]-X[i]) for i in range(N)]
        G=mp.eye(N)+J.T*mp.diag([1/(a*rho[i]**2*(1-F[i]**2/beta**2)) for i in range(N)])*J
        return val,grad,F,G
    x=mp.matrix(ref[1:-1])
    # The derivative is taken independently of the analytic Hessian-at-reference.
    for step in range(5):
        val,g,F,G=value_gradient(x)
        H=mp.matrix(N,N)
        for j in range(N):
            def col(t):
                y=x.copy();y[j]=t
                return value_gradient(y)[1]
            for i in range(N): H[i,j]=mp.diff(lambda t:col(t)[i],x[j])
        if max(abs(t) for t in g)<mp.mpf('1e-40'):break
        x-=mp.lu_solve(H,g)
    val,g,F,G=value_gradient(x)
    assert max(abs(t) for t in g)<mp.mpf('1e-35')
    assert max(abs(H[i,j]-H[j,i]) for i in range(N) for j in range(N))<mp.mpf('1e-40')
    hv=np.array(H.tolist(),dtype=float);gv=np.array(G.tolist(),dtype=float)
    eig=eigh(hv,gv,eigvals_only=True)
    assert eig[0]>1
    return {'N':N,'mu':'20','a':'1','beta':'50','iterations':step+1,
            'gradient_max':mp.nstr(max(abs(t) for t in g),12),
            'maximum_displacement':mp.nstr(max(abs(x[i]-ref[i+1]) for i in range(N)),20),
            'maximum_abs_F':mp.nstr(max(abs(t) for t in F),20),
            'position_low_eigenvalues':eig[:4].tolist(),
            'scope':'Classical local position minimum with aligned residual rotors; no quantum state claim'}


def main():
    theta,L=s.symbols('theta Lambda',real=True)
    centered=L**2*s.sin(theta)**2
    nearest=4*L**2*s.sin(theta/2)**2
    difference=(nearest-centered-4*L**2*s.sin(theta/2)**4).subs(theta,2*theta)
    assert s.trigsimp(s.expand_trig(difference))==0
    out={'scope':'Finite-regulator classical diagnostics only',
         'exact_uniform_stiffness_increment':'4 Lambda^2 sin(theta/2)^4',
         'wall_samples':[wall_matrices(N) for N in [15,31,63,127]],
         'critical_point':critical_point()}
    path=(Path(__file__).resolve().parents[1] / 'data_exports/mqm/mqm_rotor_regulator_state_results.json')
    path.write_text(json.dumps(out,indent=2)+'\n')
    print(json.dumps(out,indent=2))

if __name__=='__main__':main()
