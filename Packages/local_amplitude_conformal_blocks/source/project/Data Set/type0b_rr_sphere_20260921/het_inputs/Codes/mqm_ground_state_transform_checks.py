"""Bounded checks of exact state engineering; no continuum state claim."""
import json
from pathlib import Path
import numpy as np
import sympy as sp


def h_function(s, d):
    s=np.maximum(s,0)
    p=np.sqrt(2*s)
    r=np.sinh(np.arcsinh(3*np.sqrt(3*d)*p)/3)/np.sqrt(3*d)
    return r*r/2+3*d*r**4


def main():
    x,h=sp.symbols('x h',real=True,positive=True)
    W=x*x/2+x**4/16
    psi=sp.exp(-W/h)
    # Independent coordinate expression for the curved metric ds²=e^(2x) dx².
    lap_psi=sp.exp(-x)*sp.diff(sp.exp(-x)*sp.diff(psi,x),x)
    lap_W=sp.exp(-x)*sp.diff(sp.exp(-x)*sp.diff(W,x),x)
    U=sp.exp(-2*x)*sp.diff(W,x)**2/2-h*lap_W/2
    assert sp.simplify((-h*h*lap_psi/2+U*psi)/psi)==0

    # Local Riccati jet preserves a supplied potential through quartic order.
    z,omega,alpha,beta,gamma=sp.symbols('z omega alpha beta gamma',real=True)
    A=alpha/(3*omega)
    B=(2*beta-9*A*A+gamma*omega*omega)/(8*omega)
    Wjet=omega*z*z/2+A*z**3+B*z**4
    V=omega**2*z*z/2+alpha*z**3+beta*z**4
    residual=sp.series(sp.diff(Wjet,z)**2/(2*(1+gamma*z*z))-V,z,0,5).removeO()
    assert sp.simplify(residual)==0

    # The algebraic scalar derivative f=1-t f³ has Fuss--Catalan coefficients.
    t=sp.symbols('t')
    f=sum((-t)**n*sp.binomial(3*n,n)/(2*n+1) for n in range(11))
    assert sp.series(f-1+t*f**3,t,0,11).removeO()==0

    # A graph analogue of the exact scalar subordinate-Laplacian construction.
    N=37; hb=.3; mass=.7; c=.2
    L=2*np.eye(N)-np.roll(np.eye(N),1,axis=0)-np.roll(np.eye(N),-1,axis=0)
    lam,O=np.linalg.eigh(L)
    K=(O*h_function(hb*hb*lam/(2*mass),c/(mass*mass)))@O.T
    q=2*np.pi*np.arange(N)/N
    ps=np.exp(-.2*(1-np.cos(q))/hb)
    potential=-(K@ps)/ps
    H=K+np.diag(potential)
    eig=np.linalg.eigvalsh(H)
    rng=np.random.default_rng(20260908)
    v=rng.normal(size=N)+1j*rng.normal(size=N)
    direct=np.vdot(ps*v,H@(ps*v)).real
    jump=-K.copy();np.fill_diagonal(jump,0)
    weighted=.5*np.sum(jump*(ps[:,None]*ps[None,:])*abs(v[:,None]-v[None,:])**2)
    assert np.max(K-np.diag(np.diag(K)))<1e-14
    assert np.linalg.norm(H@ps)<1e-13
    assert eig[0]>-1e-13 and eig[1]>0
    assert abs(direct-weighted)<1e-12

    # Positive kinetic energy alone is not enough for the same conclusion.
    Kbad=np.ones((2,2))
    positive_state=np.ones(2)
    Hbad=Kbad+np.diag(-(Kbad@positive_state)/positive_state)
    assert np.linalg.eigvalsh(Kbad)[0]>=0
    assert np.allclose(Hbad@positive_state,0)
    assert np.linalg.eigvalsh(Hbad)[0]<0

    # A local W deformation inside scalar functional calculus changes quartics.
    p,k,d=sp.symbols('p k d')
    s=(p*p+k*k*z*z)/2
    engineered=s-4*d*s*s
    desired=p*p/2-d*p**4+k*k*z*z/2
    mismatch=sp.expand(engineered-desired)
    assert sp.expand(mismatch+2*d*k*k*z*z*p*p+d*k**4*z**4)==0

    out={
       "curved_LB_state_identity":"exact symbolic pass",
       "Riccati_quartic_jet_identity":"exact symbolic pass",
       "Fuss_Catalan_coefficients_checked":11,
       "scalar_subordinate_graph":{
          "sites":N,"ground_energy":float(eig[0]),"first_gap":float(eig[1]),
          "state_residual":float(np.linalg.norm(H@ps)),
          "largest_offdiagonal_kinetic_entry":float(np.max(K-np.diag(np.diag(K)))),
          "ground_transform_form_identity_error":float(abs(direct-weighted))},
       "generic_positive_kinetic_counterexample_eigenvalues":np.linalg.eigvalsh(Hbad).tolist(),
       "functional_calculus_classical_quartic_mismatch":str(mismatch),
       "scope":"Finite and scalar benchmarks; no full coupled AW ground state or MQM continuum limit."
    }
    (Path(__file__).resolve().parents[1] / 'data_exports/mqm/mqm_ground_state_transform_results.json').write_text(json.dumps(out,indent=2)+"\n")
    print(json.dumps(out,indent=2))


if __name__=="__main__":
    main()
