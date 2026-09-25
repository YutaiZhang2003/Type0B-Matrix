"""Schur inertia at the aligned flat-model reference; finite samples only."""
import json
from pathlib import Path
import numpy as np
from scipy.optimize import brentq
from scipy.linalg import cho_factor,cho_solve


def one_case(N):
    lam=float(N);labels=np.arange(N+2)+1.
    tau=np.array([brentq(lambda t:lam*(np.sinh(t)*np.cosh(t)-t)-ci,
                         1e-8,4.) for ci in labels])
    x=np.sqrt(2*lam)*np.cosh(tau)
    v=np.sqrt(2*lam)*np.sinh(tau)
    ell=tau[2:]-tau[:-2]
    mass=(x[2:]-x[:-2])**2/4
    J=np.zeros((N,N))
    for i in range(N):
        if i>0:J[i,i-1]=v[i]/ell[i]
        if i<N-1:J[i,i+1]=-v[i+2]/ell[i]
    G=np.eye(N)+J.T@(mass[:,None]*J)
    direct=J[0]@cho_solve(cho_factor(G),J[0])
    B=np.sqrt(mass)[:,None]*J
    resolvent=np.eye(N)+B@B.T
    r11=cho_solve(cho_factor(resolvent),np.eye(N)[0])[0]
    ratio=1/(1-r11)
    assert abs(ratio-1/(mass[0]*direct))<1e-9
    assert ratio>=1
    return {"N":N,"Lambda":lam,"m1":float(mass[0]),
            "MW_over_m1":float(ratio),"resolvent11":float(r11)}


def main():
    rows=[one_case(N) for N in [16,32,64,128,256,17,33,65,129,257]]
    (Path(__file__).resolve().parents[1] / 'data_exports/mqm/mqm_reference_boundary_anisotropy_results.json').write_text(json.dumps(rows,indent=2)+"\n")
    print(json.dumps(rows,indent=2))


if __name__=="__main__":
    main()
