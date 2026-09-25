"""Independent Rayleigh--Ritz audit of the massless AW boundary oscillator.

This uses a sine/cosine basis, not the candidate's finite-difference stencil.
"""
import json
from pathlib import Path
import numpy as np
from scipy.linalg import eigh
from scipy.optimize import brentq
from scipy.special import gamma, hyp1f1, roots_legendre

M = gamma(7/6)/np.sqrt(np.pi)

def gaussian_upper(eps):
    x = brentq(lambda x:M*x*x/(1+x)**(1/3)-eps, 1e-12,
               100*(1+eps), xtol=1e-14)
    return eps/(2*x)+.75*M*(1+x)**(2/3)

def solve(eps, L, nb, nq):
    z,w=roots_legendre(nq)
    z,w=L*z,L*w
    pot=.75*M*hyp1f1(-2/3,.5,-z*z)
    gp=2*M*z*hyp1f1(1/3,1.5,-z*z)
    k=(np.arange(nb)+.5)*np.pi/L
    basis=np.cos(np.outer(z,k))/np.sqrt(L)
    matrix=np.diag(eps*k*k)+basis.T@((w*pot)[:,None]*basis)
    en,vec=eigh(matrix,subset_by_index=[0,0])
    psi=basis@vec[:,0]
    prob=w*psi*psi
    ko=(np.arange(nb)+1)*np.pi/L
    bo=np.sin(np.outer(z,ko))/np.sqrt(L)
    odd=np.diag(eps*ko*ko)+bo.T@((w*pot)[:,None]*bo)
    eo=eigh(odd,subset_by_index=[0,0],eigvals_only=True)[0]
    K=float((vec[:,0]**2)@(k*k))
    e0=float(en[0])
    out={"epsilon":eps,"half_box":L,"basis_size":nb,
         "quadrature_points":nq,"ground":e0,"gap":float(eo-e0),
         "threshold":.75*M,"z_variance":float(prob@(z*z)),
         "Q_variance_over_h":K,"gprime_squared":float(prob@(gp*gp)),
         "normalization_error":abs(float(prob.sum())-1),
         "virial_residual":float(prob@(z*gp)-2*eps*K),
         "gaussian_upper":gaussian_upper(eps),
         "ground_asymptotic_remainder":
             e0-.75*M-np.sqrt(M*eps)+eps/12,
         "gap_asymptotic_remainder":
             float(eo-e0)-2*np.sqrt(M*eps)+eps/3}
    assert out["normalization_error"]<1e-11
    assert out["ground"] < out["gaussian_upper"]
    assert out["z_variance"]*K >= .25-1e-10
    assert abs(out["virial_residual"])<2e-9
    return out

def main():
    rows=[]
    for eps,L in [(1.,10.),(.01,3.),(.001,1.8),(.0001,1.)]:
        case=[solve(eps,L,nb,nq)
              for nb,nq in [(32,256),(48,384),(64,512)]]
        for key in ["ground","gap","z_variance","gprime_squared"]:
            assert abs(case[-1][key]-case[-2][key])<2e-10, (eps,key)
        rows.append(case)
    candidate=json.loads(Path('data_exports/mqm/mqm_boundary_oscillator_results.json').read_text())
    ref=candidate["massless_epsilon1_Richardson"]
    fine=rows[0][-1]
    discrepancies={key:abs(fine[key]-ref[other]["extrapolated"])
                   for key,other in [
                       ("ground","ground_energy"),("gap","gap"),
                       ("z_variance","z_variance"),
                       ("gprime_squared","gprime_squared_expectation")]}
    assert max(discrepancies.values())<2e-9
    output={"method":"Independent finite-box trigonometric Rayleigh--Ritz",
            "M":M,"rows":rows,
            "candidate_Richardson_discrepancies":discrepancies,
            "scope":"One-coordinate full-line oscillator approximations; no matrix bath."}
    (Path(__file__).resolve().parents[1] / 'data_exports/mqm/mqm_boundary_oscillator_audit_results.json').write_text(
        json.dumps(output,indent=2)+"\n")
    print(json.dumps({"epsilon1":fine,
                      "candidate_discrepancies":discrepancies,
                      "epsilon_1e-4":rows[-1][-1]},indent=2))

if __name__=="__main__":
    main()
