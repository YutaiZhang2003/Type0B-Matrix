#!/usr/bin/env python3
"""Exact soft-singlet limits of the stipulated wall-on string targets."""
import json
from pathlib import Path
import sympy as sp


def main():
    a,b,e = sp.symbols("a b epsilon", positive=True)
    energy=a+b
    w0=energy+e
    n=lambda w: sp.sqrt(1+w*w)
    w=w0*e*a*b
    f3svv=energy*a*b/n(energy)
    f4ssvv=w/(n(w0)*n(e))*(1+sp.I*w0)*(1+sp.I*(2*w0-e))/(1+sp.I*energy)
    f3sss=energy*a*b*(1+a*a+b*b+a*b)/(n(energy)*n(a)*n(b))
    pair_sum=1/(1+sp.I*(e+a))+1/(1+sp.I*(e+b))+1/(1+sp.I*energy)
    f4ssss=w/(n(w0)*n(e)*n(a)*n(b))*(w*pair_sum+(1+2*sp.I*w0)*(1+(w0*w0+e*e+a*a+b*b)/2))
    ratios=[sp.simplify((f4/e).subs(e,0)/f3) for f4,f3 in [(f4ssvv,f3svv),(f4ssss,f3sss)]]
    expected=1+2*sp.I*energy
    assert all(sp.simplify(r-expected)==0 for r in ratios)
    mu,E=sp.symbols("mu E",positive=True)
    scaling=mu**(-1-2*sp.I*E)
    assert sp.simplify(-mu*sp.diff(scaling,mu)/scaling-(1+2*sp.I*E))==0
    # A moving reflection-symmetric basis carries a derivative connection.
    symmetric=sp.Function("A_symmetric")(mu)
    wave=mu**(-2*sp.I*E)*symmetric
    connected=mu**(-2*sp.I*E)*(-mu*sp.diff(symmetric,mu)+2*sp.I*E*symmetric)
    assert sp.simplify(-mu*sp.diff(wave,mu)-connected)==0
    # Exact reflection prefactor, followed by its weak-end threshold limit.
    # This is not a proof of an integrated wall-insertion Ward identity.
    p,phi=sp.symbols("P phi", real=True)
    reflection=-mu**(-2*sp.I*p)*(sp.gamma(1+sp.I*p)/sp.gamma(1-sp.I*p))**2
    inverse_sqrt=-sp.I*mu**(sp.I*p)*sp.gamma(1-sp.I*p)/sp.gamma(1+sp.I*p)
    assert sp.simplify(inverse_sqrt**2*reflection-1)==0
    incoming=sp.exp(phi)*(sp.exp(sp.I*p*phi)+reflection*sp.exp(-sp.I*p*phi))
    normalized=inverse_sqrt*incoming
    assert normalized.subs(p,0)==0
    puncture=sp.simplify(sp.diff(normalized,p).subs(p,0))
    assert sp.simplify(puncture-2*(phi+sp.log(mu)+2*sp.EulerGamma)*sp.exp(phi))==0
    # Write Gamma(nu)/Gamma(1-nu) without the apparent Gamma(0) at b=1.
    critical_b=sp.symbols("critical_b",positive=True)
    nu=(critical_b**2+1)/2
    renorm_gamma=(1-nu)*sp.gamma(nu)/sp.gamma(2-nu)
    screening_p=(critical_b-1/critical_b)/(2*sp.I)
    assert sp.simplify(sp.diff(renorm_gamma,critical_b).subs(critical_b,1)+1)==0
    screening_ratio=sp.simplify(sp.diff(screening_p,critical_b).subs(critical_b,1)/sp.diff(renorm_gamma,critical_b).subs(critical_b,1))
    assert screening_ratio==sp.I
    results={"all_passed":True,"SSVV_soft_S_ratio":str(ratios[0]),"SSSS_soft_S_ratio":str(ratios[1]),
             "ratio_convention":"F4/(epsilon F3), outgoing singlet epsilon, hard incoming E=a+b",
             "wall_derivative_identity":"-mu_L d/dmu_L [mu_L^(-1-2iE)] = (1+2iE) mu_L^(-1-2iE)",
             "symmetric_basis_connection":"-mu_L d/dmu_L + 2iE at fixed hard E",
             "weak_end_threshold_puncture":str(puncture),
             "screening_P_over_gamma_limit":str(screening_ratio),
             "scope":"Exact target and reflection identities; the puncture expression uses only the weak-end asymptotic. Integrated soft-operator and MQM background identifications still require derivation. No wall parameter is set to zero."}
    (Path(__file__).resolve().parents[1] / 'data_exports/mqm/mqm_soft_singlet_results.json').write_text(json.dumps(results,indent=2)+"\n")
    print(json.dumps(results,indent=2))


if __name__ == "__main__":
    main()
