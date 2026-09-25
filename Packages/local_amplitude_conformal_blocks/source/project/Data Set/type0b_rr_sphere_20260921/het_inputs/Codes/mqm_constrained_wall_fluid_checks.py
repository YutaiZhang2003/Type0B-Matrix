#!/usr/bin/env python3
"""Cubic identities and prescribed quartic integrals; not a duality test."""
import json
from pathlib import Path

import mpmath as mp
import sympy as sp


def symbolic_checks():
    a, b, t = sp.symbols("a b t", positive=True)
    w = [a+b, a, b]
    k = [-w[0], w[1], w[2]]
    s = [sp.sin(x*t) for x in w]
    c = [sp.cos(x*t) for x in w]
    scalar = sp.prod(w[i]*c[i] for i in range(3))
    scalar += sum(w[i]*c[i]*sp.prod(sp.I*k[j]*s[j] for j in range(3) if j != i)
                  for i in range(3))
    kernels = [sp.trigsimp(sp.expand_trig(scalar/sp.prod(w)))]
    for i in range(3):
        j, l = [r for r in range(3) if r != i]
        mixed = w[i]*c[i]*(-k[j]*k[l]*c[j]*c[l]+w[j]*w[l]*s[j]*s[l])
        mixed += sp.I*k[i]*s[i]*(-sp.I*k[j]*w[l]*c[j]*s[l]
                                           -sp.I*k[l]*w[j]*c[l]*s[j])
        kernels.append(sp.trigsimp(sp.expand_trig(mixed/sp.prod(w))))
    assert kernels == [1, -1, 1, 1]
    numerator = 1-w[1]*w[2]+w[0]*w[1]+w[0]*w[2]
    assert sp.expand(numerator-(1+a*a+b*b+a*b)) == 0

    x, q = sp.symbols("x q", positive=True)
    profile = 1/sp.sinh(x)**2
    h = sp.coth(x)*sp.cos(q*x)
    source = 2*q*profile*sp.sin(q*x)-sp.diff(profile,x)*sp.cos(q*x)
    assert sp.simplify(sp.expand_trig(sp.diff(h,x,2)+q*q*h-source)) == 0

    # Legendre transform with two representative director components.
    g, ex, et, x1, x2, t1, t2 = sp.symbols("g ex et x1 x2 t1 t2")
    l3 = -g*(ex**3/6+ex*et**2/2+ex*(t1*t1+t2*t2+x1*x1+x2*x2)/2
              +et*(t1*x1+t2*x2))
    l4 = g*g*(ex*ex*et*et/2+ex*ex*(t1*t1+t2*t2)/2
               +2*et*ex*(t1*x1+t2*x2)+et*et*(x1*x1+x2*x2)/2)
    h4 = sum(sp.diff(l3,z)**2 for z in [et,t1,t2])/2-l4
    assert sp.simplify(h4-g*g*(t1*x1+t2*x2)**2/2) == 0
    p = sp.symbols("p", positive=True)
    ji = -sp.pi/2*(1+p*p)*sp.tanh(sp.pi*p/2)
    subtracted = ji**2/((1+p*p)*(-1-p*p))+sp.pi**2/4
    assert sp.simplify(subtracted-sp.pi**2/(4*sp.cosh(sp.pi*p/2)**2)) == 0
    assert sp.simplify((2/sp.pi)*(sp.pi**2/4)*(2/sp.pi)) == 1
    return {"pointwise_cubic_kernels": [str(z) for z in kernels],
            "constrained_SSS_numerator": str(sp.expand(numerator)),
            "source_identity": True, "parent_quartic_Legendre_identity": True,
            "exchange_at_q_i": 1}


def exchange(q):
    q = mp.mpf(q)
    bq = q*q/(1+q*q)
    def j(p):
        if abs(p-q) < mp.mpf('1e-35'):
            return -q
        return (q*q-p*p)*mp.pi/2*mp.sinh(mp.pi*p)/(mp.cosh(mp.pi*p)-mp.cosh(mp.pi*q))
    def integrand(p):
        if abs(p-q) < mp.mpf('1e-18'):
            h = mp.mpf('1e-12')
            deriv = (j(q+h)**2/(1+(q+h)**2)-j(q-h)**2/(1+(q-h)**2))/(2*h)
            return -deriv/(2*q)+mp.pi**2/4
        return (j(p)**2/(1+p*p)-bq)/(q*q-p*p)+mp.pi**2/4
    cutoff = mp.mpf(30)
    # Exponentially accurate analytic tail; includes the pole subtraction.
    tail = mp.pi**2/4*(1+q*q)*(mp.pi/2-mp.atan(cutoff))
    tail += bq/(2*q)*mp.log((cutoff+q)/(cutoff-q))
    real = 2/mp.pi*(mp.quad(integrand,[0,q/2,q,2*q+1,15,cutoff])+tail)
    expected = mp.pi*q*mp.coth(mp.pi*q)+1/(1+q*q)
    assert abs(real-expected) < mp.mpf('1e-28')
    return {"q":str(q), "real_integral":mp.nstr(real,32),
            "proposed_formula_error":mp.nstr(real-expected,8),
            "imaginary_part":mp.nstr(-q/(1+q*q),20)}


def phase_constraint_contact(q):
    q = mp.mpf(q)
    ultraviolet = mp.pi**2*q*q/4
    def xcoth(z):
        return 2/mp.pi if abs(z)<mp.mpf('1e-30') else z*mp.coth(mp.pi*z/2)
    def integrand(p):
        f = mp.pi/4*(xcoth(p+q)-xcoth(p-q))
        return p*p/(1+p*p)*f*f-ultraviolet
    cutoff=mp.mpf(30)
    val=2/mp.pi*(mp.quad(integrand,[0,q,2*q+1,15,cutoff])
                -ultraviolet*(mp.pi/2-mp.atan(cutoff)))
    return {"q":str(q), "minimal_subtraction_contact":mp.nstr(val,28),
            "negative_thermal_term":mp.nstr(-mp.pi*q*mp.coth(mp.pi*q),28)}


def main():
    mp.mp.dps=50
    result={"symbolic":symbolic_checks(),
            "exchange_numerical":[exchange(q) for q in ['0.2','0.5','1','2','3','5']],
            "linear_phase_constraint_contact":[phase_constraint_contact(q) for q in ['0.5','1','2']],
            "all_checks_passed":True,
            "scope":"Cubic kernels, Legendre/source identities and I(i)=1 are exact. Numerical exchange checks support the independent analytic proof in mqm_wall_dirac_audit.md for the stated finite-part scheme. No microscopic sea, absolute amplitude normalization or duality is asserted."}
    path=(Path(__file__).resolve().parents[1] / 'data_exports/mqm/mqm_constrained_wall_fluid_results.json')
    path.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))


if __name__=='__main__':
    main()
