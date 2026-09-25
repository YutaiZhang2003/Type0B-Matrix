#!/usr/bin/env python3
"""Independent contact expansion and direct-space cubic check for SSVV."""
from __future__ import annotations

import itertools
import json
from pathlib import Path

import mpmath as mp
import sympy as s

import mqm_wall_mixed_quartic_checks as peer


def coefficient(expression, variables):
    return s.Poly(s.expand(expression), *variables).coeff_monomial(s.prod(variables))


def quartic_checks():
    lam, f, etax, etat, qt, qx = s.symbols(
        "lam f etax etat qt qx", real=True)
    exact = (lam*qt-f*lam**2*etat*qx/(1+f*lam*etax))**2\
        /(2*(1+f*lam*etax))-(1+f*lam*etax)*lam**2*qx**2/2
    fourth = s.series(exact, lam, 0, 5).removeO().coeff(lam, 4)
    expected = f**2*(etax**2*qt**2/2+2*etax*etat*qt*qx+etat**2*qx**2/2)
    assert s.expand(fourth-expected) == 0
    e = s.symbols("e0:4")
    c = s.symbols("c0:4")
    z = s.symbols("s0:4")
    w = s.symbols("w0:4", positive=True)
    tx = w[0]*c[0]*e[0]+w[1]*c[1]*e[1]
    tt = -s.I*w[0]*z[0]*e[0]+s.I*w[1]*z[1]*e[1]
    vt = s.I*w[2]*c[2]*e[2]+s.I*w[3]*c[3]*e[3]
    vx = -w[2]*z[2]*e[2]-w[3]*z[3]*e[3]
    contact = coefficient(tx**2*vt**2/2+2*tx*tt*vt*vx+tt**2*vx**2/2, e)
    a, b, cc, tau = s.symbols("a b cc tau", positive=True)
    energies = [a+b+cc, a, b, cc]
    trig = dict(zip(w, energies))
    trig.update({c[i]:s.cos(energies[i]*tau) for i in range(4)})
    trig.update({z[i]:s.sin(energies[i]*tau) for i in range(4)})
    q, r, d = b+cc, 2*a+b+cc, b-cc
    kernel = -s.Rational(3,2)+s.cos(2*q*tau)/2\
        -s.cos((r+d)*tau)/2-s.cos((r-d)*tau)/2
    lhs = contact.subs(trig)/s.prod(energies)
    assert s.simplify(s.expand((lhs-kernel).rewrite(s.exp))) == 0
    # All-Q deformation: S wave W_i=w_i cos(w_i tau), so compare
    # directly with the vector channel without silently dropping W.
    wt = -s.I*w[0]**2*c[0]*e[0]+s.I*w[1]**2*c[1]*e[1]
    wx = -w[0]**2*z[0]*e[0]-w[1]**2*z[1]*e[1]
    bulk = coefficient((wt*wx+vt*vx)**2, e)
    target = -2*s.prod(energies)*energies[0]*energies[1]*s.sin(q*tau)**2
    assert s.simplify(s.expand((bulk.subs(trig)-target).rewrite(s.exp))) == 0
    boundary = coefficient((wt**2+vt**2)**2, e).subs({x:1 for x in c})
    assert s.expand(boundary+8*s.prod(w)*w[0]*w[1]) == 0
    return {
        "parent_L4": str(expected),
        "parent_SSVV_kernel_over_W": str(kernel),
        "added_bulk_kernel_over_W": "-2 w0 w1 sin(q tau)^2",
        "added_boundary_coefficient_over_W": "-8 w0 w1",
    }


def independent_cubic(types, kappas, energies):
    """Expand the whole cubic Lagrangian as a polynomial in three waves."""
    z = s.symbols("z0:3")
    tau = s.symbols("tau", real=True)
    eta = sum(z[i]*s.sin(kappas[i]*tau) for i in range(3) if types[i]=="S")
    etat = sum(z[i]*s.I*energies[i]*s.sin(kappas[i]*tau)
               for i in range(3) if types[i]=="S")
    etax = s.diff(eta, tau)
    spin_t, spin_x = {}, {}
    for i in range(3):
        flavor = "W" if types[i]=="S" else types[i]
        shape = (kappas[i] if types[i]=="S" else 1)*s.cos(kappas[i]*tau)
        spin_t[flavor] = spin_t.get(flavor,0)+z[i]*s.I*energies[i]*shape
        spin_x[flavor] = spin_x.get(flavor,0)+z[i]*s.diff(shape,tau)
    L3 = -(etax**3+3*etax*etat**2)/6
    for flavor in spin_t:
        T, X = spin_t[flavor], spin_x[flavor]
        L3 -= etat*T*X+etax*(T*T+X*X)/2
    poly = coefficient(L3, z)
    return poly, tau


def direct_vertex_check():
    mp.mp.dps = 45
    p = s.symbols("p", real=True)
    w0,w1,w2,w3 = map(s.Rational,["1.7",".4",".6",".7"])
    q = w2+w3
    cases = [
        (["S","S","S"],[w0,w1,p],[-w0,w1,q]),
        (["a","a","S"],[w2,w3,p],[w2,w3,-q]),
        (["S","a","a"],[w0,w2,p],[-w0,w2,w0-w2]),
        (["S","a","a"],[w1,w3,p],[w1,w3,-w1-w3]),
    ]
    records = []
    for types,kappas,energies in cases:
        raw,tau = independent_cubic(types,kappas,energies)
        # Compare the full pointwise off-shell cubic with the peer's
        # separate exponential-generator implementation.
        generated = sum(coeff*s.exp(s.I*sum(signs[i]*kappas[i]
                          for i in range(3))*tau)
                        for signs,coeff in peer.vertex_terms(types,kappas,energies))
        assert s.simplify(s.expand((raw-generated).rewrite(s.exp))) == 0
        evaluate,_ = peer.make_vertex(types,kappas[:2],energies)
        for pp in [s.Rational(3,10),s.Rational(9,10),s.Rational(23,10)]:
            expr = s.simplify(raw.subs(p,pp))
            fun = s.lambdify(tau,expr,"mpmath")
            g0 = mp.mpf(str(expr.subs(tau,0)))
            g2 = mp.mpf(str(s.diff(expr,tau,2).subs(tau,0)/2))
            g4 = mp.mpf(str(s.diff(expr,tau,4).subs(tau,0)/24))
            eps = mp.mpf("0.0001")
            # Direct tau-space Hadamard integral, with the first two
            # omitted finite Taylor terms restored. Error is O(eps^5).
            integral = mp.quad(lambda t:fun(t)/mp.sinh(t)**2,
                               [eps,mp.mpf(".1"),1,4,12,mp.inf])
            finite = integral-g0/eps+(g2-g0/3)*eps\
                +(g4-g2/3+g0/15)*eps**3/3
            transform = evaluate(mp.mpf(str(pp)))
            err = abs(finite-transform)
            assert err < mp.mpf("2e-17"), (types,pp,err)
            records.append({"types":types,"p":str(pp),
                            "direct_tau_FP":mp.nstr(finite,25),
                            "cosine_transform":mp.nstr(transform,25),
                            "absolute_error":mp.nstr(err,5)})
    return records


def mixed_exchange_remainder():
    """A separate direct spectral integration of the new mixed kernel."""
    mp.mp.dps = 50
    records=[]
    for qtext in [".2",".7","1.2","1.3","2.1"]:
        q=mp.mpf(qtext)
        Nq=q**3/(1+q*q)
        def smooth(p):
            if abs(p-q)<mp.mpf("1e-20"):
                return -q/(1+q*q)**2
            denominator=mp.cosh(mp.pi*p)-mp.cosh(mp.pi*q)
            j=(q*q-p*p)*mp.pi/2*mp.sinh(mp.pi*p)/denominator
            k=(q*q-p*p)*mp.pi/2*mp.sinh(mp.pi*q)/denominator
            numerator=p*j*k/(1+p*p)
            return (numerator-Nq)/(q*q-p*p)
        cutoff=mp.mpf(60)
        integral=mp.quad(smooth,[0,q/2,q,q+1,q+8,cutoff])
        integral+=Nq/(2*q)*mp.log((cutoff+q)/(cutoff-q))
        direct=2*integral/mp.pi-1j*q*q/(1+q*q)
        analytic=q/2+q/(1+1j*q)+(1+q*q)/(8j)*(
            mp.polygamma(1,(1+1j*q)/2)-mp.polygamma(1,(1-1j*q)/2))
        assert abs(direct-analytic)<mp.mpf("1e-35")
        H=mp.pi*q*mp.coth(mp.pi*q)
        delta=mp.mpf(2)/3-(4+q*q)*H/3-q*analytic-1/(1+1j*q)-1j*q
        assert abs(mp.im(delta))<mp.mpf("1e-45")
        records.append({"q":qtext,"direct_J":mp.nstr(direct,35),
                        "trigamma_J":mp.nstr(analytic,35),
                        "error":mp.nstr(abs(direct-analytic),5),
                        "mixed_remainder":mp.nstr(delta,35)})
    # At q=i the mixed integrand vanishes pointwise because sinh(pi i)=0.
    p=s.symbols("p", nonnegative=True)
    k_i=((-1-p*p)*s.pi/2*s.sinh(s.pi*s.I)/
         (s.cosh(s.pi*p)-s.cosh(s.pi*s.I)))
    assert s.simplify(k_i)==0
    # The constant term after cancellation of the two simple poles.
    delta=s.symbols("delta")
    qi=s.I+delta
    Hseries=s.I/delta+1+s.I*s.pi**2*delta/3
    Jseries=delta/2  # follows also from the trigamma Laurent expansion
    remainder=s.Rational(2,3)-(4+qi*qi)*Hseries/3\
        -qi*Jseries-1/(1+s.I*qi)-s.I*qi
    assert s.limit(remainder,delta,0)==s.Rational(4,3)
    return {"numerical_checks":records,
            "q_to_zero_limit":"-5/3",
            "continued_q_i_value":"4/3",
            "q_i_mixed_integrand":"identically zero",
            "scope":"Fixed continuum finite parts; not the uncomputed finite-N scattering."}


def main():
    result = {
        "scope": "Independent contact and off-shell cubic checks; no quantum endpoint limit.",
        "quartic":quartic_checks(),
        "direct_position_space_vertex_checks":direct_vertex_check(),
        "mixed_exchange_remainder":mixed_exchange_remainder(),
    }
    (Path(__file__).resolve().parents[1] / 'data_exports/mqm/mqm_wall_mixed_quartic_audit_results.json').write_text(
        json.dumps(result,indent=2)+"\n")
    print(json.dumps(result,indent=2))


if __name__=="__main__":
    main()
