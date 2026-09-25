#!/usr/bin/env python3
"""Independent exact and numerical checks of the prescribed wall exchange.

The finite parts tested here are explicitly chosen continuum prescriptions;
they are not a microscopic matrix endpoint derivation.
"""
import json
from pathlib import Path
import mpmath as mp
import sympy as s


def exact_checks():
    q, t = s.symbols('q t', positive=True, real=True)
    c, sh = s.cosh(s.pi*q), s.sinh(s.pi*q)
    k0 = -q/sh
    k1 = -s.pi/(2*(1+c))+q/((1+q*q)*sh)
    first = 2*c*k0+sh/s.pi*s.diff(k0, q)
    second = s.pi/2+2*c*k1+sh/s.pi*s.diff(k1, q)
    first_expected = -q*s.coth(s.pi*q)-1/s.pi
    second_expected = q*s.coth(s.pi*q)/(1+q*q)+(1-q*q)/(s.pi*(1+q*q)**2)
    assert s.simplify((first-first_expected).rewrite(s.exp)) == 0
    assert s.simplify((second-second_expected).rewrite(s.exp)) == 0
    real_result = -s.pi*first/2+s.pi*(1+q*q)*second/2
    assert s.simplify((real_result-s.pi*q*s.coth(s.pi*q)-1/(1+q*q)).rewrite(s.exp)) == 0
    profile = s.coth(t)*s.cos(q*t)
    source = s.diff(profile, t, 2)+q*q*profile
    expected_source = 2/s.sinh(t)**2*(q*s.sin(q*t)+s.coth(t)*s.cos(q*t))
    assert s.simplify(source-expected_source) == 0
    p = s.symbols('p', positive=True, real=True)
    j_at_i = -(1+p*p)*s.pi/2*s.tanh(s.pi*p/2)
    integrand_at_i = j_at_i**2/((1+p*p)*(-1-p*p))+s.pi**2/4
    assert s.simplify(integrand_at_i-s.pi**2/(4*s.cosh(s.pi*p/2)**2)) == 0
    # d/dp tanh(pi*p/2) directly integrates the remaining sech² term.
    assert s.simplify((2/s.pi)*integrand_at_i-s.diff(s.tanh(s.pi*p/2), p)) == 0
    phase_profile_i = s.pi/4*((p+s.I)*s.coth(s.pi*(p+s.I)/2)
                              -(p-s.I)*s.coth(s.pi*(p-s.I)/2))
    assert s.simplify((phase_profile_i-s.I*s.pi*s.tanh(s.pi*p/2)/2).rewrite(s.exp)) == 0
    return {"K0": str(k0), "K1": str(k1),
            "exchange": "pi*q*coth(pi*q)+1/(1+i*q)",
            "exchange_at_q_i": "1", "phase_contact_at_q_i": "1+pi**2/6",
            "symbolic_status": "pass"}


def real_exchange(q):
    pole_numerator = q*q/(1+q*q)
    def smooth(p):
        if abs(p-q) < mp.mpf('1e-17'):
            return -(1+mp.pi*q*mp.coth(mp.pi*q))/(2*(1+q*q))+q*q/(1+q*q)**2+mp.pi**2/4
        j = (q*q-p*p)*mp.pi/2*mp.sinh(mp.pi*p)/(mp.cosh(mp.pi*p)-mp.cosh(mp.pi*q))
        return (j*j/(1+p*p)-pole_numerator)/(q*q-p*p)+mp.pi**2/4
    bulk = 2/mp.pi*mp.quad(smooth, [0, q/2, q, q+10, 50])
    # Beyond 50 the discarded hyperbolic corrections are exponentially
    # below the requested precision; integrate the full algebraic tail.
    tail = 2/mp.pi*mp.quad(
        lambda p: mp.pi**2/4*(1+q*q)/(1+p*p)-pole_numerator/(q*q-p*p),
        [50, mp.inf])
    return bulk+tail


def main():
    mp.mp.dps = 45
    results = {"exact": exact_checks(), "numerical": []}
    sech_integral = mp.quad(lambda p: mp.sech(mp.pi*p/2)**2/(1+p*p), [0, 1, mp.inf])
    assert abs(sech_integral-mp.pi/6) < mp.mpf('1e-40')
    phase_contact_i = mp.pi/2*mp.quad(
        lambda p: mp.sech(mp.pi*p/2)**2+mp.tanh(mp.pi*p/2)**2/(1+p*p),
        [0, 1, mp.inf])
    assert abs(phase_contact_i-(1+mp.pi**2/6)) < mp.mpf('1e-40')
    results['phase_contact_i'] = mp.nstr(phase_contact_i, 40)
    for value in ['0.2', '0.7', '1.3', '2.4']:
        q = mp.mpf(value)
        measured = real_exchange(q)
        expected = mp.pi*q*mp.coth(mp.pi*q)+1/(1+q*q)
        error = abs(measured-expected)
        assert error < mp.mpf('1e-28')
        results['numerical'].append({"q": value,
            "real_integral": mp.nstr(measured, 35),
            "closed_form": mp.nstr(expected, 35),
            "absolute_error": mp.nstr(error, 5),
            "imaginary_part": mp.nstr(-q/(1+q*q), 30)})
    output = (Path(__file__).resolve().parents[1] / 'data_exports/mqm/mqm_wall_dirac_quartic_results.json')
    output.write_text(json.dumps(results, indent=2)+'\n')
    print(json.dumps(results, indent=2))


if __name__ == '__main__':
    main()
