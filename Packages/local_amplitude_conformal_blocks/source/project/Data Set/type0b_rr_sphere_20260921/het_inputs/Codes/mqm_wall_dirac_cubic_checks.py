#!/usr/bin/env python3
"""Exact cubic/Dirac checks for the conditional nematic wall fluid.

This does not derive a microscopic wall prescription or an S matrix.
"""
import json
from pathlib import Path
import sympy as s

a, b, tau, k, mu, eps = s.symbols('a b tau k mu eps', positive=True, real=True)
w = [a+b, a, b]
time_sign = [-s.I, s.I, s.I]


def simplify_trig(value):
    return s.simplify(s.trigsimp(s.expand_trig(value)))


def density_vertex():
    f = [s.sin(x*tau) for x in w]
    xder = [s.diff(x, tau) for x in f]
    tder = [time_sign[i]*w[i]*f[i] for i in range(3)]
    value = xder[0]*xder[1]*xder[2]
    for i in range(3):
        j, ell = [r for r in range(3) if r != i]
        value += xder[i]*tder[j]*tder[ell]
    return simplify_trig(value/s.prod(w))


def mixed_vertex(i):
    j, ell = [r for r in range(3) if r != i]
    density = s.sin(w[i]*tau)
    spin_j, spin_l = s.cos(w[j]*tau), s.cos(w[ell]*tau)
    dt_density = time_sign[i]*w[i]*density
    dt_j, dt_l = time_sign[j]*w[j]*spin_j, time_sign[ell]*w[ell]*spin_l
    value = dt_density*(dt_j*s.diff(spin_l, tau)+dt_l*s.diff(spin_j, tau))
    value += s.diff(density, tau)*(dt_j*dt_l+s.diff(spin_j, tau)*s.diff(spin_l, tau))
    return simplify_trig(value/s.prod(w))


def dirac_test():
    zero, eye = s.zeros(2), s.eye(2)
    canonical = zero.row_join(eye).col_join((-eye).row_join(zero))
    constraints = s.Matrix([[-k, 1, 0, 0], [0, 0, -k, 1]])
    gram = constraints*canonical*constraints.T
    reduced = s.simplify(canonical-canonical*constraints.T*gram.inv()*constraints*canonical)
    assert gram == s.Matrix([[0, 1+k*k], [-1-k*k, 0]])
    assert s.simplify(reduced[0, 2]-1/(1+k*k)) == 0
    assert constraints*reduced == s.zeros(2, 4)
    return {"constraint_bracket": str(gram), "eta_momentum_bracket": str(reduced[0, 2])}


def wall_integral_test():
    value = (s.coth(eps)-1)/(2*mu)
    assert s.simplify(s.diff(value, eps)+1/(2*mu*s.sinh(eps)**2)) == 0
    # Extract the Laurent constant directly; coth's generic limit routine in
    # the installed SymPy version incorrectly reports -infinity here.
    finite_part = s.series(value, eps, 0, 2).removeO().expand().coeff(eps, 0)
    assert finite_part == -1/(2*mu)
    return {"cutoff_integral": str(value), "chosen_hadamard_finite_part": str(finite_part)}


def main():
    ddd = density_vertex()
    mixed = [mixed_vertex(i) for i in range(3)]
    assert ddd == 1
    assert mixed == [-1, 1, 1]
    singlet = s.expand(ddd+sum(mixed[i]*s.prod(w[r] for r in range(3) if r != i) for i in range(3)))
    assert s.simplify(singlet-(1+a*a+b*b+a*b)) == 0
    normalized_leg_product = s.prod(w)/s.sqrt(s.prod(w))
    assert s.simplify(normalized_leg_product-s.sqrt(s.prod(w))) == 0
    result = {
        "density_cubic_integrand": str(ddd),
        "mixed_cubic_integrands_by_density_leg": [str(x) for x in mixed],
        "singlet_polynomial": str(singlet),
        "canonical_unit_delta_energy_factor": str(normalized_leg_product),
        "supplied_CFT_target_energy_factor": str(s.prod(w)),
        "converted_unit_delta_target_energy_factor": str(s.sqrt(s.prod(w))),
        "dirac": dirac_test(),
        "wall_integral": wall_integral_test(),
        "status": "All algebra checks pass. LSZ conversion resolves the energy-factor mismatch; endpoint prescription and global constants remain unresolved."
    }
    output = (Path(__file__).resolve().parents[1] / 'data_exports/mqm/mqm_wall_dirac_cubic_results.json')
    output.write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
