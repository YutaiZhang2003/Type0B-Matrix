#!/usr/bin/env python3
"""Exact algebra of a constrained current phase space, not an MQM S-matrix."""
import json
from pathlib import Path
import sympy as sp


def main():
    d = sp.symbols("D", nonzero=True)
    c = d * (1-d*d)
    reduced = sp.factor(d - d*d/c*(-d*d))
    assert sp.simplify(reduced-d/(1-d*d)) == 0
    assert sp.simplify(reduced * (1-d*d) - d) == 0
    x = sp.symbols("x", real=True)
    u, v = sp.Function("u")(x), sp.Function("v")(x)
    up = sp.diff(u, x)
    density = u**3/3 + u*up**2 + u*v**2
    du = sp.diff(density, u) - sp.diff(sp.diff(density, up), x)
    expected = u**2-up**2-2*u*sp.diff(u, x, 2)+v**2
    assert sp.simplify(du-expected) == 0
    defect = sp.diff(2*u*up, x) - sp.diff(u*u+up*up+v*v, x, 2)
    assert sp.simplify(defect+sp.diff(up*up+v*v, x, 2)) == 0
    assert sp.factor(c * d/(1-d*d) - d*d) == 0

    k = sp.symbols("k0:3", real=True)
    z = sp.symbols("z0:3")
    mode = sum(z)
    derivative = sum(sp.I*k[i]*z[i] for i in range(3))
    cubic = sp.expand(mode**3/3+mode*derivative**2)
    vertex = cubic.coeff(z[0],1).coeff(z[1],1).coeff(z[2],1)
    kernel = 1-sum(k[i]*k[j] for i in range(3) for j in range(i+1,3))
    assert sp.factor(vertex-2*kernel) == 0
    assert sp.factor((kernel-1-sum(ki*ki for ki in k)/2).subs(k[0],-k[1]-k[2])) == 0

    mu, a = sp.symbols("mu a", real=True)
    shifted = (mu+a*u)**3/3 + (mu+a*u)*(a*up)**2 + (mu+a*u)*(a*v)**2
    quadratic = sp.expand(shifted).coeff(a,2)
    assert sp.simplify(quadratic-mu*(u*u+up*up+v*v)) == 0
    q = sp.symbols("q")
    bracket = sp.I*q/(1+q*q)
    residues = [sp.residue(bracket,q,pole) for pole in (sp.I,-sp.I)]
    assert residues == [sp.I/2,sp.I/2]
    results = {
        "all_passed": True,
        "reduced_bracket": str(reduced),
        "zero_mode": "D=0 is a Casimir leaf restriction, excluded from the inverse",
        "cubic_functional_derivative": str(expected),
        "parent_constraint_defect": "-d_x^2[(u_x)^2+v^2]",
        "constraint_multiplier": "d_x (1-d_x^2)^-1[(u_x)^2+v^2]",
        "symmetric_cubic_vertex": str(vertex),
        "SVV_vertex": "2 delta_ab; common cubic coefficient",
        "constant_background_quadratic": "2 mu H2; not a derived Fermi-sea coupling",
        "continued_bracket_residues": {"+i": str(residues[0]), "-i": str(residues[1])},
        "scope": "Reduced positive current phase space and cubic flavor/momentum ratio only. Canonical unit-delta oscillators supply sqrt(omega) per current leg. No finite-N matrix origin, wall state, or scattering amplitude is derived."
    }
    (Path(__file__).resolve().parents[1] / 'data_exports/mqm/mqm_dirac_current_results.json').write_text(json.dumps(results,indent=2)+"\n")
    print(json.dumps(results,indent=2))


if __name__ == "__main__":
    main()
