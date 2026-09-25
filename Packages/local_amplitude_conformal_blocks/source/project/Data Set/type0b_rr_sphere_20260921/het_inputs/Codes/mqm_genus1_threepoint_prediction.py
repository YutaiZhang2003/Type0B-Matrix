#!/usr/bin/env python3
"""One-loop S -> V^a V^a prediction of the specified coupling-flow unitary.

No worldsheet amplitude enters this calculation. Existing research files are
read only. Run with the repository's scientific Python environment, e.g.
  /Users/sam/miniconda3/bin/python mqm_genus1_threepoint_prediction.py

All kernels below use [a(E),a^dagger(E')]=delta(E-E'). The coupling in
these kernels is gamma=g/sqrt(2*pi), where g is the draft's literal
Fourier-normalized coupling. The result is A/A_tree=1+g^2*relative_g2+O(g^4).
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import mpmath as mp


def kernels(a, b, vector_species=23, *, dps=60):
    """Return real reduced G3G4, G4G3, and G3^3 matrix elements.

Each is divided by K=sqrt(2*E*a*b)/sqrt(1+E^2), with E=a+b.
The labels 34 and 43 specify operator order (rightmost acts first).
All integrals are over physical positive-energy intermediate particles.
"""
    if isinstance(vector_species, bool) or not isinstance(vector_species, int):
        raise TypeError("vector_species must be an integer")
    if vector_species < 1:
        raise ValueError("at least one vector species is required")
    with mp.workdps(dps):
        a, b = mp.mpf(a), mp.mpf(b)
        if not (mp.isfinite(a) and mp.isfinite(b) and a > 0 and b > 0):
            raise ValueError("outgoing energies must be positive and finite")
        e = a + b
        n2 = 1 + e*e

        def p(x, y):
            return 1 + x*x + y*y + x*y

        def integrate(f, upper):
            return mp.quad(f, [0, upper/2, upper])

        def split_weight(x, upper):
            return x*(upper-x)/(1+x*x)

        c34 = sum(integrate(
            lambda x: split_weight(x, upper)*(-1+e*x/(1+(e-x)**2)),
            upper,
        ) for upper in (a, b))

        def c43_integrand(x):
            y = e-x
            singlet = p(x, y)/((1+x*x)*(1+y*y))*(-1-x*y/n2)
            vector = (vector_species/n2 + 1/(1+(x-a)**2)
                      + 1/(1+(x-b)**2))
            return x*y*(singlet+vector)/2

        c43 = integrate(c43_integrand, e)
        singlet_splitting_norm = e/n2*(
            vector_species*e**3/6 + integrate(
                lambda x: x*(e-x)*p(x, e-x)**2/
                ((1+x*x)*(1+(e-x)**2)), e,
            )
        )
        three_particle_path = 2*e*sum(integrate(
            lambda x: split_weight(x, upper)*
            (2+x*(e-x)/(1+(e-x)**2)), upper,
        ) for upper in (a, b))
        c333 = singlet_splitting_norm + three_particle_path
        tree_kernel = mp.sqrt(2*e*a*b)/mp.sqrt(n2)
        relative_gamma2 = -c333/6-1j*(2*c34+c43)/3
        relative_g2 = relative_gamma2/(2*mp.pi)
        loop_gamma3 = tree_kernel*(-(2*c34+c43)/3+1j*c333/6)
        return dict(
            energy=e, outgoing_a=a, outgoing_b=b,
            vector_species=vector_species, tree_kernel=tree_kernel,
            c34=c34, c43=c43, c333=c333,
            singlet_splitting_norm=singlet_splitting_norm,
            three_particle_path=three_particle_path,
            relative_gamma2=relative_gamma2, relative_g2=relative_g2,
            tree_g_coefficient=-1j*tree_kernel/mp.sqrt(2*mp.pi),
            loop_g3_coefficient=loop_gamma3/(2*mp.pi)**mp.mpf('1.5'),
        )


def serialized(result, digits=45):
    def convert(x):
        if isinstance(x, int):
            return x
        if isinstance(x, mp.mpc):
            return {"real": mp.nstr(x.real, digits),
                    "imag": mp.nstr(x.imag, digits)}
        return mp.nstr(x, digits)
    return {k: convert(v) for k, v in result.items()}


def source_hashes():
    root = Path(__file__).resolve().parents[1]
    names = ('docs/mqm/mqm_clock_coupling_isotopy.md',
             'docs/mqm/mqm_clock_continuum_cubic_audit.md',
             'docs/mqm/mqm_clock_continuum_quartic_audit.md',
             ('Codes/' + Path(__file__).name))
    return {name: hashlib.sha256((root/name).read_bytes()).hexdigest()
            for name in names}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--a", default="1/3")
    parser.add_argument("--b", default="2/3")
    parser.add_argument("--dps", type=int, default=60)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = {
        "schema": "mqm-genus1-threepoint-prediction-v1",
        "target": "S(E) -> V^a(a) V^a(b), E=a+b, real positive energies",
        "prescription": "dU/dg=-i U Wick(B_g), U(0)=1; fixed linear leg phases removed",
        "coupling": "gamma=g/sqrt(2*pi)",
        "expansion": "A/A_tree=1+g^2*relative_g2+O(g^4)",
        "working_decimal_digits": args.dps,
        "serialized_decimal_digits": 45,
        "source_sha256": source_hashes(),
        "prediction": serialized(kernels(args.a, args.b, dps=args.dps)),
        "scope": "Candidate continuum S-matrix prediction; no worldsheet data used or matched.",
    }
    rendered = json.dumps(result, indent=2)+"\n"
    if args.output:
        # A prediction artifact must be deliberately regenerated, never
        # silently overwrite an existing frozen target.
        with args.output.open("x") as stream:
            stream.write(rendered)
    else:
        print(rendered, end="")


if __name__ == "__main__":
    main()
