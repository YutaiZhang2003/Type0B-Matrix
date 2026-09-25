"""BRY external normalization, fixed by three-point factorization.

References: arXiv:2201.05621 (2.6), (3.4), (3.8)-(3.9), (4.2), (4.14).
The matrix-model prediction is defined separately from the assembly.
"""
from __future__ import annotations

import math
import sys

from frozen import ROOT

sys.path.insert(0, str(ROOT / 'Code/c_Recursion'))
from super_liouville_structure_constants import (
    n_ns, n_r, ns_structure_constant, ns_tilde_structure_constant,
    rr_ns_structure_constants,
)


def external_factor(momenta, sectors, precision=50):
    nr = sectors.count('R')
    if nr not in (2, 4) or len(momenta) != 4 or len(sectors) != 4:
        raise ValueError('four external fields with two or four Ramond legs required')
    numerator = math.prod((n_r if s == 'R' else n_ns)(p, precision)
                          for p, s in zip(momenta, sectors))
    # At the NS primary pole: C_R^+=(C_even+C_odd)/2 and
    # N_NS(P)^2=-U_NS(2+2iP) U_NS(2-2iP).
    # Match C_R^+ C_NS (two R) or C_R^+ C_R^+ (four R),
    # including the half in EACH physical RRNS three-point constant.
    return (-1)**(nr//2)*numerator/2**(2+nr//2)


def four_r_amplitude(raw, momenta):
    """Return mu_F^2 M, energy delta function omitted."""
    return 1j*16/math.pi*math.prod(momenta)*external_factor(momenta, ('R',)*4)*raw


def mixed_amplitude(raw, momenta):
    """Convert the mixed integral in the tested full-field convention.

    Stored HJS W has i times the BRY full-field convention. The one
    raised NS vertex contributes (-i)/4 after the odd-modulus insertion.
    This phase must be common to all four resolved PCO contributions.
    """
    return (1j*16/math.pi*momenta[0]*momenta[1]
            *external_factor(momenta, ('R', 'R', 'NS', 'NS'))*(-1j/4)*raw)


def matrix_prediction(t):
    """Comparison only; never called by the integrand or normalization."""
    return 1j*t**4*(1-2*t)/27


def trinion_checks(precision=50):
    rows = []
    for a, b in ((.17, .29), (.09j, .16j), (.14+.03j, .22+.04j)):
        w = a+b
        ce, _ = rr_ns_structure_constants(a, b, w, precision)
        _, co = rr_ns_structure_constants(w, a, b, precision)
        ct = ns_tilde_structure_constant(w, a, b, precision)
        rows.append(dict(momenta=(w, a, b),
            relative_even=abs(ce-w/2)/abs(w/2),
            relative_odd=abs(co-b/2)/abs(b/2),
            relative_ns=abs(ct-w*a*b)/abs(w*a*b)))
    return rows
