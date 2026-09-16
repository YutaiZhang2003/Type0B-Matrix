"""Physical NSRR sewing with the Human Note's graded, bilinear BPZ dual.

The eight inputs are descendant-only two-lift blocks, ordered by
(form parity, eta_left, eta_right). The two-lift operation is a BASIS
projection, not an assignment of a marked surface spin characteristic.
Physical tube signs are a separate, mandatory argument in NS,R1,R0 order.
See NSRR_BILINEAR_PAIRING_2026-09-15.md for the all-level Clifford reduction.
"""
from __future__ import annotations

import math
import numpy as np

CHANNELS = tuple((f, eta, etap) for f in (0, 1)
                 for eta in (1, -1) for etap in (1, -1))
CONVENTION = "physical NSRR; Human-Note graded BPZ bilinear; HJS antichiral basis"
BASE_PROJECTION_LIFTS = ((1, 1, 1), (1, -1, 1))


def physical_signs(physical_lifts_slots):
    """Return the two gauge-invariant full-state tube characters (s,r)."""
    omega = tuple(physical_lifts_slots)
    if len(omega) != 3 or any(x not in (-1, 1) for x in omega):
        raise ValueError("Supply three physical tube signs +/-1 in NS,R1,R0 order")
    return omega[0]*omega[2], omega[1]*omega[2]


def projected_components(components):
    """(F(+++)+F(+-+))/sqrt(2), for native NS,R1,R0 parity components."""
    if len(components) != 8:
        raise ValueError("Eight absolute-parity components are required")
    return math.sqrt(2)*sum(complex(z) for p, z in enumerate(components) if not p & 2)


def nsrr_bilinear_matrix(left_bry, right_bry, *, physical_lifts_slots):
    """M_(f,eta,eta'),(g,zeta,zeta') in the saved projected block basis.

    B_+=E, B_-=O are the supplied BRY constants on EACH pant. No scalar
    coefficient is conjugated. M contains no momentum measure or q^h.
    """
    if len(left_bry) != 2 or len(right_bry) != 2:
        raise ValueError("Supply (E,O) independently for both pants")
    sign, rsign = physical_signs(physical_lifts_slots)
    left = dict(zip((1, -1), map(complex, left_bry)))
    right = dict(zip((1, -1), map(complex, right_bry)))
    if not all(math.isfinite(z.real) and math.isfinite(z.imag)
               for z in (*left.values(), *right.values())):
        raise ValueError("Three-point coefficients must be finite")
    matrix = np.zeros((8, 8), complex)
    for i, (f, eta, etap) in enumerate(CHANNELS):
        if eta*etap == -rsign:
            for g in (0, 1):
                j = CHANNELS.index((g, eta, etap))
                matrix[i, j] = left[eta]*right[etap]/8 * (1 if f == g else sign)
    return matrix


def contract_nsrr_bilinear(*, descendant_blocks, antiholomorphic_blocks,
                           left_bry, right_bry, physical_lifts_slots,
                           primary, antiholomorphic_primary):
    """P Ptilde F^T M Ftilde, with independent anti data and primary powers.

    On the real physical slice only, the HJS anti blocks at conjugate q
    equal conjugate(F). The caller must state that specialization; it is
    never applied inside this bilinear API.
    """
    if set(descendant_blocks) != set(CHANNELS) or set(antiholomorphic_blocks) != set(CHANNELS):
        raise ValueError("All eight projected channels are required on both sides")
    matrix = nsrr_bilinear_matrix(left_bry, right_bry,
                                  physical_lifts_slots=physical_lifts_slots)
    f = np.array([descendant_blocks[label] for label in CHANNELS], complex)
    ft = np.array([antiholomorphic_blocks[label] for label in CHANNELS], complex)
    propagation = complex(primary)*complex(antiholomorphic_primary)
    terms, diagonal_terms = {}, {}
    for eta in (1, -1):
        for etap in (1, -1):
            ix = [CHANNELS.index((p, eta, etap)) for p in (0, 1)]
            m = matrix[np.ix_(ix, ix)]
            terms[eta, etap] = complex(propagation*(f[ix] @ m @ ft[ix]))
            diagonal_terms[eta, etap] = complex(propagation*sum(f[i]*matrix[i, i]*ft[i] for i in ix))
    total, diagonal = sum(terms.values()), sum(diagonal_terms.values())
    return dict(total=total, diagonal=diagonal, interference=total-diagonal,
                terms=terms, diagonal_terms=diagonal_terms,
                sewing_convention=CONVENTION)
