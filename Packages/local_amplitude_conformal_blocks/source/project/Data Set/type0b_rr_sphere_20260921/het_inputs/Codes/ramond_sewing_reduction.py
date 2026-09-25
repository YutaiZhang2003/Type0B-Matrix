"""Ramond completeness reduction in the literal NS-at-infinity human frame.

Research layer, NOT an interacting Liouville/fixed-spin production adapter.
The theta theorem takes the two ordered three-point coefficient tensors as
INPUT. It does not infer their BPZ/reflection dictionary from ground norms.
No function conjugates momenta, vertex coefficients, or block values.

See docs/so7e8/SO7E8_RAMOND_SEWING_DERIVATION_20260915.md.
"""
from dataclasses import dataclass
from itertools import product

import numpy as np
import sympy as sp

from graded_sphere_sewing import physical_tensor_gram

SIGNS = (1, -1)


def liouville_primary_weights(b, momentum):
    """Ordinary central charge; momentum is physical P, not human algebraic iP."""
    b, momentum = map(sp.sympify, (b, momentum))
    if b.is_zero is True:
        raise ValueError("b must be nonzero")
    q_background = b + 1 / b
    c = sp.Rational(3, 2) + 3 * q_background**2
    h_ns = q_background**2 / 8 + momentum**2 / 2
    return tuple(map(sp.factor, (c, h_ns, h_ns + sp.Rational(1, 16))))


def branching_onset(n, sector):
    """Level ABOVE the SCA+auxiliary ground, not an absolute primary weight."""
    n = sp.Rational(n)
    if sector == "NS":
        if (2 * n).q != 1:
            raise ValueError("NS n must belong to half-integers")
        return 2 * n**2
    if sector == "R":
        if (2 * n - sp.Rational(1, 2)).q != 1:
            raise ValueError("R n must belong to 1/4 + half-integers")
        return 2 * n**2 - sp.Rational(1, 8)
    raise ValueError("sector must be NS or R")


@dataclass(frozen=True)
class BranchWeights:
    central_charges: tuple
    virasoro_weights: tuple
    sca_primary_weight: sp.Expr
    auxiliary_primary_weight: sp.Expr
    enlarged_ground_weight: sp.Expr
    onset: sp.Expr


def double_virasoro_weights(b, momentum, n, sector):
    """Rational form of the human note's branch weights with coherent roots.

    The two individual Virasoro systems are singular at b^2=1. Their SUM
    is not; the existing coefficient generator takes its regularized limit.
    """
    b, momentum, n = map(sp.sympify, (b, momentum, n))
    onset = branching_onset(n, sector)
    if b.is_zero is True or (b**2 - 1).is_zero is True:
        raise ValueError("individual double-Virasoro weights need b != 0, b^2 != 1")
    _, h_ns, h_r = liouville_primary_weights(b, momentum)
    q_background = b + 1 / b
    algebraic_momentum = sp.I * momentum
    denominator = (2 * (1 - b**2), 2 * (1 - b**-2))
    shift = (2 * b * n, 2 * n / b)
    hs = tuple(sp.factor((q_background**2 / 4 - (algebraic_momentum + s)**2) / d)
               for s, d in zip(shift, denominator))
    cs = tuple(sp.factor(1 + 6 * q_background**2 / d) for d in denominator)
    h_sca = h_ns if sector == "NS" else h_r
    h_aux = sp.S.Zero if sector == "NS" else sp.Rational(1, 16)
    return BranchWeights(cs, hs, h_sca, h_aux, h_sca + h_aux, onset)


def small_ramond_projector(chi_h, chi_a, holomorphic_parities, antiholomorphic_parities):
    r"""Pi_-=(1-J)/2, J=i chi_h tensor (chi_a (-1)^F_a).

    chi_h is the ordinary commuting odd map Dw+ -> Dw-, Dw- -> i Dw+.
    chi_a uses -i. In particular chi is NOT G0/(exp(i*pi/4)*beta)
    on descendants. Only a supplied, consistent pair of module frames is
    supported. This specifies the ket restriction, not physical vertices.
    """
    h, a = np.asarray(chi_h, complex), np.asarray(chi_a, complex)
    ph, pa = tuple(holomorphic_parities), tuple(antiholomorphic_parities)
    for matrix, bits, square in ((h, ph, 1j), (a, pa, -1j)):
        if (matrix.shape != (len(bits), len(bits)) or not len(bits)
                or any(p not in (0, 1) for p in bits)
                or not np.all(np.isfinite(matrix))):
            raise ValueError("finite square chi and matching parity bits required")
        parity = np.diag([(-1)**p for p in bits])
        if not (np.allclose(matrix @ matrix, square * np.eye(len(bits)))
                and np.allclose(matrix @ parity, -parity @ matrix)):
            raise ValueError("chi square or parity relation failed")
    j = 1j * np.kron(h, a @ np.diag([(-1)**p for p in pa]))
    return (np.eye(len(ph) * len(pa)) - j) / 2


def small_ramond_coevaluation(gram_h, gram_a, parities_h, parities_a, chi_h, chi_a):
    """Canonical ORIENTED coevaluation Pi_- D^-1 in the specified frames.

    This matrix is not a Hermitian norm and need not be symmetric. The
    compatible bra subspace is determined by this oriented pairing. It
    must not be used with unchanged vertices from a different dual frame.
    """
    d = physical_tensor_gram(gram_h, gram_a, parities_h, parities_a)
    projector = small_ramond_projector(chi_h, chi_a, parities_h, parities_a)
    return np.linalg.solve(d.T, projector.T).T


@dataclass(frozen=True)
class ThetaSewingTerm:
    coefficient: complex
    holomorphic_signs: tuple[int, int]
    antiholomorphic_signs: tuple[int, int]
    flip_third_lift: bool


def theta_reduction_terms(left, right):
    r"""Reduce Pi_- on BOTH R edges to human EVEN blocks at two lifts.

    left/right[(f,s,t)] are the coefficients in
      T_v=(-1)^tau sum t_v[f,s,t] rho_f^s bar(rho_f^t).
    Ordered slots: (NS at infinity, R at 1, R at 0). NS primary is even;
    holomorphic and antiholomorphic tube lifts are the same. The primary
    propagation factors and inverse physical two-point normalization are
    outside this coefficient formula. Two vertex orientations stay distinct.

    This algebraic theorem does NOT certify that a proposed physical vertex
    has the assumed all-level expansion. In particular a ground-only fit of
    left/right is insufficient; a failed attempt is documented in the note.
    """
    labels = tuple(product((0, 1), SIGNS, SIGNS))
    if set(left) != set(labels) or set(right) != set(labels):
        raise ValueError("both ordered vertices require all eight (f,s,t) entries")
    left, right = ({k: complex(v) for k, v in x.items()} for x in (left, right))
    if any(not np.isfinite(v) for x in (left, right) for v in x.values()):
        raise ValueError("finite vertex coefficients required")
    terms = []
    for s, u, v in product(SIGNS, repeat=3):
        r = right[0, u, v] - 1j * right[1, u, v]
        for flip, l in ((False, left[0, s, s]), (True, -1j * left[1, s, s])):
            coefficient = l * r / 2
            if coefficient != 0:
                terms.append(ThetaSewingTerm(coefficient, (s, u), (s, v), flip))
    return tuple(terms)


def contract_theta_reduction(terms, holomorphic_blocks, antiholomorphic_blocks):
    """Block banks keyed by (left sign,right sign,flip_third_lift).

    Supply the two chiralities independently, including for complex momenta.
    No complex conjugation, absolute square, integration, or spin sum here.
    """
    return sum((term.coefficient
                * holomorphic_blocks[(*term.holomorphic_signs, term.flip_third_lift)]
                * antiholomorphic_blocks[(*term.antiholomorphic_signs, term.flip_third_lift)]
                for term in terms), 0j)
