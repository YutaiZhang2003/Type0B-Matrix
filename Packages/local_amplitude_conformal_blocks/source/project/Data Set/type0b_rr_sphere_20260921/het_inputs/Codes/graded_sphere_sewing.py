"""Graded BPZ completeness, in the Type0B human-note convention.

This is the physical, holomorphic/antiholomorphic tensor product, NOT the
primed auxiliary pairing used to generate double-Virasoro coefficients.
All pairings are bilinear. A caller using HJS sesquilinear ground tensors
must translate its dual frame explicitly; this module never conjugates
couplings, momenta, or external-state coefficients.

The routines do not choose a GSO projection or RRNS three-point constants.
In particular, ``sphere_parity_routes`` is not a substitute for the physical
Ramond small-representation projection. See
docs/so7e8/SO7E8_GRADED_SPHERE_SEWING.md for the derivation and limitations.
"""

from dataclasses import dataclass
from itertools import product
from typing import Sequence

import numpy as np


def _bits(values, *, length=None):
    result = tuple(values)
    if (length is not None and len(result) != length) or any(
        value not in (0, 1) for value in result
    ):
        raise ValueError("parities must be zero/one entries of the required length")
    return tuple(int(value) for value in result)


def koszul_permutation_sign(parities, permutation):
    """Sign of reordering labelled factors into ``permutation`` order.

    ``permutation[j]`` is the original index of the new j-th factor. Only
    inversions of two odd factors contribute. Labels must remain distinct
    even when their parities agree.
    """
    parities = _bits(parities)
    order = tuple(permutation)
    if sorted(order) != list(range(len(parities))):
        raise ValueError("permutation must contain each factor index exactly once")
    exponent = sum(
        parities[order[i]] * parities[order[j]]
        for i in range(len(order)) for j in range(i + 1, len(order))
        if order[i] > order[j]
    )
    return -1 if exponent % 2 else 1


def chiral_separation_sign(holomorphic, antiholomorphic):
    """Reorder h3,b3,h2,b2,h1,b1 into h3,h2,h1,b3,b2,b1.

    Inputs are in the actual displayed operator order, not increasing
    puncture order. The implementation also accepts any number of slots.
    """
    hol = _bits(holomorphic)
    anti = _bits(antiholomorphic, length=len(hol))
    interleaved = tuple(value for pair in zip(hol, anti) for value in pair)
    n = len(interleaved)
    return koszul_permutation_sign(
        interleaved, tuple(range(0, n, 2)) + tuple(range(1, n, 2))
    )


def _square_matrix(value, label):
    matrix = np.asarray(value, dtype=np.complex128)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError(f"{label} must be a square matrix")
    if not np.all(np.isfinite(matrix)):
        raise ValueError(f"{label} must have finite entries")
    return matrix


def _even_gram(value, parities, label):
    matrix = _square_matrix(value, label)
    bits = _bits(parities, length=len(matrix))
    cross = np.not_equal.outer(bits, bits)
    if np.any(np.abs(matrix[cross]) > 1e-12 * max(1, np.max(np.abs(matrix)))):
        raise ValueError(f"{label} must preserve homogeneous state parity")
    return matrix, bits


def physical_tensor_gram(holomorphic_gram, antiholomorphic_gram,
                         holomorphic_parities, antiholomorphic_parities):
    r"""Return B_(i,j;k,l) = (-1)^(bar_p_j p_k) G_ik Gbar_jl.

    Flattened indices are (holomorphic, antiholomorphic), in that order.
    Use TOTAL state parity, including a Ramond ground-state parity.
    The primary normalization D and any change from formal tensor states
    to physical states belong in the caller's Gram/embedding data.
    """
    gh, ph = _even_gram(holomorphic_gram, holomorphic_parities, "holomorphic Gram")
    ga, pa = _even_gram(antiholomorphic_gram, antiholomorphic_parities, "antiholomorphic Gram")
    sign = 1 - 2 * np.multiply.outer(pa, ph)
    tensor = np.einsum("ik,jl,jk->ijkl", gh, ga, sign)
    return tensor.reshape(len(gh) * len(ga), len(gh) * len(ga))


def sewing_kernel(gram, embedding=None, *, dual_embedding=None):
    r"""Inverse metric/co-evaluation, optionally on a specified subspace.

    For a single bilinear frame E, K = E (E^T B E)^-1 E^T. The state
    projector is K B, NOT K. With distinct ket and dual frames E and F,
    K = E (F^T B E)^-1 F^T. Frames are explicit data, never inferred from
    a trinion structure sign. No Hermitian conjugation is performed.
    """
    metric = _square_matrix(gram, "Gram matrix")
    if embedding is None:
        if dual_embedding is not None:
            raise ValueError("dual_embedding requires embedding")
        return np.linalg.solve(metric, np.eye(len(metric), dtype=complex))
    ket = np.asarray(embedding, dtype=complex)
    dual = ket if dual_embedding is None else np.asarray(dual_embedding, dtype=complex)
    if (ket.ndim != 2 or dual.shape != ket.shape or ket.shape[0] != len(metric)
            or ket.shape[1] == 0 or not np.all(np.isfinite(ket))
            or not np.all(np.isfinite(dual))):
        raise ValueError("ket and dual embeddings must be finite, matching n-by-r matrices")
    restricted = dual.T @ metric @ ket
    return ket @ np.linalg.solve(restricted, dual.T)


def sew_covectors(left, kernel, right):
    """Contract two full three-point covectors with an inverse metric."""
    metric_inverse = _square_matrix(kernel, "sewing kernel")
    left, right = np.asarray(left, dtype=complex), np.asarray(right, dtype=complex)
    if left.shape != (len(metric_inverse),) or right.shape != left.shape:
        raise ValueError("trinion covectors must match the sewing kernel")
    return complex(left @ metric_inverse @ right)


@dataclass(frozen=True)
class SphereParityRoute:
    """One homogeneous (p, pbar) sector of the unprojected tensor module."""

    internal_parities: tuple[int, int]
    holomorphic_form_parities: tuple[int, int]  # (left, right)
    antiholomorphic_form_parities: tuple[int, int]
    left_separation_exponent: int
    right_separation_exponent: int
    inverse_gram_exponent: int

    @property
    def sign(self):
        exponent = (self.left_separation_exponent
                    + self.right_separation_exponent + self.inverse_gram_exponent)
        return -1 if exponent % 2 else 1


def sphere_parity_routes(external_holomorphic, external_antiholomorphic):
    r"""Signs for <4|3(1) 2(z)|1>, with external input order (1,2,3,4).

    For internal parity (p,pbar), the exponents are
      left:  b4(a3+p) + b3 p,
      right: pbar(a2+a1) + b2 a1,
      inverse Gram: p pbar.
    These apply to *separated* trinion forms and the unprimed physical
    Gram. They must not be multiplied onto an existing phase-dressed
    correlator, which may already contain some or all of these signs.
    """
    a1, a2, a3, a4 = _bits(external_holomorphic, length=4)
    b1, b2, b3, b4 = _bits(external_antiholomorphic, length=4)
    return tuple(SphereParityRoute(
        (p, pb), (a4 ^ a3 ^ p, p ^ a2 ^ a1),
        (b4 ^ b3 ^ pb, pb ^ b2 ^ b1),
        (b4 * (a3 + p) + b3 * p) % 2,
        (pb * (a2 + a1) + b2 * a1) % 2, p * pb,
    ) for p, pb in product((0, 1), repeat=2))


@dataclass(frozen=True)
class SeparatedTrinionTerm:
    """Coefficient times a holomorphic and an antiholomorphic covector.

    The coefficient includes physical three-point data and basis changes,
    but NOT the chiral-separation sign. Covectors already include external
    descendants and their Ward-identity phases. A sum of such terms can
    encode any trinion tensor; no diagonal form-pairing rule is assumed.
    """

    coefficient: complex
    holomorphic: Sequence[complex]
    antiholomorphic: Sequence[complex]


def factorized_sphere_sewing(left_terms, right_terms, *, holomorphic_gram,
                             antiholomorphic_gram, holomorphic_parities,
                             antiholomorphic_parities, external_holomorphic,
                             external_antiholomorphic):
    """Evaluate the derived chiral-block sum for a full tensor module.

    This is a small finite-basis reference, not a production block engine.
    For a physical submodule, use ``sewing_kernel(..., embedding=...)`` on
    full tensors, or first prove that both trinions annihilate the excluded
    complement. Only in the latter case is the full inverse Gram sufficient.
    """
    gh, ph = _even_gram(holomorphic_gram, holomorphic_parities, "holomorphic Gram")
    ga, pa = _even_gram(antiholomorphic_gram, antiholomorphic_parities, "antiholomorphic Gram")
    hi, ai = sewing_kernel(gh), sewing_kernel(ga)

    def vectors(terms):
        result = []
        for term in terms:
            hol, anti = np.asarray(term.holomorphic, dtype=complex), np.asarray(term.antiholomorphic, dtype=complex)
            if hol.shape != (len(gh),) or anti.shape != (len(ga),):
                raise ValueError("separated trinion covectors have incorrect dimensions")
            result.append((complex(term.coefficient), hol, anti))
        return result

    left, right = vectors(left_terms), vectors(right_terms)
    result = 0j
    for route in sphere_parity_routes(external_holomorphic, external_antiholomorphic):
        p, pb = route.internal_parities
        hm, am = np.asarray(ph) == p, np.asarray(pa) == pb
        for lc, lh, la in left:
            for rc, rh, ra in right:
                hol = (lh * hm) @ hi @ (rh * hm)
                anti = (la * am) @ ai @ (ra * am)
                result += route.sign * lc * rc * hol * anti
    return complex(result)
