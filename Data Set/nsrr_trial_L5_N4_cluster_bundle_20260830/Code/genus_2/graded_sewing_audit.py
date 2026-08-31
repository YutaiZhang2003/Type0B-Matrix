"""Finite-dimensional audits of the Human Note's physical BPZ sewing.

This is a convention audit, not a super-Liouville partition evaluator.
In particular it supplies no Ramond state projector or three-point tensor.
The caller must supply those physical data before applying this contraction.

All pairings here are BILINEAR. An antiholomorphic pairing/tensor is an
explicit input, not an implicit complex conjugate or Hermitian replacement.
The auxiliary SCA+F tensor metric is deliberately not used or modified.
"""
from __future__ import annotations

import numpy as np


def _parities(values, dimension):
    p = np.asarray(values)
    if p.shape != (dimension,) or not np.all((p == 0) | (p == 1)):
        raise ValueError("one binary parity per basis state is required")
    return p.astype(int)


def _even_pairing(gram, parities):
    g = np.asarray(gram, dtype=complex)
    if g.ndim != 2 or g.shape[0] != g.shape[1] or not np.all(np.isfinite(g)):
        raise ValueError("a finite square pairing is required")
    p = _parities(parities, len(g))
    if np.any(abs(g[p[:, None] != p[None, :]]) > 1e-13):
        raise ValueError("the BPZ pairing must preserve parity")
    return g, p


def graded_tensor_gram(gram, parities, anti_gram, anti_parities):
    r"""G_(a,abar),(b,bbar) = (-1)^(p_b p_abar) B_ab Btilde_abar,bbar.

    Product-basis order is (holomorphic, antiholomorphic). Ground-state
    normalization factors, if needed, belong in the input pairings.
    """
    b, p = _even_pairing(gram, parities)
    bt, pt = _even_pairing(anti_gram, anti_parities)
    sign = (-1.0) ** (pt[None, :, None, None] * p[None, None, :, None])
    result = b[:, None, :, None] * bt[None, :, None, :] * sign
    return result.reshape(len(b)*len(bt), len(b)*len(bt))


def theta_orientation(parities):
    """The theta-pants reordering sign; never discard it as a metric phase."""
    if len(parities) != 3:
        raise ValueError("theta sewing has three edges")
    p = [_parities(x, len(x)) for x in parities]
    a, b, c = p[0][:, None, None], p[1][None, :, None], p[2][None, None, :]
    return (-1.0) ** (a*b+a*c+b*c)


def theta_bilinear_contraction(left, right, grams, parities):
    """Contract two given pants tensors, including the quadratic sign.

    For a physical nonchiral computation, `grams` must be the PHYSICAL
    graded pairings and the tensors must already contain the appropriate
    Ramond state restriction and three-point coefficients. There is no
    default identity projector, Hermitian conjugation, or positivity rule.
    Propagation factors can be included in either tensor; tests mix basis
    states only within a common level, so propagation commutes with them.
    """
    if len(grams) != 3 or len(parities) != 3:
        raise ValueError("theta sewing has three edges")
    checked = [_even_pairing(g, p) for g, p in zip(grams, parities)]
    shape = tuple(len(g) for g, _ in checked)
    l, r = np.asarray(left, dtype=complex), np.asarray(right, dtype=complex)
    if l.shape != shape or r.shape != shape:
        raise ValueError("pants tensor dimensions must match the three pairings")
    if not np.all(np.isfinite(l)) or not np.all(np.isfinite(r)):
        raise ValueError("finite pants tensors are required")
    inverse = [np.linalg.inv(g) for g, _ in checked]
    return complex(np.einsum("abc,ad,be,cf,def->", l*theta_orientation(parities),
                             *inverse, r, optimize=True))
