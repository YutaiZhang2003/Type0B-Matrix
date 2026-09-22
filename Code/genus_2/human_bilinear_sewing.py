"""Genus-two assembly with the bilinear pairing of Human Notes/SCblock.tex.

Holomorphic and antiholomorphic data are independent inputs. Transposition
never conjugates a coefficient. The graded full-field pairing in the note
is distinct from its ungraded, primed SCA-times-auxiliary-fermion pairing.
All primary propagation factors remain outside the descendant blocks and M.
"""
from __future__ import annotations

from typing import Mapping, Sequence

import numpy as np

try:
    from .theta_partition import theta_sector_pair
except ImportError:  # Direct execution from Code/genus_2.
    from theta_partition import theta_sector_pair


CONVENTION = "Human-Note graded BPZ bilinear; independent antiholomorphic data"


def graded_product_gram(holomorphic_gram, antiholomorphic_gram,
                        holomorphic_parities, antiholomorphic_parities):
    """B_(a,at),(b,bt)=(-1)^(p_b p_at) B_ab Btilde_at,bt.

    These are absolute state parities, including the ground-state parity.
    The inputs must already be Gram matrices in their specified dual bases.
    No Hermitian replacement or physical Ramond restriction is performed.
    """
    h = np.asarray(holomorphic_gram, dtype=complex)
    a = np.asarray(antiholomorphic_gram, dtype=complex)
    hp = tuple(holomorphic_parities)
    ap = tuple(antiholomorphic_parities)
    if (h.shape != (len(hp), len(hp)) or a.shape != (len(ap), len(ap))
            or any(p not in (0, 1) for p in hp + ap)):
        raise ValueError("Square Gram matrices and their absolute parity lists are required")
    result = np.kron(h, a)
    for i in range(len(hp)):
        for j, anti_parity in enumerate(ap):
            for k, holo_parity in enumerate(hp):
                result[i*len(ap)+j, k*len(ap):(k+1)*len(ap)] *= (-1)**(anti_parity*holo_parity)
    return result


def restricted_bilinear_completeness(full_gram, ket_embedding, dual_embedding):
    """Restrict to declared ket/dual bases BEFORE inverting the pairing.

    Return (G_phys, K) with G_phys=E_L^T G_full E_R and
    K=E_R G_phys^(-1) E_L^T. The caller must supply the physical embeddings
    at the requested descendant level; a ground embedding is insufficient.
    """
    g = np.asarray(full_gram, dtype=complex)
    right = np.asarray(ket_embedding, dtype=complex)
    left = np.asarray(dual_embedding, dtype=complex)
    if (g.ndim != 2 or right.ndim != 2 or left.ndim != 2
            or g.shape != (left.shape[0], right.shape[0])
            or left.shape[1] != right.shape[1]):
        raise ValueError("The Gram and two embeddings must describe matching dual spaces")
    physical = left.T @ g @ right
    try:
        completeness = right @ np.linalg.solve(physical, left.T)
    except np.linalg.LinAlgError as exc:
        raise ValueError("The supplied physical ket/dual pairing is singular") from exc
    return physical, completeness


def pairing_in_new_bases(matrix, left_change, right_change):
    """For F'=U F and Ftilde'=V Ftilde, return U^(-T) M V^(-1).

    This applies to bases with the same external primary factors. A
    weight-changing transformation must also transport the explicit P's.
    """
    m = np.asarray(matrix, dtype=complex)
    u = np.asarray(left_change, dtype=complex)
    v = np.asarray(right_change, dtype=complex)
    if m.ndim != 2 or u.shape != (m.shape[0], m.shape[0]) or v.shape != (m.shape[1], m.shape[1]):
        raise ValueError("Basis changes must be square and match the two block spaces")
    return np.linalg.solve(v.T, np.linalg.solve(u.T, m).T).T


def descendant_basis_change(propagated_change, old_primary, new_primary):
    """If P'F'=U(PF), return D_(P')^(-1) U D_P.

    Primary factors are evaluated with the caller's continued logarithms.
    Keeping them explicit matters when a channel changes primary weights.
    This finite-dimensional operation is not a Liouville fusion kernel.
    """
    u = np.asarray(propagated_change, dtype=complex)
    p = np.asarray(old_primary, dtype=complex)
    pp = np.asarray(new_primary, dtype=complex)
    if u.ndim != 2 or p.shape != (u.shape[1],) or pp.shape != (u.shape[0],):
        raise ValueError("Provide one explicit primary factor per old and new block")
    if np.any(pp == 0):
        raise ValueError("New primary factors must be nonzero")
    return u * p[None, :] / pp[:, None]


def contract_bilinear(*, matrix, holomorphic_blocks, antiholomorphic_blocks,
                      holomorphic_primary=1, antiholomorphic_primary=1):
    """Contract descendant blocks; each side may have its own primary powers."""
    f = np.asarray(holomorphic_blocks, dtype=complex)
    ft = np.asarray(antiholomorphic_blocks, dtype=complex)
    m = np.asarray(matrix, dtype=complex)
    if f.ndim != 1 or ft.ndim != 1 or m.shape != (len(f), len(ft)):
        raise ValueError("The matrix dimensions must match both block vectors")
    p = np.asarray(holomorphic_primary, dtype=complex)
    pt = np.asarray(antiholomorphic_primary, dtype=complex)
    if p.ndim > 1 or pt.ndim > 1 or (p.ndim == 1 and p.shape != f.shape) or (pt.ndim == 1 and pt.shape != ft.shape):
        raise ValueError("Supply a scalar or one primary factor per block on each side")
    return complex((p*f) @ m @ (pt*ft))


def all_ns_theta_matrix(left_constants: Sequence[complex],
                        right_constants: Sequence[complex], *,
                        holomorphic_primary_parities=(0, 0, 0),
                        antiholomorphic_primary_parities=(0, 0, 0)):
    """Literal Human-Note theta M, including its absolute-parity sign.

    Constants are ordered by the holomorphic form index a. They are the
    two pants' full three-point constants C^(a), not their absolute values.
    For the supplied BRY constants use (C, i*Ctilde) on EACH pant.
    The quadratic K sign remains inside the supplied literal chiral blocks.
    """
    if len(left_constants) != 2 or len(right_constants) != 2:
        raise ValueError("Both three-point coefficients are required on each pant")
    result = np.zeros((2, 2), dtype=complex)
    for sector in (0, 1):
        pair = theta_sector_pair(
            sector, holomorphic_primary_parities=holomorphic_primary_parities,
            antiholomorphic_primary_parities=antiholomorphic_primary_parities)
        result[sector, pair.antiholomorphic_sector] = (
            pair.sign * complex(left_constants[sector]) * complex(right_constants[sector]))
    return result


def nsrr_tensor_product_matrix(left_vertex: Mapping, right_vertex: Mapping):
    """Exact grading reduction for a SPECIFIED full NSRR vertex tensor.

    t[f,eta,zeta] multiplies rho_f^eta times rhotilde_f^zeta in the same
    Human-Note bilinear frame, with an even NS primary. The output labels
    are (f,eta_left,eta_right).
    This is for the full tensor-product edge modules. It does NOT infer the
    physical small-Ramond-module restriction from BRY's two constants.
    """
    labels = tuple((f, a, b) for f in (0, 1) for a in (1, -1) for b in (1, -1))
    result = np.zeros((8, 8), dtype=complex)
    for i, (f, eta, etap) in enumerate(labels):
        for j, (g, zeta, zetap) in enumerate(labels):
            if f == g:
                result[i, j] = ((-1)**f * complex(left_vertex.get((f, eta, zeta), 0))
                                * complex(right_vertex.get((f, etap, zetap), 0)))
    return labels, result
