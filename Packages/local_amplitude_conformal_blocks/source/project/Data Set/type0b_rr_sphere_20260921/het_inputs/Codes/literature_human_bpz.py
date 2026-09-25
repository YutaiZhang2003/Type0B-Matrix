"""Candidate raw-plane BPZ transport, separate from production.

CORRECTION after comparison with the actual Type0B code: this is NOT the
complete fixed-parity Human Note dictionary. The NR values match Type0B's
literal plane PBW routine. A separate Type0B check assumes a different
NS-first Ward identity; its parity rephasing has not been derived from the
Human Note's definitions. The earlier Appendix A correction claims are
withdrawn; see audit_type0b_pbw_conventions.py and its report.

This implements unit component seeds, the NS component table, and a
same-beta LINEAR BPZ pairing with G_n^T=G_-n and B(w+,w+)=1. These imply
B(w-,w-)=i. Algebraic invariance of complete sewing does not identify this
raw frame with Type0B's normalized ordered-slot convention.

Numerical domain is inherited from literature_component_blocks: c=3, real
momentum labels, primary multiplet external components, principal slit plane.
Antiholomorphic frames are the complex-conjugate frames on this contour.
"""
from itertools import product
import math

import numpy as np

from literature_component_blocks import (
    ComponentBlocks, SECTORS, _bit, _components, _ground, _ns, _parity,
    ramond_vertex,
)


SIGNS = (1, -1)


def momentum_dictionary(p):
    """Choose beta_H=-P_H/sqrt(2) once, including negative reflected labels."""
    if complex(p).imag or not math.isfinite(float(p)):
        raise ValueError('a finite real literature momentum is required')
    p = float(p)
    return dict(p=p, P_H=1j*p, beta=-1j*p/math.sqrt(2),
                lambda_NS=-2j*p, h_NS=.0625+p*p/2, h_R=.125+p*p/2)


def sign_matrix(order, parity, *, human):
    """Rows (+,-), columns (++ , --) or (+- , -+) in the ordered slots."""
    if order not in ('NR', 'RN', 'RR'):
        raise ValueError('expected NR=(NS,R,R), RN=(R,R,NS), or RR=(R,NS,R)')
    if _bit(parity) == 0:
        return np.array([[1, 1], [1, -1]], complex)
    if not human and order == 'RN':
        return np.array([[1j, 1], [-1j, 1]], complex)
    return np.array([[1, 1j], [1, -1j]], complex)


def notation_map(order, parity):
    """Only rename sign forms, retaining the literature's antilinear dual."""
    return sign_matrix(order, parity, human=True) @ np.linalg.inv(
        sign_matrix(order, parity, human=False))


def bpz_sign_map(order, parity):
    """Seed-normalized map BEFORE the actual bra factor i^ground(bra).

    NR: I. RN: M (even), diag(1,-1) M (odd). RR: M in both parities.
    The matrix M is 1/2 [[1-i,1+i],[1+i,1-i]].
    """
    seed_dual_inverse = np.eye(2) if order == 'NR' else np.diag([1, -1j])
    return sign_matrix(order, parity, human=True) @ seed_dual_inverse @ np.linalg.inv(
        sign_matrix(order, parity, human=False))


def bpz_three_point(vector, order, parity, eta, *, bra_ground=0):
    """Linear BPZ form from the four literature unit-seed coefficient forms."""
    if eta not in SIGNS:
        raise ValueError('eta must be +1 or -1')
    sign_matrix(order, parity, human=True)  # validates order and parity
    values = np.asarray(vector, complex).copy()
    if values.shape != (4,):
        raise ValueError('four component forms (++,+-,-+,--) are required')
    if order != 'NR':
        values *= (1j**_bit(bra_ground))*np.array([1, 1, -1j, -1j])
    return values[0]+eta*values[3] if parity == 0 else values[1]+1j*eta*values[2]


def ns_ordering_sign(descendants, primaries=(0, 0, 0)):
    """Human Note NS table, slots (infinity,1,0); levels do not replace parity."""
    a, b, d = tuple(map(_bit, descendants))
    p, q, _ = tuple(map(_bit, primaries))
    return (-1)**(((a+b+d) % 2)*d+p*a+q*d)


def bpz_gram(module, level, parity):
    """Same-beta B=D K, not S^dagger K S; D_vv=i^ground(v)."""
    gram = module.gram(level, parity)
    if module.sector == 'NS':
        return gram.copy()
    basis = module.basis(level, parity)
    return np.diag([1j**v[1] for v in basis]) @ gram


def sign_pairs(family):
    if family not in SECTORS:
        raise ValueError('unknown family')
    return tuple(product((1,) if family == 'mixed_ns' else SIGNS, SIGNS))


def block_map(family, external, internal_parity):
    """F_H=T F_lit; internal bra factors have cancelled against B^-1.

    Use the actual block's right sign after momentum reflection. No branch,
    coordinate, external weight, or momentum is changed by this map.
    """
    a1, a2, a3, a4 = tuple(map(_bit, external))
    k = _bit(internal_parity)
    if family == 'rrrr':
        return 1j**a4*np.kron(bpz_sign_map('RN', (a4+a3+k) % 2), np.eye(2))
    if family == 'mixed_ns':
        return ns_ordering_sign((a4, a3, k))*np.eye(2)
    if family == 'mixed_r':
        return np.kron(np.eye(2), bpz_sign_map('RN', (k+a2+a1) % 2))
    raise ValueError('unknown family')


def transport_coefficient(coefficient, holomorphic_map, antiholomorphic_map):
    """C_H=T_h^(-T) C_lit T_a^(-1), with transpose, never adjoint."""
    c, th, ta = map(lambda x: np.asarray(x, complex),
                    (coefficient, holomorphic_map, antiholomorphic_map))
    if (th.ndim != 2 or ta.ndim != 2 or th.shape[0] != th.shape[1]
            or ta.shape[0] != ta.shape[1] or c.shape != (len(th), len(ta))):
        raise ValueError('incompatible coefficient tensor and basis maps')
    return np.linalg.solve(ta.T, np.linalg.solve(th.T, c).T).T


class HumanBPZBlocks(ComponentBlocks):
    """Direct raw-plane BPZ sewing; name retained for the existing audit.

    Three-point tensors AND the internal Gram matrix are changed. The
    block_map identity is checked against this contraction. This class is
    not certified as the Type0B fixed-parity Human Note block convention.
    """

    def coefficients(self, external=(0, 0, 0, 0), internal_parity=0, left=1, right=1):
        a1, a2, a3, a4 = tuple(map(_bit, external))
        k = _bit(internal_parity)
        lf, rf = (a4+a3+k) % 2, (k+a2+a1) % 2
        result = np.zeros(self.order+1, complex)
        for level in range(self.order+1):
            basis = self.internal.basis(level, k)
            if not basis:
                continue
            if self.family == 'rrrr':
                lv = [bpz_three_point(self.left.rn(_ground(a4), a3, v),
                      'RN', lf, left, bra_ground=a4) for v in basis]
                rv = [bpz_three_point(self.right.nr(v, a2, _ground(a1)),
                      'NR', rf, right) for v in basis]
            elif self.family == 'mixed_ns':
                lv = [ns_ordering_sign((a4, a3, _parity(v)))*
                      self.left.nn(a4, a3, v)[lf] for v in basis]
                rv = [bpz_three_point(self.right.nr(v, a2, _ground(a1)),
                      'NR', rf, right) for v in basis]
            else:
                lv = [bpz_three_point(self.left.nr(_ns(a4), a3, v),
                      'NR', lf, left) for v in basis]
                rv = [bpz_three_point(self.right.rn(v, a2, _ns(a1)),
                      'RN', rf, right, bra_ground=v[1]) for v in basis]
            result[level] = np.asarray(lv) @ np.linalg.solve(
                bpz_gram(self.internal, level, k), np.asarray(rv))
        return result


def sewn_integrand(blocks, constants, P, z, external):
    """Transport the complete native sewing tensor, including external duals.

    This preserves the established physical fields and contour. It verifies
    a change of basis. It does not verify the additional Type0B fixed-parity
    and ordered-slot transport or an absolute heterotic amplitude normalization.
    """
    if not isinstance(blocks, HumanBPZBlocks):
        raise TypeError('HumanBPZBlocks required')
    family = blocks.family
    sectors, pairs = SECTORS[family], sign_pairs(family)
    if len(external) != 4:
        raise ValueError('four external components are required')
    ends = [_components(sectors[i], external[i]) for i in (0, 3)]
    total = 0j
    for ((a1, b1), ket), ((a4, b4), bra) in product(*ends):
        a3, b3 = (0, 0) if sectors[2] == 'R' else tuple(map(_bit, external[2]))
        hol, anti = (a1, 0, a3, a4), (b1, 0, b3, b4)
        for k, kb in product((0, 1), repeat=2):
            lf, rf, blf, brf = (a4+k) % 2, (k+a1) % 2, (b4+kb) % 2, (kb+b1) % 2
            coefficient = np.zeros((len(pairs), len(pairs)), complex)
            for sl, sr in pairs:
                right = ramond_vertex(external[1], sr, rf, brf)
                if not right:
                    continue
                if family == 'mixed_ns':
                    t, bt = (lf+a3) % 2, (blf+b3) % 2
                    if t != bt:
                        continue
                    left = (-1)**(t+t*b3)*(1j if t else 1)
                    density = constants.density(family, blocks.p, P, t, 1, sr)
                else:
                    left = ramond_vertex(external[2], sl, lf, blf)
                    density = constants.density(family, blocks.p, P, 0, sl, sr)
                factor = ket*np.conj(bra)*left*right*(-1)**(brf*a1+blf*k)
                if family == 'mixed_r':
                    factor *= (-1)**(rf+brf)
                # Keep the reflected parity factor together with the sign flip.
                j = pairs.index((sl, -sr if family == 'mixed_r' else sr))
                coefficient[j, j] = factor*density
            if not np.any(coefficient):
                continue
            th = block_map(family, hol, k)
            ta = block_map(family, anti, kb).conjugate()
            changed = transport_coefficient(coefficient, th, ta)
            vh = np.array([blocks.value(z, hol, k, sl, sr) for sl, sr in pairs])
            va = np.array([blocks.value(z, anti, kb, sl, sr) for sl, sr in pairs]).conjugate()
            total += vh @ changed @ va
    return total
