"""Geometric spin data, independent of conformal-block basis conventions.

Characteristics use binary bits [alpha|beta]. Geometry edge order is
(zero, one, infinity). A numerical eta label is never a spin certificate.
"""
from __future__ import annotations

from dataclasses import dataclass
from itertools import product
import cmath
import hashlib
import json
import math
import numpy as np


def _bits(values, length, name):
    values = tuple(values)
    if len(values) != length or any(x not in (0, 1) for x in values):
        raise ValueError(f"{name} must contain {length} binary bits")
    return tuple(int(x) for x in values)


def _integer_matrix(values, shape, name):
    a = np.asarray(values)
    if a.shape != shape or not np.all(np.isfinite(a)) or not np.array_equal(a, np.rint(a)):
        raise ValueError(f"{name} must be an integral {shape[0]} by {shape[1]} matrix")
    return a.astype(np.int64)


@dataclass(frozen=True)
class SpinCharacteristic:
    alpha: tuple[int, int]
    beta: tuple[int, int]

    def __post_init__(self):
        object.__setattr__(self, 'alpha', _bits(self.alpha, 2, 'alpha'))
        object.__setattr__(self, 'beta', _bits(self.beta, 2, 'beta'))

    @classmethod
    def from_pairs(cls, value):
        if len(value) != 2:
            raise ValueError('a characteristic has alpha and beta rows')
        return cls(tuple(value[0]), tuple(value[1]))

    @property
    def pairs(self):
        return self.alpha, self.beta

    @property
    def arf(self):
        return sum(a*b for a, b in zip(self.alpha, self.beta)) % 2

    def quadratic_refinement(self, a_cycles, b_cycles):
        """q(a,b)=a.b+alpha.a+beta.b mod 2 in the marked homology basis."""
        a = _bits(a_cycles, 2, 'a_cycles')
        b = _bits(b_cycles, 2, 'b_cycles')
        return (sum(x*y for x, y in zip(a, b))
                + sum(x*y for x, y in zip(self.alpha, a))
                + sum(x*y for x, y in zip(self.beta, b))) % 2

    def transport(self, matrix):
        """Transport with Omega'=(A Omega+B)(C Omega+D)^(-1)."""
        m = _integer_matrix(matrix, (4, 4), 'symplectic transformation')
        j = np.block([[np.zeros((2, 2), dtype=int), np.eye(2, dtype=int)],
                      [-np.eye(2, dtype=int), np.zeros((2, 2), dtype=int)]])
        if not np.array_equal(m.T@j@m, j):
            raise ValueError('spin transport requires a symplectic matrix')
        a, b, c, d = m[:2, :2], m[:2, 2:], m[2:, :2], m[2:, 2:]
        alpha = (d@self.alpha-c@self.beta+np.diag(c@d.T)) % 2
        beta = (-b@self.alpha+a@self.beta+np.diag(a@b.T)) % 2
        result = type(self)(tuple(alpha), tuple(beta))
        if result.arf != self.arf:
            raise ArithmeticError('spin transport changed the Arf parity')
        return result

    def charge_frame(self, period_branch):
        b = _integer_matrix(period_branch, (2, 2), 'period branch')
        if not np.array_equal(b, b.T):
            raise ValueError('the integer period branch must be symmetric')
        beta = (np.asarray(self.beta)-b@self.alpha+np.diag(b)) % 2
        return type(self)(self.alpha, tuple(beta))

    @property
    def theta_edge_sectors(self):
        a0, a1 = self.alpha
        return tuple('R' if a else 'NS' for a in (a0, a1, a0 ^ a1))

    def all_ns_determinant_lifts(self):
        """Raw Majorana determinant lifts; not literal Human-Note lifts.

        Gauge fixes the infinity lift to +1. This method intentionally
        rejects Ramond sectors: their sewing involves a Clifford module.
        """
        if self.alpha != (0, 0):
            raise ValueError('Ramond spin sewing requires its own verified adapter')
        return ((-1)**self.beta[0], (-1)**self.beta[1], 1)


@dataclass(frozen=True)
class ThetaSpinFrame:
    """A marked spin and the explicit branch of a particular plumbing chart."""
    marked_spin: SpinCharacteristic
    q_values: tuple[complex, complex, complex]
    period_branch: tuple[tuple[int, int], tuple[int, int]]
    edge_order: tuple[str, str, str] = ('zero', 'one', 'infinity')
    propagation: str = 'q^L0'

    def __post_init__(self):
        q = tuple(complex(z) for z in self.q_values)
        if len(q) != 3 or any(not np.isfinite(z) or not 0 < abs(z) < 1 for z in q):
            raise ValueError('three finite nonzero plumbing coordinates with |q|<1 are required')
        b = _integer_matrix(self.period_branch, (2, 2), 'period branch')
        if not np.array_equal(b, b.T):
            raise ValueError('the integer period branch must be symmetric')
        if self.edge_order != ('zero', 'one', 'infinity') or self.propagation != 'q^L0':
            raise ValueError('an explicit adapter is required for a different edge order or propagation convention')
        object.__setattr__(self, 'q_values', q)
        object.__setattr__(self, 'period_branch', tuple(tuple(int(x) for x in row) for row in b))

    @property
    def charge_spin(self):
        return self.marked_spin.charge_frame(self.period_branch)

    def validate_periods(self, marked, charge, *, tolerance=1e-8):
        marked, charge = np.asarray(marked, dtype=complex), np.asarray(charge, dtype=complex)
        for label, omega in (('marked', marked), ('charge', charge)):
            if (omega.shape != (2, 2) or not np.all(np.isfinite(omega))
                    or np.max(abs(omega-omega.T)) > tolerance
                    or np.linalg.eigvalsh(omega.imag)[0] <= 0):
                raise ValueError(f'invalid {label} period matrix')
        error = float(np.max(abs(charge-marked-np.asarray(self.period_branch))))
        if error > tolerance:
            raise ValueError(f'charge/marked period branch mismatch: {error:.6g}')
        return error

    def record(self):
        return {'marked_characteristic': self.marked_spin.pairs,
                'charge_characteristic': self.charge_spin.pairs,
                'arf': self.marked_spin.arf,
                'edge_sectors': self.charge_spin.theta_edge_sectors,
                'edge_order': self.edge_order, 'propagation': self.propagation,
                'q_values': [[z.real, z.imag] for z in self.q_values],
                'period_branch': self.period_branch}

    @property
    def digest(self):
        return hashlib.sha256(json.dumps(self.record(), sort_keys=True).encode()).hexdigest()


def theta_log_windings(reference_branch, current_branch):
    """Infer continued-log minus principal-log windings in a fixed theta chart.

    Omega_principal=Omega_marked+B and the reference logs are principal.
    Charge conservation a_infinity=a_zero+a_one gives the log-period matrix
    [[w_zero+w_infinity, w_infinity], [w_infinity, w_one+w_infinity]].
    It must equal B_reference-B_current along the continued marked family.
    This tracks local log branches, not a path-dependent modular holonomy.
    """
    branches = [_integer_matrix(b, (2, 2), 'period branch')
                for b in (reference_branch, current_branch)]
    if any(not np.array_equal(b, b.T) for b in branches):
        raise ValueError('period branches must be symmetric')
    difference = branches[0]-branches[1]
    return (int(difference[0, 0]-difference[0, 1]),
            int(difference[1, 1]-difference[0, 1]), int(difference[0, 1]))


@dataclass(frozen=True)
class ThetaLogBranch:
    """Continuous plumbing logs, separate from the marked spin characteristic.

    Start at zero windings when the anchor uses principal logs. Advance along
    sufficiently sampled paths; endpoints alone do not specify continuation.
    The resulting root signs describe eta*sqrt(q). They do not prescribe the
    Clifford/BPZ action on Ramond states or identify HJS three-form signs.
    """
    q_values: tuple[complex, complex, complex]
    windings: tuple[int, int, int] = (0, 0, 0)

    def __post_init__(self):
        q = tuple(complex(z) for z in self.q_values)
        if len(q) != 3 or any(not np.isfinite(z) or not 0 < abs(z) < 1 for z in q):
            raise ValueError('three finite nonzero plumbing coordinates with |q|<1 are required')
        w = tuple(self.windings)
        if len(w) != 3 or any(not np.isfinite(x) or int(x) != x for x in w):
            raise ValueError('three integer log windings are required')
        object.__setattr__(self, 'q_values', q)
        object.__setattr__(self, 'windings', tuple(int(x) for x in w))

    @property
    def logs(self):
        return tuple(cmath.log(z)+2j*math.pi*w for z, w in zip(self.q_values, self.windings))

    @property
    def square_roots(self):
        return tuple(cmath.exp(z/2) for z in self.logs)

    @property
    def root_signs(self):
        return tuple(-1 if w % 2 else 1 for w in self.windings)

    @property
    def period_shift(self):
        w0, w1, wi = self.windings
        return ((w0+wi, wi), (wi, w1+wi))

    def advance(self, q_values, *, maximum_phase_step=math.pi/2):
        q = type(self)(q_values).q_values
        steps = tuple(cmath.phase(b/a) for a, b in zip(self.q_values, q))
        if not 0 < maximum_phase_step < math.pi:
            raise ValueError('phase step bound must lie strictly between zero and pi')
        if max(map(abs, steps)) >= maximum_phase_step:
            raise ValueError('sample the plumbing path more finely before continuing its logs')
        w = tuple(round((old.imag+step-cmath.phase(z))/(2*math.pi))
                  for old, step, z in zip(self.logs, steps, q))
        return type(self)(q, w)

    def principal_root_lifts(self, reference_lifts):
        """Represent eta_reference*sqrt_cont(q) using principal square roots."""
        e = tuple(reference_lifts)
        if len(e) != 3 or any(x not in (-1, 1) for x in e):
            raise ValueError('three +/-1 root lifts are required')
        return tuple(int(a*b) for a, b in zip(e, self.root_signs))

    def record(self):
        return {'q_values': [[z.real, z.imag] for z in self.q_values],
                'continuous_logs': [[z.real, z.imag] for z in self.logs],
                'log_windings': self.windings, 'root_signs': self.root_signs,
                'continuous_minus_principal_period': self.period_shift}


ALL_LIFTS = tuple(product((1, -1), repeat=3))
ALL_PARITIES = tuple(product((0, 1), repeat=3))


def human_theta_orientation(parities):
    a, b, c = _bits(parities, 3, 'state parities')
    return (-1)**(a*b+a*c+b*c)


def resolve_lift_components(amplitudes_by_lift):
    """Exact Fourier inversion; input must retain all complex amplitudes."""
    if set(amplitudes_by_lift) != set(ALL_LIFTS):
        raise ValueError('all eight complex lift amplitudes are required; contracted norms cannot be converted')
    return {p: sum(complex(amplitudes_by_lift[e])
                   * np.prod([sign**bit for sign, bit in zip(e, p)]) for e in ALL_LIFTS)/8
            for p in ALL_PARITIES}


def remove_human_theta_quadratic_factor(amplitudes_by_lift, output_lifts):
    """Algebraic basis conversion, not an interacting spin certificate.

    It removes exactly the displayed (-1)^K factor in a complete Human
    theta block. The full physical vertex/pairing adapter must be verified
    separately, including in the Ramond sector.
    """
    lifts = tuple(output_lifts)
    if lifts not in ALL_LIFTS:
        raise ValueError('three +/-1 output lifts are required')
    return sum(value*human_theta_orientation(p)
               * np.prod([sign**bit for sign, bit in zip(lifts, p)])
               for p, value in resolve_lift_components(amplitudes_by_lift).items())


class UnverifiedSpinSewing(ValueError):
    pass


def require_spin_sewing_certificate(certificate, *, frame, theory, implementation_sha256):
    """Block a physical comparison when its boundary prescription is unverified."""
    expected = {'schema': 'spin-sewing-certificate-v1', 'frame_digest': frame.digest,
                'theory': theory, 'implementation_sha256': implementation_sha256,
                'status': 'verified'}
    if not isinstance(certificate, dict) or any(certificate.get(k) != v for k, v in expected.items()):
        raise UnverifiedSpinSewing('A matching verified spin-sewing certificate is required; eta labels and free tests alone are insufficient.')
    required = ('geometric_transport', 'conformal_frame', 'vertex_pairing', 'free_field_limit')
    if any(certificate.get('checks', {}).get(k) is not True for k in required):
        raise UnverifiedSpinSewing('Spin sewing has incomplete geometric, frame, vertex, or free-field validation.')
    return certificate
