"""Two-Virasoro evaluation of the crossing-checked literature blocks.

Public coefficients are in the native 1012.2974 / 0810.1203 convention.
Internally we use a linear BPZ frame, polynomial Ramond grounds, and the
imaginary auxiliary fermion of 1312.4520, equation (3.3). Only the finite
branch highest vectors use PBW algebra; descendant towers use ordinary
Virasoro c-recursion. No Human Note null-factorization formula is used.

The default reproduces literature_component_blocks at c=3. An explicit b
also permits generic central charge and analytic, complex signed momenta.
The native sign frame is continued analytically from the real-momentum
slice; it must not be confused with conjugating physical momenta. The
singular b=1 embedding is handled by literature_self_dual_blocks instead.
"""
from fractions import Fraction
from functools import lru_cache
from itertools import product
import math

import numpy as np
import sympy as sp

from literature_component_blocks import (
    ComponentBlocks, Module, SECTORS, _binom, _bit, _level, _parity,
    _series_power,
)
from ns_algebra.ns_sca import G
from ns_algebra.ns_three_point_tensor import ns_three_point
from ramond_algebra.ramond_sca import PBWState
from ramond_algebra.ns_rr_three_point_tensor import ns_rr_three_point_from_ground_tensor
from spin23_two_virasoro_ramond import (
    AuxiliaryFermionState, TensorBasisState, auxiliary_fermion_mode_action,
    auxiliary_ns_ns_ns_three_point, auxiliary_ns_r_r_three_point,
    embedded_branch_state,
)
from virasoro_sphere_c_recursion import sphere_c_coefficients


# Q=1/sqrt(2), choosing Im(b)>0 once. The embedding routine correlates the
# remaining square roots; signed momentum is never reconstructed from h.
B_C3 = (1 + 1j*math.sqrt(7))/(2*math.sqrt(2))
SIGNS = (1, -1)
_PARAMETERS = sp.symbols("h3 h2 h1 c t00 t01 t10 t11")


class AnalyticModule(Module):
    """Linear SCA action at generic c; no Hermitian Gram continuation."""
    def __init__(self, sector, p, b=B_C3):
        if sector not in ("NS", "R"):
            raise ValueError("sector must be NS or R")
        self.sector, self.p = sector, complex(p)
        b = complex(b)
        if b == 0 or not all(math.isfinite(v) for v in
                             (b.real, b.imag, self.p.real, self.p.imag)):
            raise ValueError("finite momenta and nonzero finite b required")
        q = b + 1/b
        self.c = 1.5 + 3*q*q
        self.h = q*q/8 + self.p*self.p/2 + (1/16 if sector == "R" else 0)
        self.beta = -1j*self.p/math.sqrt(2)
        self.act = lru_cache(None)(self.act)

    def gram(self, *args):
        raise TypeError("a Hermitian Gram matrix cannot analytically continue momenta")

    inverse_gram = gram


def _native_state(state):
    word = state if isinstance(state, tuple) else state.word
    ground = 0 if isinstance(state, tuple) else state.ground_parity
    return tuple((m.kind, int(2*m.index)) for m in word), ground


class _LinearRN:
    """Full ordered RN Ward form, linear in its bra, from source (27).

    This is not obtained by reversing NR. It has four unit component seeds,
    packaged as (++,--)=(1,eta) and (+-,-+)=(1,i*eta).
    """
    def __init__(self, momenta, parity, eta, b=B_C3):
        self.modules = tuple(AnalyticModule(s, p, b) for s, p in
                             zip(("R", "R", "NS"), momenta))
        self.parity, self.eta = parity, eta
        self.value = lru_cache(None)(self.value)

    def _action(self, states, slot, kind, twice_mode):
        answer = 0j
        for target, coefficient in self.modules[slot].act(
                kind, int(twice_mode), states[slot]):
            changed = list(states)
            changed[slot] = target
            answer += coefficient*self.value(*changed)
        return answer

    def value(self, first, middle, last):
        states = (first, middle, last)
        if sum(map(_parity, states)) % 2 != self.parity:
            return 0j
        bound = sum(map(_level, states))//2 + 4
        for slot in (2, 0, 1):
            if not states[slot][0]:
                continue
            (kind, twice_mode) = states[slot][0][0]
            rest = states[slot][0][1:], states[slot][1]
            reduced = list(states)
            reduced[slot] = rest
            reduced = tuple(reduced)
            magnitude = -twice_mode/2
            act = lambda j, k, n: self._action(reduced, j, k, n)
            if kind == "L":
                n = int(magnitude)
                if slot == 0:
                    return act(2, "L", 2*n) + sum(
                        _binom(n+1, j+1)*act(1, "L", 2*j)
                        for j in range(-1, n+1))
                if slot == 2:
                    return act(0, "L", 2*n) - sum(
                        _binom(1-n, j+1)*act(1, "L", 2*j)
                        for j in range(-1, bound))
                if n == 1:
                    h = [m.h+_level(s)/2 for m, s in zip(self.modules, reduced)]
                    return (h[0]-h[1]-h[2])*self.value(*reduced)
                return sum(_binom(n-2+j, n-2)*(
                    act(0, "L", 2*(n+j)) + (-1)**n*act(2, "L", 2*(j-1)))
                    for j in range(bound))

            sign = (-1)**(_parity(reduced[0])+_parity(reduced[2])+1)
            if slot == 2:
                n = int(magnitude-.5)
                leading = 1j*sign*sum(
                    _binom(-n, j)*act(1, "G", 2*j)
                    - _binom(.5, j)*(-1)**j*act(0, "G", 2*(n+j))
                    for j in range(bound))
                return leading - sum(_binom(.5, j)*(-1)**j
                                     * act(2, "G", 2*j+twice_mode)
                                     for j in range(1, bound))
            n = int(magnitude)
            if slot == 0:
                leading = sum(
                    _binom(n, j)*act(1, "G", 2*j)
                    + 1j*sign*_binom(.5, j)*(-1)**j
                    * act(2, "G", 2*(j+n)-1) for j in range(bound))
                return leading - sum(_binom(.5, j)*(-1)**j
                                     * act(0, "G", 2*(j-n))
                                     for j in range(1, bound))
            return sum(_binom(.5-n, j)*(-1)**j*(
                act(0, "G", 2*(j+n))
                - 1j*sign*(-1)**n*act(2, "G", 2*j-1)) for j in range(bound))
        return {(0, 0): 1, (0, 1): 1, (1, 0): 1j*self.eta,
                (1, 1): self.eta}[first[1], middle[1]]


class _AuxiliaryRN:
    """Even ordered Majorana form with f_r^T=-f_-r.

    The weight-1/2 contour is the RN counterpart of the NR fermion contour.
    In its binomial kernels 1/2 is replaced by -1/2. The bra action has
    an additional minus sign. The ground metric is diag(1,-1).
    """
    def __init__(self):
        self.value = lru_cache(None)(self.value)

    def _action(self, states, slot, twice_mode):
        answer = 0j
        for target, coefficient in auxiliary_fermion_mode_action(
                int(twice_mode), states[slot]).items():
            changed = list(states)
            changed[slot] = target
            answer += coefficient*self.value(*changed)
        return answer

    def value(self, first, middle, last):
        states = (first, middle, last)
        if sum(s.parity for s in states) % 2:
            return 0j
        bound = sum(s.twice_level for s in states)//2 + 4
        for slot in (2, 0, 1):
            state = states[slot]
            if not state.twice_modes:
                continue
            twice_mode = state.twice_modes[0]
            reduced = list(states)
            reduced[slot] = AuxiliaryFermionState(
                state.sector, state.twice_modes[1:], state.ground_parity)
            reduced = tuple(reduced)
            sign = (-1)**(reduced[0].parity+reduced[2].parity+1)
            act = lambda j, n: self._action(reduced, j, n)
            if slot == 2:
                n = (twice_mode+1)//2
                leading = -1j*sign*sum(
                    _binom(-n, j)*act(1, 2*j)
                    + _binom(-.5, j)*(-1)**j*act(0, 2*(n+j))
                    for j in range(bound))
                return leading - sum(_binom(-.5, j)*(-1)**j
                                     * act(2, 2*j-twice_mode)
                                     for j in range(1, bound))
            n = twice_mode//2
            if slot == 0:
                leading = -sum(
                    _binom(n, j)*act(1, 2*j)
                    - 1j*sign*_binom(-.5, j)*(-1)**j
                    * act(2, 2*(j+n)+1) for j in range(bound))
                return leading - sum(_binom(-.5, j)*(-1)**j
                                     * act(0, 2*(j-n)) for j in range(1, bound))
            return sum(_binom(-.5-n, j)*(-1)**j*(
                -act(0, 2*(j+n)) + 1j*sign*(-1)**n*act(2, 2*j+1))
                for j in range(bound))
        return 1 if first.ground_parity == 0 else -1


@lru_cache(None)
def _physical_template(order, states):
    h3, h2, h1, c, *ground = _PARAMETERS
    if order == "NR":
        value = ns_rr_three_point_from_ground_tensor(
            *states, h_infinity=h3, h_middle=h2, h_zero=h1, c=c,
            ground_tensor=sp.Matrix(2, 2, ground), factor_result=False)
    elif order == "NN":
        value = ns_three_point(
            *states, h_infinity=h3, h_middle=h2, h_zero=h1, c=c, simplify=False)
    else:
        raise ValueError(order)
    return sp.lambdify(_PARAMETERS, value, modules="numpy", docstring_limit=0)


class BranchVertex:
    """A literature-derived enlarged trinion in literal (infinity,1,0) order."""
    def __init__(self, order, momenta, parity, sign=1, b=B_C3):
        if order not in ("NN", "NR", "RN") or sign not in SIGNS:
            raise ValueError("unsupported ordered trinion or structure sign")
        self.order, self.p, self.parity, self.sign = order, tuple(momenta), _bit(parity), sign
        self.sectors = {"NN": ("NS",)*3, "NR": ("NS", "R", "R"),
                        "RN": ("R", "R", "NS")}[order]
        self.modules = tuple(AnalyticModule(s, p, b) for s, p in zip(self.sectors, self.p))
        self.ground_scales = tuple(m.g0(0) if m.sector == "R" else 1
                                   for m in self.modules)
        self.rn = _LinearRN(self.p, self.parity, sign, b) if order == "RN" else None
        self.auxiliary_rn = _AuxiliaryRN() if order == "RN" else None
        self.elementary = lru_cache(None)(self.elementary)

    def elementary(self, first, middle, last):
        states = first, middle, last
        auxiliary = tuple(s.auxiliary for s in states)
        physical = tuple(s.super_state for s in states)
        native = tuple(_native_state(s) for s in physical)
        mu = tuple(s.parity for s in auxiliary)
        nu = tuple(map(_parity, native))
        if sum(mu) % 2 or sum(nu) % 2 != self.parity:
            return 0j
        if self.order == "RN":
            aux = self.auxiliary_rn.value(*auxiliary)
            if aux == 0:
                return 0j
            physical_value = self.rn.value(*native)*math.prod(
                scale**s[1] for scale, s in zip(self.ground_scales, native))
            exponent = (self.parity+nu[1])*mu[2]
        else:
            if self.order == "NR":
                aux = auxiliary_ns_r_r_three_point(*auxiliary, form_parity=0)
                g2, g1 = self.ground_scales[1:]
                terminal = (1, g1, 1j*self.sign*g2, self.sign*g2*g1)
                # The NR helper uses a positive bra-contour mode. (-1)^mu3
                # converts it to the auxiliary f^T=-f convention before sewing.
                exponent = (self.parity+nu[1])*mu[2]+mu[0]
            else:
                aux = auxiliary_ns_ns_ns_three_point(*auxiliary)
                terminal = (0,)*4
                exponent = mu[0]*nu[1]+self.parity*(mu[0]+mu[1])
            if aux == 0:
                return 0j
            physical_value = _physical_template(self.order, physical)(
                *(m.h for m in self.modules), self.modules[0].c, *terminal)
        return complex((-1)**exponent*aux*physical_value)

    def evaluate_vectors(self, vectors):
        return sum(a*b*c*self.elementary(x, y, z)
                   for (x, a), (y, b), (z, c) in
                   product(*(vector.items() for vector in vectors)))

    def branch_value(self, first, middle, last):
        return self.evaluate_vectors(tuple(dict(zip(v.basis, v.coefficients))
                                           for v in (first, middle, last)))


def branch_state(sector, momentum, number, parity=None, b=B_C3):
    """Translate the embedding's P=i*p to the literature beta=-i*p/sqrt(2)."""
    return embedded_branch_state(b=complex(b), sector=sector,
                                 physical_momentum=-complex(momentum),
                                 branch_number=Fraction(number), parity=parity)


@lru_cache(None)
def external_resolution(sector, momentum, component, b=B_C3):
    component = _bit(component)
    if sector == "NS":
        labels = (Fraction(0),) if not component else (Fraction(-1, 2), Fraction(1, 2))
        target = TensorBasisState(AuxiliaryFermionState("NS"),
                                 (G(Fraction(-1, 2)),) if component else ())
    else:
        labels = Fraction(-1, 4), Fraction(1, 4)
        target = TensorBasisState(AuxiliaryFermionState("R"), PBWState((), component))
    branches = tuple(branch_state(sector, momentum, n, component, b) for n in labels)
    basis = branches[0].basis
    if any(v.basis != basis for v in branches):
        raise AssertionError("incompatible external branch bases")
    matrix = np.array([v.coefficients for v in branches]).T
    vector = np.array([s == target for s in basis], complex)
    coefficients = np.linalg.solve(matrix, vector)
    if np.linalg.norm(matrix @ coefficients-vector) > 1e-11:
        raise ArithmeticError("external component resolution failed")
    return tuple(zip(coefficients, branches))


def branch_numbers(sector, twice_level, parity):
    if sector == "NS":
        bound = math.isqrt(twice_level)
        return tuple(Fraction(j, 2) for j in range(-bound, bound+1)
                     if j % 2 == parity)
    positive = [Fraction(2*j+1, 4) for j in range(math.isqrt(twice_level)+1)
                if j*(j+1) <= twice_level]
    return tuple(sorted([-n for n in positive]+positive))


@lru_cache(None)
def _virasoro_product(internal, external, order):
    factors = [sphere_c_coefficients(
        c=getattr(internal, "c_"+str(j)), h=getattr(internal, "h_"+str(j)),
        external_weights=tuple(getattr(v, "h_"+str(j)) for v in external),
        order=order) for j in (1, 2)]
    return np.convolve(*factors)[:order+1]


@lru_cache(None)
def auxiliary_series(family, order):
    """Strip the auxiliary leading z power; keep its actual sphere block."""
    if family not in SECTORS:
        raise ValueError(family)
    one_minus_z = np.zeros(order+1, complex)
    one_minus_z[0] = 1
    if order:
        one_minus_z[1] = -1
    if family == "mixed_ns":
        result = np.zeros(order+1, complex)
        result[0] = 1
        return result
    result = _series_power(one_minus_z, -1/8)
    if family == "rrrr":
        # Identity-channel Ising four-spin block, divided by z^(-1/8).
        inner = _series_power(one_minus_z, .5)/2
        inner[0] += .5
        result = np.convolve(result, _series_power(inner, .5))[:order+1]
    return result


def linear_block_map(family, external, parity):
    """Native antilinear/sign frame -> linear unit-seed frame, no Ward change."""
    a1, a2, a3, a4 = external
    matrix = np.array([[1-1j, 1+1j], [1+1j, 1-1j]])/2
    if family == "mixed_ns":
        return np.eye(2)
    f = (parity+a2+a1) % 2 if family == "mixed_r" else (a4+a3+parity) % 2
    if f:
        matrix = np.diag([1, -1]) @ matrix
    return (np.kron(np.eye(2), matrix) if family == "mixed_r"
            else 1j**a4*np.kron(matrix, np.eye(2)))


class LiteratureDoubleVirasoroBlocks(ComponentBlocks):
    """Drop-in literature block evaluator using finite branching and c-recursion."""
    def __init__(self, family, momenta, P, maximum_twice_level=8, *, b=B_C3):
        if family not in SECTORS or len(momenta) != 4:
            raise ValueError("unknown channel or non-four-point data")
        if (not isinstance(maximum_twice_level, int)
                or isinstance(maximum_twice_level, bool) or maximum_twice_level < 0):
            raise ValueError("a nonnegative integral truncation is required")
        self.b = complex(b)
        if self.b in (0, 1, -1):
            raise ValueError("singular embedding: use the assembled self-dual limit for b=1")
        self.family, self.p, self.order = family, tuple(map(complex, momenta)), maximum_twice_level
        self.modules = tuple(AnalyticModule(s, p, b) for s, p in zip(SECTORS[family], self.p))
        self.internal = AnalyticModule("R" if family == "mixed_r" else "NS", P, b)
        if any(m.sector == "R" and m.p == 0 for m in (*self.modules, self.internal)):
            raise ValueError("Ramond zero momentum requires a separate branch limit")
        self.bpz_coefficients = lru_cache(None)(self.bpz_coefficients)
        self._native_components = lru_cache(None)(self._native_components)
        for name in ("coefficients", "elliptic_transform", "elliptic_coefficients", "value"):
            setattr(self, name, lru_cache(None)(getattr(self, name)))

    def prefactor_exponents(self, external):
        h = [m.h + (a/2 if m.sector == "NS" else 0)
             for m, a in zip(self.modules, external)]
        base = (self.internal.c-1.5)/24
        return (base-h[0]-h[1]+(1/16 if self.family == "mixed_r" else 0),
                base-h[1]-h[2]+(1/16 if self.family == "mixed_ns" else 0),
                12*base-4*sum(h)+(.5 if self.family != "rrrr" else 0))

    def bpz_coefficients(self, external=(0, 0, 0, 0), internal_parity=0, left=1, right=1):
        external = tuple(map(_bit, external))
        if len(external) != 4 or left not in SIGNS or right not in SIGNS:
            raise ValueError("four component bits and two structure signs required")
        a1, a2, a3, a4 = external
        k = _bit(internal_parity)
        p1, p2, p3, p4 = self.p
        P = self.internal.p
        lf, rf = (a4+a3+k) % 2, (k+a2+a1) % 2
        orders = {"mixed_ns": ("NN", "NR"), "mixed_r": ("NR", "RN"),
                  "rrrr": ("RN", "NR")}[self.family]
        lv = BranchVertex(orders[0], (p4, p3, P), lf, left, self.b)
        rv = BranchVertex(orders[1], (P, p2, p1), rf, right, self.b)
        resolutions = tuple(external_resolution(s, p, a, self.b)
                            for s, p, a in zip(SECTORS[self.family], self.p, external))
        enlarged = np.zeros(self.order+1, complex)
        for number in branch_numbers(self.internal.sector, self.order, k):
            internal = branch_state(self.internal.sector, P, number, k, self.b)
            onset = int(4*number*number
                        - (Fraction(1, 4) if self.internal.sector == "R" else 0))
            for terms in product(*resolutions):
                zero, moving, one, infinity = (v for _, v in terms)
                weight = math.prod(d for d, _ in terms)
                weight *= (lv.branch_value(infinity, one, internal)
                           * rv.branch_value(internal, moving, zero)/internal.norm)
                if abs(weight) == 0:
                    continue
                descendants = _virasoro_product(
                    internal.parameters, tuple(v.parameters for _, v in terms),
                    (self.order-onset)//2)
                for level, value in enumerate(descendants):
                    enlarged[onset+2*level] += weight*value
        # The embedding uses |->poly=G0|+>=g+(p)|w-> on every R leg.
        enlarged /= math.prod(m.g0(0)**a for m, a in zip(self.modules, external)
                              if m.sector == "R")
        auxiliary = auxiliary_series(self.family, self.order//2)
        physical = enlarged.copy()
        for level in range(self.order+1):
            physical[level] -= sum(auxiliary[j]*physical[level-2*j]
                                   for j in range(1, level//2+1))
        return physical

    def _native_components(self, external, parity):
        pairs = tuple(product((1,) if self.family == "mixed_ns" else SIGNS, SIGNS))
        changed = np.array([self.bpz_coefficients(external, parity, a, b)
                            for a, b in pairs])
        return np.linalg.solve(linear_block_map(self.family, external, parity), changed)

    def coefficients(self, external=(0, 0, 0, 0), internal_parity=0, left=1, right=1):
        external = tuple(map(_bit, external))
        k = _bit(internal_parity)
        if len(external) != 4 or left not in SIGNS or right not in SIGNS:
            raise ValueError("four component bits and two structure signs required")
        pairs = tuple(product((1,) if self.family == "mixed_ns" else SIGNS, SIGNS))
        key = (1 if self.family == "mixed_ns" else left, right)
        return self._native_components(external, k)[pairs.index(key)]
