#!/usr/bin/env python3
"""Direct finite-level sewing for the genus-two NRR theta block.

This module is deliberately independent of a Kac determinant or a
Zamolodchikov recursion.  It combines

* one NS Verma module;
* two generic long-R Verma modules, with their two-dimensional ground
  fibres kept open;
* the full R-R-NS three-descendant Ward identities; and
* direct Gram-matrix contraction of two theta-graph trinions.

The Ramond states use the unnormalised ground basis

    e_+ = w^+,  e_- = G_0 w^+.

For the HJS three-form labelled by ``sign=+/-1``, the primary ground matrix
is either ``diag(1, sign*kappa_left*kappa_right)`` (even form) or has
off-diagonal entries ``(kappa_right, i*sign*kappa_left)`` (odd form).  The
Ward recursion below is the R-R form in section 2 of Suchanek,
arXiv:1012.2974, supplemented by the sector-independent Virasoro Ward
identities.  All values are evaluated in the standard ``(infinity,1,0)``
frame.

Levels are represented as twice-levels on the NS edge and ordinary integer
levels on the two R edges.  ``max_total_level`` in the command-line report is
the total physical plumbing level

    ns_twice_level/2 + r1_level + r2_level.
"""

from __future__ import annotations

import argparse
import cmath
from dataclasses import asdict, dataclass
from functools import lru_cache
import json
import math
from typing import Iterable, Sequence

import numpy as np

from mixed_ns_ramond_descendant_blocks import (
    NSVermaModule,
    State as NSState,
    state_twice_level,
)
from ns_regular_block import THETA_ORIENTATION
from ramond_descendant_blocks import (
    RamondVermaModule,
    State as RState,
    state_level as r_state_level,
    state_parity as r_state_parity,
)


def generalized_binomial(value: complex, order: int) -> complex:
    """Return the generalized binomial coefficient ``(value choose order)``."""

    if order < 0:
        return 0.0 + 0.0j
    result = 1.0 + 0.0j
    for offset in range(order):
        result *= (complex(value) - offset) / (offset + 1)
    return result


def _is_r_ground(state: RState) -> bool:
    return state in ((), (("G", 0),))


def _r_ground_index(state: RState) -> int:
    if state == ():
        return 0
    if state == (("G", 0),):
        return 1
    raise ValueError(f"expected a Ramond ground state, got {state!r}")


def _has_r_creation(state: RState) -> bool:
    return any(index < 0 for _, index in state)


class RRNSDescendantThreeForm:
    """Full R-R-NS trilinear form in the ``(infinity,1,0)`` frame."""

    def __init__(
        self,
        *,
        c: complex,
        left_weight: complex,
        ns_weight: complex,
        right_weight: complex,
        sign: int,
        form_parity: int = 0,
    ) -> None:
        self.c = complex(c)
        self.weights = (
            complex(left_weight),
            complex(ns_weight),
            complex(right_weight),
        )
        self.sign = int(sign)
        if self.sign not in (-1, 1):
            raise ValueError("sign must be +1 or -1")
        self.form_parity = int(form_parity)
        if self.form_parity not in (0, 1):
            raise ValueError("form_parity must be 0 or 1")
        self.left_module = RamondVermaModule(
            c=self.c, weight=self.weights[0]
        )
        self.ns_module = NSVermaModule(c=self.c, weight=self.weights[1])
        self.right_module = RamondVermaModule(
            c=self.c, weight=self.weights[2]
        )
        if (
            abs(self.left_module.kappa_squared) == 0.0
            or abs(self.right_module.kappa_squared) == 0.0
        ):
            raise ValueError("the direct NRR block requires generic long-R modules")
        self.left_kappa = cmath.sqrt(self.left_module.kappa_squared)
        self.right_kappa = cmath.sqrt(self.right_module.kappa_squared)

    @property
    def ground_bottom(self) -> tuple[tuple[complex, ...], ...]:
        if self.form_parity == 1:
            return (
                (0.0j, self.right_kappa),
                (1j * self.sign * self.left_kappa, 0.0j),
            )
        return (
            (1.0 + 0.0j, 0.0j),
            (
                0.0j,
                self.sign * self.left_kappa * self.right_kappa,
            ),
        )

    def _r_action(
        self,
        *,
        slot: int,
        mode: tuple[str, int],
        left: RState,
        middle: NSState,
        right: RState,
    ) -> complex:
        module = self.left_module if slot == 0 else self.right_module
        state = left if slot == 0 else right
        result = 0.0 + 0.0j
        for acted, coefficient in module.mode_action(mode, state).items():
            if slot == 0:
                result += coefficient * self.value(acted, middle, right)
            else:
                result += coefficient * self.value(left, middle, acted)
        return result

    def _ns_action(
        self,
        *,
        mode: tuple[str, int],
        left: RState,
        middle: NSState,
        right: RState,
    ) -> complex:
        result = 0.0 + 0.0j
        for acted, coefficient in self.ns_module.mode_action(mode, middle).items():
            result += coefficient * self.value(left, acted, right)
        return result

    @staticmethod
    def _state_weight(primary: complex, level: float) -> complex:
        return primary + level

    @lru_cache(maxsize=None)
    def value(
        self,
        left: RState,
        middle: NSState,
        right: RState,
    ) -> complex:
        """Return ``rho_RR(left,middle,right|1)`` from the Ward identities."""

        # First remove creation modes from the bra R leg.  This ordering is
        # the one used by the independently tested two-R-leg Ward matrix.
        if _has_r_creation(left):
            creation_index = max(
                index for index, item in enumerate(left) if item[1] < 0
            )
            mode = left[creation_index]
            if mode[1] >= 0:
                raise ValueError(f"unexpected Ramond PBW word {left!r}")
            tail = left[:creation_index] + left[creation_index + 1 :]
            n = -mode[1]
            if mode[0] == "L":
                result = self._r_action(
                    slot=2,
                    mode=("L", n),
                    left=tail,
                    middle=middle,
                    right=right,
                )
                for m in range(-1, n + 1):
                    result += math.comb(n + 1, m + 1) * self._ns_action(
                        mode=("L", 2 * m),
                        left=tail,
                        middle=middle,
                        right=right,
                    )
                return result

            parity_sign = (-1) ** (
                r_state_parity(tail) + r_state_parity(right) + 1
            )
            result = parity_sign * self._r_action(
                slot=2,
                mode=("G", n),
                left=tail,
                middle=middle,
                right=right,
            )
            # k=-1/2,...,n+1/2, or j=k+1/2=0,...,n+1.
            for j in range(n + 2):
                k_twice = 2 * j - 1
                result += generalized_binomial(n + 0.5, j) * self._ns_action(
                    mode=("G", k_twice),
                    left=tail,
                    middle=middle,
                    right=right,
                )
            return result

        # Next remove the NS word.  Virasoro identities are sector
        # independent; the supercurrent identity is the second R-R Ward
        # identity of arXiv:1012.2974 at z=1.
        if middle:
            mode = middle[0]
            if mode[1] >= 0:
                raise ValueError(f"unexpected NS PBW word {middle!r}")
            tail = middle[1:]
            if mode[0] == "L":
                n = -mode[1] // 2
                if n == 1:
                    exponent = (
                        self._state_weight(self.weights[0], 0.0)
                        - self._state_weight(
                            self.weights[1], state_twice_level(tail) / 2.0
                        )
                        - self._state_weight(
                            self.weights[2], r_state_level(right)
                        )
                    )
                    return exponent * self.value(left, tail, right)
                if n < 1:
                    raise ValueError("expected a negative Virasoro mode")
                # The nominally infinite bra sum starts with L_n.  The bra is
                # already in its ground fibre, so this term and all later
                # positive modes vanish.  Keeping L_n explicit makes the Ward
                # identity and its truncation transparent.
                result = self._r_action(
                    slot=0,
                    mode=("L", n),
                    left=left,
                    middle=tail,
                    right=right,
                )
                max_right = r_state_level(right) + 1
                for m in range(max_right + 1):
                    coefficient = generalized_binomial(n - 2 + m, n - 2)
                    result += ((-1) ** n) * coefficient * self._r_action(
                        slot=2,
                        mode=("L", m - 1),
                        left=left,
                        middle=tail,
                        right=right,
                    )
                return result

            r_twice = -mode[1]
            if r_twice <= 0 or r_twice % 2 != 1:
                raise ValueError("expected a negative half-integral NS G mode")
            r = r_twice / 2.0
            result = 0.0 + 0.0j

            # Move the p>=1 terms on the left side to the right.  Their
            # target middle level is the original level minus p.
            max_middle_p = state_twice_level(middle) // 2
            for p in range(1, max_middle_p + 1):
                coefficient = generalized_binomial(0.5, p)
                result -= coefficient * self._ns_action(
                    mode=("G", int(round(2.0 * (p - r)))),
                    left=left,
                    middle=tail,
                    right=right,
                )

            top = 0.5 - r
            # With the bra already reduced to its ground fibre, only the
            # zero-mode term (r=1/2,p=0) can survive.
            for p in range(1):
                coefficient = generalized_binomial(top, p) * ((-1) ** p)
                result += coefficient * self._r_action(
                    slot=0,
                    mode=("G", int(round(p + r - 0.5))),
                    left=left,
                    middle=tail,
                    right=right,
                )

            parity_sign = (-1) ** (
                r_state_parity(left) + r_state_parity(right) + 1
            )
            max_right_p = r_state_level(right)
            for p in range(max_right_p + 1):
                coefficient = generalized_binomial(top, p)
                power_sign = (-1) ** int(round(0.5 - r - p))
                result -= parity_sign * coefficient * power_sign * self._r_action(
                    slot=2,
                    mode=("G", p),
                    left=left,
                    middle=tail,
                    right=right,
                )
            return result

        # Finally remove creation modes from the ket R leg.  These are the
        # R-R field-action identities with the bra in its ground fibre.
        if _has_r_creation(right):
            mode = right[0]
            if mode[1] >= 0:
                raise ValueError(f"unexpected Ramond PBW word {right!r}")
            tail = right[1:]
            n = -mode[1]
            if mode[0] == "L":
                coefficient = (
                    self.weights[2]
                    + r_state_level(tail)
                    - self.weights[0]
                    + n * self.weights[1]
                )
                return coefficient * self.value(left, middle, tail)

            # rho(bottom,G_-n tail) = -rho(G_-1/2 bottom,tail).
            return -self.value(left, (("G", -1),), tail)

        if middle:
            raise AssertionError("middle recursion did not terminate")
        if not _is_r_ground(left) or not _is_r_ground(right):
            raise AssertionError("Ramond recursion did not terminate")
        return self.ground_bottom[_r_ground_index(left)][_r_ground_index(right)]


def theta_orientation_sign(parities: Sequence[int]) -> int:
    if len(parities) != 3:
        raise ValueError("theta sewing has three edges")
    return THETA_ORIENTATION.sign(tuple(int(value) % 2 for value in parities))


class DirectNRRThetaOracle:
    """Direct finite-c NRR theta-sewing coefficients."""

    def __init__(
        self,
        *,
        c: complex,
        h_ns: complex,
        beta_1: complex,
        beta_2: complex,
        signs: Sequence[int] = (1, 1),
        lifts: Sequence[int] = (1, 1, 1),
        form_parity: int = 0,
        normalize_ground: bool = True,
    ) -> None:
        self.c = complex(c)
        self.h_ns = complex(h_ns)
        self.betas = (complex(beta_1), complex(beta_2))
        self.r_weights = tuple(self.c / 24.0 - beta**2 for beta in self.betas)
        self.signs = tuple(int(value) for value in signs)
        if len(self.signs) != 2 or any(value not in (-1, 1) for value in self.signs):
            raise ValueError("signs must contain the two HJS form signs")
        self.lifts = tuple(int(value) for value in lifts)
        if len(self.lifts) != 3 or any(value not in (-1, 1) for value in self.lifts):
            raise ValueError("lifts must contain the NS, R1, and R2 signs")
        self.form_parity = int(form_parity)
        if self.form_parity not in (0, 1):
            raise ValueError("form_parity must be 0 or 1")
        self.ns_module = NSVermaModule(c=self.c, weight=self.h_ns)
        self.r_modules = tuple(
            RamondVermaModule(c=self.c, weight=weight)
            for weight in self.r_weights
        )
        self.forms = tuple(
            RRNSDescendantThreeForm(
                c=self.c,
                left_weight=self.r_weights[0],
                ns_weight=self.h_ns,
                right_weight=self.r_weights[1],
                sign=sign,
                form_parity=self.form_parity,
            )
            for sign in self.signs
        )
        leading = self.evaluate_components(self.coefficient_components(0, 0, 0))
        if normalize_ground and abs(leading) == 0.0:
            raise ValueError(
                "the selected HJS forms and R lifts have zero ground coefficient"
            )
        self.normalization = leading if normalize_ground else 1.0

    def evaluate_components(self, components: Sequence[complex]) -> complex:
        """Evaluate an eight-component parity vector at the plumbing lifts.

        Component bits are ordered as ``(NS, R1, R2)``.  The theta-graph
        quadratic orientation sign is already included in each component.
        The canonical zero-puncture Ramond frame contributes the fixed minus
        sign multiplying the user-visible third lift.
        """

        if len(components) != 8:
            raise ValueError("theta parity data must have eight components")
        answer = 0.0 + 0.0j
        for index, value in enumerate(components):
            answer += (
                complex(value)
                * self.lifts[0] ** (index & 1)
                * self.lifts[1] ** ((index >> 1) & 1)
                * (-self.lifts[2]) ** ((index >> 2) & 1)
            )
        return complex(answer)

    @lru_cache(maxsize=None)
    def ns_basis(self, twice_level: int) -> tuple[NSState, ...]:
        return tuple(self.ns_module.basis(int(twice_level)))

    @lru_cache(maxsize=None)
    def r_basis(self, edge: int, level: int, parity: int) -> tuple[RState, ...]:
        return tuple(self.r_modules[int(edge)].basis(int(level), int(parity)))

    @lru_cache(maxsize=None)
    def ns_inverse(self, twice_level: int) -> np.ndarray:
        return np.linalg.inv(
            np.asarray(self.ns_module.gram_matrix(int(twice_level)), dtype=np.complex128)
        )

    @lru_cache(maxsize=None)
    def r_inverse(self, edge: int, level: int, parity: int) -> np.ndarray:
        return np.linalg.inv(
            np.asarray(
                self.r_modules[int(edge)].gram_matrix(int(level), int(parity)),
                dtype=np.complex128,
            )
        )

    @lru_cache(maxsize=None)
    def raw_coefficient_components(
        self, ns_twice_level: int, r1_level: int, r2_level: int
    ) -> tuple[complex, ...]:
        """Return open PBW contractions before the theta orientation sign.

        The component index is ``p_NS + 2*p_R1 + 4*p_R2``.  Only components
        satisfying ``p_NS+p_R1+p_R2=form_parity (mod 2)`` can be nonzero.
        """

        ns_level = int(ns_twice_level)
        levels_r = (int(r1_level), int(r2_level))
        ns_basis = self.ns_basis(ns_level)
        ns_parity = ns_level % 2
        inverse_ns = self.ns_inverse(ns_level)
        components = [0.0j] * 8
        for parity_1 in (0, 1):
            parity_2 = (self.form_parity + ns_parity + parity_1) % 2
            basis_1 = self.r_basis(0, levels_r[0], parity_1)
            basis_2 = self.r_basis(1, levels_r[1], parity_2)
            inverse_1 = self.r_inverse(0, levels_r[0], parity_1)
            inverse_2 = self.r_inverse(1, levels_r[1], parity_2)
            tensors = []
            for form in self.forms:
                tensor = np.empty(
                    (len(basis_1), len(ns_basis), len(basis_2)),
                    dtype=np.complex128,
                )
                for i, state_1 in enumerate(basis_1):
                    for j, state_ns in enumerate(ns_basis):
                        for k, state_2 in enumerate(basis_2):
                            tensor[i, j, k] = form.value(
                                state_1, state_ns, state_2
                            )
                tensors.append(tensor)
            contracted = np.einsum(
                "abc,ad,be,cf,def->",
                tensors[0],
                inverse_1,
                inverse_ns,
                inverse_2,
                tensors[1],
                optimize=True,
            )
            component_index = ns_parity | (parity_1 << 1) | (parity_2 << 2)
            components[component_index] += contracted
        return tuple(complex(value) for value in components)

    @lru_cache(maxsize=None)
    def coefficient_components(
        self, ns_twice_level: int, r1_level: int, r2_level: int
    ) -> tuple[complex, ...]:
        """Return parity components with the Human-Note theta sign included."""

        raw = self.raw_coefficient_components(
            int(ns_twice_level), int(r1_level), int(r2_level)
        )
        return tuple(
            theta_orientation_sign(
                ((index & 1), ((index >> 1) & 1), ((index >> 2) & 1))
            )
            * value
            for index, value in enumerate(raw)
        )

    @lru_cache(maxsize=None)
    def coefficient(self, ns_twice_level: int, r1_level: int, r2_level: int) -> complex:
        components = self.coefficient_components(
            int(ns_twice_level), int(r1_level), int(r2_level)
        )
        return complex(self.evaluate_components(components) / self.normalization)


@dataclass(frozen=True)
class AnalyticRamondResidueCheck:
    """Algebraic level-one R-edge residue comparison.

    ``direct`` is obtained from the exact singular part of the level-one
    inverse Gram matrix,

        Res B(c)^-1 = v v^T / (v^T B'(c_*) v),

    and therefore involves no displacement from the Kac pole.  ``predicted``
    is the fixed-beta Zamolodchikov kernel assembled from the published
    inverse null norm, the c-plane Jacobian, and the two RRNS fusion
    polynomials.
    """

    branch: int
    pole: complex
    direct: complex
    predicted: complex

    @property
    def absolute_error(self) -> float:
        return float(abs(self.direct - self.predicted))

    @property
    def relative_error(self) -> float:
        return float(
            abs(self.direct - self.predicted)
            / max(abs(self.direct), abs(self.predicted), 1.0e-300)
        )


def _level_one_inverse_gram_residue(
    *,
    pole: complex,
    beta: complex,
    parity: int,
) -> np.ndarray:
    """Return the algebraic fixed-beta residue of ``B_R(1,parity)^-1``.

    At fixed beta, ``h_R=c/24-beta**2`` and every level-one Gram entry is
    affine in ``c``.  The centered unit-step difference below is consequently
    the exact polynomial derivative (up to floating-point roundoff), not a
    finite-displacement estimate of the inverse-Gram pole.
    """

    def gram(c_value: complex) -> np.ndarray:
        module = RamondVermaModule(
            c=c_value,
            weight=c_value / 24.0 - beta**2,
        )
        return np.asarray(
            module.gram_matrix(1, int(parity)), dtype=np.complex128
        )

    singular = gram(pole)
    derivative = (gram(pole + 1.0) - gram(pole - 1.0)) / 2.0
    if singular.shape != (2, 2):
        raise AssertionError("the long-R level-one parity block must be 2x2")

    # For a symmetric 2x2 matrix at determinant zero, either (b,-a) or
    # (d,-b) is a right null vector.  Choose the better-scaled expression.
    a, b = singular[0]
    _, d = singular[1]
    candidates = (
        np.asarray((b, -a), dtype=np.complex128),
        np.asarray((d, -b), dtype=np.complex128),
    )
    null = min(
        candidates,
        key=lambda vector: np.linalg.norm(singular @ vector)
        / max(np.linalg.norm(vector), 1.0e-300),
    )
    denominator = null.T @ derivative @ null
    if abs(denominator) == 0.0:
        raise ZeroDivisionError("the level-one Kac zero is not simple")
    return np.outer(null, null) / denominator


def analytic_first_r_kac_residue_checks(
    *,
    h_ns: complex,
    beta_1: complex,
    beta_2: complex,
    signs: Sequence[int] = (1, 1),
    lifts: Sequence[int] = (1, 1, 1),
) -> tuple[AnalyticRamondResidueCheck, ...]:
    """Compare the two level-one ``(2,1)`` R1 residues algebraically.

    This is the preferred generic-long-module certificate for the local
    fixed-beta R kernel.  The equal even-form theta component is kept in its
    full two-state ``G_0`` ground fibre during the direct contraction.
    """

    from mixed_ramond_sphere_blocks import (
        _r_a_beta,
        _r_ns_fusion_polynomial,
    )
    from ramond_fixed_beta_c_recursion import ramond_c_poles

    signs = tuple(int(value) for value in signs)
    lifts = tuple(int(value) for value in lifts)
    if len(signs) != 2 or len(lifts) != 3:
        raise ValueError("expected two HJS signs and three plumbing lifts")
    if signs[0] != signs[1] or lifts != (1, 1, 1):
        raise NotImplementedError(
            "the analytic scalar projection is currently convention-locked "
            "to equal HJS forms and identity plumbing lifts"
        )

    checks = []
    for pole_data in ramond_c_poles(beta_1, 2, 1):
        oracle = DirectNRRThetaOracle(
            c=pole_data.c,
            h_ns=h_ns,
            beta_1=beta_1,
            beta_2=beta_2,
            signs=signs,
            lifts=lifts,
        )
        direct = 0.0 + 0.0j
        for parity_1 in (0, 1):
            parity_2 = parity_1
            basis_1 = oracle.r_basis(0, 1, parity_1)
            basis_2 = oracle.r_basis(1, 0, parity_2)
            residue_1 = _level_one_inverse_gram_residue(
                pole=pole_data.c,
                beta=beta_1,
                parity=parity_1,
            )
            inverse_2 = oracle.r_inverse(1, 0, parity_2)
            matrices = tuple(
                np.asarray(
                    [
                        [form.value(left, (), right) for right in basis_2]
                        for left in basis_1
                    ],
                    dtype=np.complex128,
                )
                for form in oracle.forms
            )
            contraction = np.einsum(
                "ab,ac,bd,cd->",
                matrices[0],
                residue_1,
                inverse_2,
                matrices[1],
                optimize=True,
            )
            direct += (
                theta_orientation_sign((parity_1, 0, parity_2))
                * lifts[1] ** parity_1
                * (-lifts[2]) ** parity_2
                * contraction
            )
        direct /= oracle.normalization

        fusion_left = _r_ns_fusion_polynomial(
            b=pole_data.b,
            r=2,
            s=1,
            ramond_beta_value=beta_2,
            ns_weight=h_ns,
            sign=signs[0],
        )
        fusion_right = _r_ns_fusion_polynomial(
            b=pole_data.b,
            r=2,
            s=1,
            ramond_beta_value=beta_2,
            ns_weight=h_ns,
            sign=signs[1],
        )
        predicted = (
            -_r_a_beta(pole_data.b, 2, 1, 1.0e-12)
            * fusion_left
            * fusion_right
            / pole_data.derivative_beta_c
        )
        checks.append(
            AnalyticRamondResidueCheck(
                branch=pole_data.branch,
                pole=pole_data.c,
                direct=complex(direct),
                predicted=complex(predicted),
            )
        )
    return tuple(checks)


def level_triples(max_total_level: int) -> Iterable[tuple[int, int, int]]:
    """Yield ``(ns_twice_level,r1_level,r2_level)`` up to total level."""

    cutoff = 2 * int(max_total_level)
    for total_twice in range(cutoff + 1):
        for ns_twice in range(total_twice + 1):
            remainder = total_twice - ns_twice
            if remainder % 2:
                continue
            for r1 in range(remainder // 2 + 1):
                r2 = remainder // 2 - r1
                yield ns_twice, r1, r2


@dataclass(frozen=True)
class CoefficientRecord:
    ns_twice_level: int
    r1_level: int
    r2_level: int
    total_level: float
    real: float
    imag: float


def direct_ledger(
    *,
    c: complex,
    h_ns: complex,
    beta_1: complex,
    beta_2: complex,
    max_total_level: int,
    signs: Sequence[int] = (1, 1),
    lifts: Sequence[int] = (1, 1, 1),
) -> tuple[CoefficientRecord, ...]:
    oracle = DirectNRRThetaOracle(
        c=c,
        h_ns=h_ns,
        beta_1=beta_1,
        beta_2=beta_2,
        signs=signs,
        lifts=lifts,
    )
    records = []
    for ns_twice, r1, r2 in level_triples(max_total_level):
        value = oracle.coefficient(ns_twice, r1, r2)
        records.append(
            CoefficientRecord(
                ns_twice_level=ns_twice,
                r1_level=r1,
                r2_level=r2,
                total_level=ns_twice / 2.0 + r1 + r2,
                real=float(value.real),
                imag=float(value.imag),
            )
        )
    return tuple(records)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Direct finite-level genus-two NRR theta block"
    )
    parser.add_argument("--c", type=float, default=37.25)
    parser.add_argument("--h-ns", type=float, default=0.73)
    parser.add_argument("--betas", nargs=2, type=float, default=(0.67, 0.83))
    parser.add_argument("--signs", nargs=2, type=int, default=(1, 1))
    parser.add_argument("--lifts", nargs=3, type=int, default=(1, 1, 1))
    parser.add_argument("--max-total-level", type=int, default=2)
    args = parser.parse_args()
    records = direct_ledger(
        c=args.c,
        h_ns=args.h_ns,
        beta_1=args.betas[0],
        beta_2=args.betas[1],
        max_total_level=args.max_total_level,
        signs=args.signs,
        lifts=args.lifts,
    )
    output = {
        "convention": {
            "graph": "NRR theta",
            "cutoff": "ns_level + r1_level + r2_level <= max_total_level",
            "ramond_ground_basis": ["w+", "G0 w+"],
            "ground_normalization": "selected nonzero ground sewing divided out",
        },
        "parameters": {
            "c": args.c,
            "h_ns": args.h_ns,
            "beta_1": args.betas[0],
            "beta_2": args.betas[1],
            "signs": list(args.signs),
            "lifts": list(args.lifts),
            "max_total_level": args.max_total_level,
        },
        "coefficient_count": len(records),
        "coefficients": [asdict(record) for record in records],
        "analytic_first_r_kac_residue_checks": [
            {
                "branch": check.branch,
                "pole": [check.pole.real, check.pole.imag],
                "direct": [check.direct.real, check.direct.imag],
                "predicted": [check.predicted.real, check.predicted.imag],
                "absolute_error": check.absolute_error,
                "relative_error": check.relative_error,
            }
            for check in analytic_first_r_kac_residue_checks(
                h_ns=args.h_ns,
                beta_1=args.betas[0],
                beta_2=args.betas[1],
                signs=args.signs,
                lifts=args.lifts,
            )
        ],
    }
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
