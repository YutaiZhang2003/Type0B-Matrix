#!/usr/bin/env python3
r"""Physical NSRR theta blocks from the double-Virasoro ``c`` recursion.

The branching formula first produces the enlarged
``SCA x auxiliary-Majorana`` block. The auxiliary Ramond Majorana is singular
in four star-algebra characters. Star characters are NOT physical spin
projections: the latter are ordinary Walsh sums of parity coefficients,
which already include the Human-Note quadratic sewing sign. We divide only
supported star characters and apply the physical projection last.

For equal HJS signs the Ramond Ward identities put the physical series in
the supported ideal. Opposite HJS signs lie in the complementary ideal and
are annihilated by this auxiliary block. Production calls reject those
components. An explicitly requested ``pbw_diagnostic`` completion is available
for tests only; it is never selected automatically.
The odd form follows from the exact ground-partner Ward identity applied
to the certified even double-Virasoro series; the old odd branching-grid
extension is not used. These are chiral blocks, not a nonchiral Ramond
ground-state projector or a partition function by themselves.

An exponent ``(e0,e1,e2)`` denotes
``q_NS**(e0/2) q_R1**(e1/2) q_R2**(e2/2)``.
"""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
from functools import lru_cache
from itertools import product
import math
from pathlib import Path
import sys
from typing import Mapping, Sequence

import sympy as sp


HERE = Path(__file__).resolve().parent
CODE_ROOT = HERE.parent
for directory in (
    HERE,
    CODE_ROOT / "ramond_branching_recursion",
    CODE_ROOT / "double_virasoro" / "nsrr",
    CODE_ROOT / "c_Recursion",
    CODE_ROOT / "genus_2_cross_channel",
):
    if str(directory) not in sys.path:
        sys.path.insert(0, str(directory))

from compute_full_block import BranchingGrid, base_twice_level  # noqa: E402
from compute_q_expansion import (  # noqa: E402
    BlockSeries,
    Series,
    add_to,
    large_c_vacuum_series,
    reduced_virasoro_series,
    series_multiply,
)
from compute_target import norm_product  # noqa: E402
from nsrr_genus2_block import (  # noqa: E402
    ZERO_VECTOR,
    HumanNSRRThetaOracle,
    auxiliary_majorana_nsrr_series,
    level_triples,
)
from theta_star_algebra import from_star_spectrum, fwht, star_spectrum  # noqa: E402


Exponent = tuple[int, int, int]


def spin_character_index(lifts: Sequence[int]) -> int:
    """Return the Walsh character selected by three ``eta=+/-`` lifts."""

    values = tuple(lifts)
    if len(values) != 3 or any(value not in (-1, 1) for value in values):
        raise ValueError("lifts must contain three +/-1 values")
    return sum((value < 0) << edge for edge, value in enumerate(values))


def _subtract(first: Exponent, second: Exponent) -> Exponent | None:
    value = tuple(first[index] - second[index] for index in range(3))
    return value if min(value) >= 0 else None  # type: ignore[return-value]


def scalar_series_divide(
    numerator: Mapping[Exponent, complex],
    denominator: Mapping[Exponent, complex],
    *,
    maximum_total_twice_level: int,
    zero_tolerance: float = 1.0e-13,
) -> Series:
    """Divide multivariate series whose denominator has nonzero constant."""

    cutoff = int(maximum_total_twice_level)
    constant = complex(denominator.get((0, 0, 0), 0.0j))
    scale = max(1.0, *(abs(value) for value in denominator.values()))
    if abs(constant) <= zero_tolerance * scale:
        raise ZeroDivisionError("the selected auxiliary spin character is singular")
    support = tuple(
        exponent
        for total in range(cutoff + 1)
        for e0 in range(total + 1)
        for e1 in range(total - e0 + 1)
        for exponent in ((e0, e1, total - e0 - e1),)
    )
    quotient: Series = {}
    nonconstant_denominator = tuple(
        (exponent, complex(value))
        for exponent, value in denominator.items()
        if exponent != (0, 0, 0) and sum(exponent) <= cutoff
    )
    for exponent in support:
        residual = complex(numerator.get(exponent, 0.0j))
        for shift, coefficient in nonconstant_denominator:
            previous = _subtract(exponent, shift)
            if previous is not None:
                residual -= coefficient * quotient.get(previous, 0.0j)
        value = residual / constant
        if abs(value) > zero_tolerance:
            quotient[exponent] = value
    return quotient


def evaluate_twice_level_series(series: Mapping[Exponent, complex], q_values: Sequence[complex]) -> complex:
    q = tuple(complex(value) for value in q_values)
    if len(q) != 3:
        raise ValueError("three plumbing parameters are required")
    return sum(
        complex(coefficient)
        * q[0] ** (exponent[0] / 2)
        * q[1] ** (exponent[1] / 2)
        * q[2] ** (exponent[2] / 2)
        for exponent, coefficient in series.items()
    )


@dataclass(frozen=True)
class PhysicalNSRRBlockResult:
    form_parity: int
    eta_left: int
    eta_right: int
    spin_character: int
    cutoff: int
    value: complex
    auxiliary_ground: complex
    coefficient_count: int
    completion_method: str


class NSRRDoubleVirasoroTheta:
    """All HJS components of one physical NSRR theta block at fixed momenta."""

    def __init__(
        self,
        *,
        b: float,
        physical_momenta: Sequence[float],
        cutoff: int,
        primary_parity: int = 0,
        branching_mp_dps: int = 0,
        completion: str = "none",
        pbw_completion_max_level: int = 3,
    ) -> None:
        if len(physical_momenta) != 3:
            raise ValueError("momenta must be ordered as (P_NS,P_R1,P_R2)")
        self.b = float(b)
        if not math.isfinite(self.b) or self.b <= 0:
            raise ValueError("b must be finite and positive")
        self.physical_momenta = tuple(float(value) for value in physical_momenta)
        if any(not math.isfinite(value) or value < 0 for value in self.physical_momenta):
            raise ValueError("continuum momenta must be finite and nonnegative")
        self.note_momenta = tuple(1j * value for value in self.physical_momenta)
        self.cutoff = int(cutoff)
        self.cutoff_twice = 2 * self.cutoff
        self.primary_parity = int(primary_parity)
        if completion not in ("none", "pbw_diagnostic"):
            raise ValueError("completion must be 'none' or 'pbw_diagnostic'")
        if self.cutoff < 0 or self.primary_parity not in (0, 1):
            raise ValueError("cutoff must be nonnegative and primary_parity must be 0 or 1")
        self.completion = completion
        self.pbw_completion_max_level = int(pbw_completion_max_level)
        self.branching = BranchingGrid(
            self.b,
            self.note_momenta,
            self.cutoff,
            primary_parity=self.primary_parity,
            mp_dps=int(branching_mp_dps),
        )
        self.branching.build_actions()
        self.raw_grids: dict[
            tuple[int, int, int, int],
            dict[tuple[Fraction, Fraction, Fraction], complex],
        ] = {}
        self.ward_residual_maximum = 0.0
        # Use ONLY the package's certified f=0, eta=+ interface. In the
        # Human-Note reflected Ramond basis,
        # B_raw^-(P2;n2) = B_raw^+(-P2;-n2).
        # This follows by changing w_2^- -> -w_2^- along with beta_2 ->
        # -beta_2, and applies for both intrinsic NS-primary parities.
        # No monkey patch or noncanonical PhysicalThreePoint call is needed.
        reflected = BranchingGrid(
            self.b,
            (self.note_momenta[0], -self.note_momenta[1], self.note_momenta[2]),
            self.cutoff,
            primary_parity=self.primary_parity,
            mp_dps=int(branching_mp_dps),
        )
        reflected.build_actions()
        for eta, grid in ((1, self.branching), (-1, reflected)):
            for alpha2, alpha3 in product((0, 1), repeat=2):
                values, diagnostic = grid.solve(alpha2, alpha3)
                if eta == -1:
                    values = {(n1, -n2, n3): value
                              for (n1, n2, n3), value in values.items()}
                self.raw_grids[(0, eta, alpha2, alpha3)] = values
                self.ward_residual_maximum = max(
                    self.ward_residual_maximum,
                    float(diagnostic["relative_residual"]),
                )
        self.triples = tuple(
            labels
            for labels in product(
                self.branching.ns, self.branching.r, self.branching.r
            )
            if base_twice_level(labels) <= self.cutoff_twice
        )
        self.reduced_products: dict[
            tuple[Fraction, Fraction, Fraction], Series
        ] = {}
        for labels in self.triples:
            remaining = (
                self.cutoff_twice - base_twice_level(labels)
            ) // 2
            copies = tuple(
                reduced_virasoro_series(
                    self.branching.weights.central_charges[copy],
                    self.branching.weights.triple(labels, copy),
                    remaining,
                )
                for copy in (0, 1)
            )
            self.reduced_products[labels] = series_multiply(
                copies[0], copies[1], remaining
            )
        vacuum, _ = large_c_vacuum_series(self.cutoff)
        self.vacuum_squared = series_multiply(vacuum, vacuum, self.cutoff)
        self.auxiliary = auxiliary_majorana_nsrr_series(
            maximum_total_twice_level=self.cutoff_twice
        )

    @lru_cache(maxsize=None)
    def enlarged_series(
        self, form_parity: int, eta_left: int, eta_right: int
    ) -> BlockSeries:
        form_parity = int(form_parity)
        eta_left = int(eta_left)
        eta_right = int(eta_right)
        if form_parity not in (0, 1) or eta_left not in (-1, 1) or eta_right not in (-1, 1):
            raise ValueError("invalid form parity or HJS sign")
        if form_parity == 1:
            # rho_1 = rho_0 o J_3, Jw^+=i w^-, Jw^-=w^+,
            # J G=-G J and J^T Gram J=-i Gram. Consequently
            # F_1 = -i e_(001) star F_0; the auxiliary commutes with this
            # operation. The bit tuple is (NS,R_at_one,R_at_zero).
            return {key[:5] + (1-key[5],):
                    -1j * (-1)**(key[3]+key[4]) * coefficient
                    for key, coefficient in self.enlarged_series(
                        0, eta_left, eta_right).items()}
        reduced: BlockSeries = {}
        for labels in self.triples:
            n1 = labels[0]
            twice_n1 = int(2 * n1)
            base = (
                int(4 * labels[0] * labels[0]),
                int(4 * labels[1] * labels[1] - Fraction(1, 4)),
                int(4 * labels[2] * labels[2] - Fraction(1, 4)),
            )
            required_alpha_sum = (form_parity - twice_n1) % 2
            alpha_pairs = (
                ((0, 0), (1, 1))
                if required_alpha_sum == 0
                else ((0, 1), (1, 0))
            )
            for alpha2, alpha3 in alpha_pairs:
                left_raw = self.raw_grids[
                    (form_parity, eta_left, alpha2, alpha3)
                ][labels]
                right_raw = self.raw_grids[
                    (form_parity, eta_right, alpha2, alpha3)
                ][labels]
                norm = norm_product(
                    labels,
                    alpha2,
                    alpha3,
                    self.b,
                    self.note_momenta,
                )
                exponent = (
                    twice_n1
                    + (twice_n1 + self.primary_parity) * alpha2
                    + (twice_n1 + self.primary_parity) * alpha3
                    + alpha2 * alpha3
                )
                prefactor = (-1) ** exponent * left_raw * right_raw / (norm * norm)
                parity = (
                    (twice_n1 + self.primary_parity) % 2,
                    alpha2,
                    alpha3,
                )
                for descendant, coefficient in self.reduced_products[labels].items():
                    twice_exponent = tuple(
                        base[edge] + 2 * descendant[edge]
                        for edge in range(3)
                    )
                    if sum(twice_exponent) <= self.cutoff_twice:
                        add_to(
                            reduced,
                            twice_exponent + parity,
                            prefactor * coefficient,
                        )
        full: BlockSeries = {}
        for key, coefficient in reduced.items():
            exponent, parity = key[:3], key[3:]
            for vacuum_exponent, vacuum_coefficient in self.vacuum_squared.items():
                changed = tuple(
                    exponent[edge] + 2 * vacuum_exponent[edge]
                    for edge in range(3)
                )
                if sum(changed) <= self.cutoff_twice:
                    add_to(full, changed + parity, coefficient * vacuum_coefficient)
        return full

    @lru_cache(maxsize=None)
    def star_character_series(
        self,
        form_parity: int,
        eta_left: int,
        eta_right: int,
        spin_character: int,
    ) -> Series:
        """A supported algebraic quotient, NOT a physical fixed-spin block."""
        character = int(spin_character)
        if not 0 <= character < 8:
            raise ValueError("spin_character must be between zero and seven")
        enlarged = self.enlarged_series(form_parity, eta_left, eta_right)
        enlarged_vectors: dict[Exponent, list[complex]] = {}
        for key, coefficient in enlarged.items():
            exponent = key[:3]
            component = key[3] | (key[4] << 1) | (key[5] << 2)
            vector = enlarged_vectors.setdefault(exponent, [0.0j] * 8)
            vector[component] += complex(coefficient)
        numerator = {
            exponent: star_spectrum(vector)[character]
            for exponent, vector in enlarged_vectors.items()
        }
        denominator = {
            exponent: star_spectrum(vector)[character]
            for exponent, vector in self.auxiliary.items()
        }
        return scalar_series_divide(
            numerator,
            denominator,
            maximum_total_twice_level=self.cutoff_twice,
        )

    @lru_cache(maxsize=None)
    def physical_components(self, form_parity: int, eta_left: int, eta_right: int):
        """Recover the literal parity-resolved Human-Note ``Rblock``.

        The equal-sign Ward relation is ``spectrum[k]=0`` for k=0,1,6,7.
        It follows by simultaneously exchanging the Ramond ground partners;
        it is independently tested on every coefficient, not inferred from
        modular agreement. Opposite-sign data cannot be recovered from the
        enlarged series; only their missing channels are supplied by PBW.
        """
        if form_parity not in (0, 1) or eta_left not in (-1, 1) or eta_right not in (-1, 1):
            raise ValueError("invalid form parity or HJS sign")
        ground = star_spectrum(self.auxiliary[(0, 0, 0)])
        supported = tuple(k for k in range(8) if abs(ground[k]) > 1e-12)
        missing = tuple(k for k in range(8) if k not in supported)
        for vector in self.auxiliary.values():
            spectrum = star_spectrum(vector)
            if any(abs(spectrum[k]) > 1e-10 for k in missing):
                raise ArithmeticError("auxiliary support changed; rederive the completion")
        quotients = {k: self.star_character_series(form_parity, eta_left, eta_right, k)
                     for k in supported}
        oracle = None
        if eta_left != eta_right:
            if self.completion != "pbw_diagnostic":
                raise NotImplementedError(
                    "The checked auxiliary-star identity does not determine "
                    "opposite-HJS-sign physical components. Production must "
                    "not replace them with zero or silently use PBW. Request "
                    "completion='pbw_diagnostic' only for an explicitly "
                    "labelled low-order diagnostic."
                )
            if self.cutoff > self.pbw_completion_max_level:
                raise NotImplementedError(
                    "Opposite-HJS-sign physical blocks are annihilated by the "
                    "Human-Note auxiliary star product. Their PBW completion "
                    f"is capped at level {self.pbw_completion_max_level}; "
                    "raise pbw_completion_max_level explicitly to permit the cost."
                )
            b = sp.Rational(str(self.b))
            q_background = b + 1 / b
            p = tuple(sp.Rational(str(value)) for value in self.physical_momenta)
            oracle = HumanNSRRThetaOracle(
                central_charge=sp.Rational(3, 2) + 3*q_background**2,
                h_ns=q_background**2/8 + p[0]**2/2,
                beta_r1=sp.I*p[1]/sp.sqrt(2),
                beta_r2=sp.I*p[2]/sp.sqrt(2),
                form_parity=form_parity, primary_parity=self.primary_parity,
                etas=(eta_left, eta_right),
            )
        answer = {}
        for exponent in level_triples(self.cutoff_twice):
            spectrum = [quotients[k].get(exponent, 0j) if k in supported else 0j
                        for k in range(8)]
            if oracle is not None:
                reference = star_spectrum(oracle.coefficient_components(
                    exponent[0], exponent[1]//2, exponent[2]//2))
                for k in supported:
                    if abs(spectrum[k]-reference[k]) > 2e-8*max(1., abs(reference[k])):
                        raise ArithmeticError("double-Virasoro/PBW supported-channel mismatch")
                for k in missing:
                    spectrum[k] = reference[k]
            answer[exponent] = tuple(from_star_spectrum(spectrum))
        return answer

    @lru_cache(maxsize=None)
    def physical_series(self, form_parity: int, eta_left: int, eta_right: int,
                        spin_character: int) -> Series:
        """Apply the *ordinary* fixed-lift sum to the physical components."""
        character = int(spin_character)
        if not 0 <= character < 8:
            raise ValueError("spin_character must be between zero and seven")
        return {exponent: fwht(vector)[character]
                for exponent, vector in self.physical_components(
                    form_parity, eta_left, eta_right).items()}

    def block(
        self,
        *,
        q_values: Sequence[complex],
        lifts: Sequence[int],
        form_parity: int,
        eta_left: int,
        eta_right: int,
    ) -> PhysicalNSRRBlockResult:
        character = spin_character_index(lifts)
        series = self.physical_series(
            form_parity, eta_left, eta_right, character
        )
        auxiliary_ground = star_spectrum(
            self.auxiliary[(0, 0, 0)]
        )[character]
        return PhysicalNSRRBlockResult(
            form_parity=int(form_parity),
            eta_left=int(eta_left),
            eta_right=int(eta_right),
            spin_character=character,
            cutoff=self.cutoff,
            value=evaluate_twice_level_series(series, q_values),
            auxiliary_ground=complex(auxiliary_ground),
            coefficient_count=len(series),
            completion_method=("double-Virasoro plus equal-sign Ward support"
                               if eta_left == eta_right else
                               "PBW diagnostic nullspace completion; not pure double-Virasoro"),
        )
