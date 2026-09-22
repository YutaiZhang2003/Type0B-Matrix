#!/usr/bin/env python3
r"""Physical NSRR theta blocks from the double-Virasoro ``c`` recursion.

The branching formula first produces the enlarged
``SCA x auxiliary-Majorana`` block. The auxiliary Ramond Majorana is singular
in four star-algebra characters. Star characters are NOT physical spin
projections: the latter are ordinary Walsh sums of parity coefficients,
which already include the Human-Note quadratic sewing sign. We divide only
supported star characters and apply the physical projection last.

Physical components now come from the native C++ double-Virasoro pipeline.
Equal HJS signs use ordinary recovery; opposite signs use the inserted
Theta v_(1/2) pipeline. No physical PBW completion is performed. The older
ordinary enlarged-series implementation remains available for algebraic
identity audits only and is initialized lazily.

Pointwise ``block()`` calls use independently resummed global descendants by
default. Coefficient/series APIs retain their explicit finite-polynomial
meaning. See GLOBAL_RESUMMATION.md for the separate primary, residue and
global accuracy controls.

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
from nsrr_cpp_backend import NativeNSRR, METHOD
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
    recursion_order: int | None = None
    global_tolerance: float | None = None


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
        native_dps: int = 40,
        global_method: str = "resummed",
        recursion_order: int | None = None,
        global_tolerance: float = 1e-13,
        global_max_shell: int = 64,
    ) -> None:
        if len(physical_momenta) != 3:
            raise ValueError("momenta must be ordered as (P_NS,P_R1,P_R2)")
        self.b = float(b)
        if not math.isfinite(self.b) or self.b <= 0 or self.b == 1:
            raise ValueError("b must be finite, positive, and different from one")
        self.physical_momenta = tuple(float(value) for value in physical_momenta)
        if any(not math.isfinite(value) or value < 0 for value in self.physical_momenta):
            raise ValueError("continuum momenta must be finite and nonnegative")
        self.note_momenta = tuple(1j * value for value in self.physical_momenta)
        self.cutoff = int(cutoff)
        if self.cutoff != cutoff:
            raise ValueError("cutoff must be an integer total order")
        self.cutoff_twice = 2 * self.cutoff
        self.primary_parity = int(primary_parity)
        if completion not in ("none", "pbw_diagnostic"):
            raise ValueError("completion must be 'none' or 'pbw_diagnostic'")
        if self.cutoff < 0 or self.primary_parity not in (0, 1):
            raise ValueError("cutoff must be nonnegative and primary_parity must be 0 or 1")
        self.completion = completion
        self.pbw_completion_max_level = int(pbw_completion_max_level)
        # Legacy completion keywords remain accepted for callers with old configs;
        # they never select PBW. Both sign combinations always use native CCY.
        self.native = NativeNSRR(self.b, self.physical_momenta, self.cutoff,
                                 self.primary_parity,
                                 max(native_dps, int(branching_mp_dps)))
        if global_method not in ("resummed", "polynomial"):
            raise ValueError("global_method must be 'resummed' or 'polynomial'")
        self.global_method = global_method
        self._resummed_options = dict(
            branch_level=self.cutoff, branch_truncation="total",
            recursion_order=self.cutoff if recursion_order is None else recursion_order,
            global_tolerance=global_tolerance, global_max_shell=global_max_shell,
            primary_parity=self.primary_parity, dps=max(native_dps, int(branching_mp_dps)))
        self._legacy_ready = False
        self._branching_mp_dps = int(branching_mp_dps)

    @property
    def ward_residual_maximum(self):
        return max(self.native.ward_residual_maximum,
                   getattr(self, "_legacy_ward_residual", 0.0),
                   max((record["diagnostics"]["ward_residual"]
                        for record in getattr(getattr(self, "_resummed", None), "records", {}).values()),
                       default=0.0))

    @property
    def auxiliary(self):
        if not hasattr(self, "_auxiliary"):
            self._auxiliary = auxiliary_majorana_nsrr_series(
                maximum_total_twice_level=self.cutoff_twice)
        return self._auxiliary

    def _initialize_legacy_enlarged(self):
        """Original ordinary double-Virasoro identity audit; never physical PBW."""
        if self._legacy_ready:
            return
        branching_mp_dps = self._branching_mp_dps
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
        self._legacy_ward_residual = 0.0
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
                self._legacy_ward_residual = max(
                    self._legacy_ward_residual,
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
        self._legacy_ready = True

    @lru_cache(maxsize=None)
    def enlarged_series(
        self, form_parity: int, eta_left: int, eta_right: int
    ) -> BlockSeries:
        if hasattr(self, "native"):
            self._initialize_legacy_enlarged()
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

    def physical_components(self, form_parity: int, eta_left: int, eta_right: int):
        """Literal physical Human-Note coefficients, entirely from native CCY."""
        return self.native.physical_components(form_parity, eta_left, eta_right)

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
        if self.global_method == "resummed":
            from nsrr_resummed_backend import ResummedNSRR, METHOD as RESUMMED_METHOD
            if not hasattr(self, "_resummed"):
                self._resummed = ResummedNSRR(self.b, self.physical_momenta, **self._resummed_options)
            values = self._resummed.physical_values(q_values, form_parity, eta_left, eta_right)
            return PhysicalNSRRBlockResult(
                form_parity=int(form_parity), eta_left=int(eta_left), eta_right=int(eta_right),
                spin_character=character, cutoff=self.cutoff,
                value=self._resummed.project(values, lifts),
                auxiliary_ground=complex(star_spectrum(self.auxiliary[(0,0,0)])[character]),
                coefficient_count=0, completion_method=RESUMMED_METHOD,
                recursion_order=self._resummed_options["recursion_order"],
                global_tolerance=self._resummed_options["global_tolerance"])
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
            completion_method=METHOD,
        )
