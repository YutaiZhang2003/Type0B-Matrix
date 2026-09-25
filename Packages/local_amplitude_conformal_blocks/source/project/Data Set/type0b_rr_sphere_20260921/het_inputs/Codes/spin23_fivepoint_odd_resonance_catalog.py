#!/usr/bin/env python3
r"""Combinatorial data for odd five-point free-field resonances.

This module contains no evaluated residue.  It records the component-action
terms that have to be combined on the five resonance planes used by the
fixed-incoming reconstruction, and a small exact-rational momentum design on
those planes.

At ``b=1`` let ``C`` be a subset of the finite picture-zero singlet legs,
``m`` the number of Yukawa screens, and ``ell`` the number of bosonic wall
screens.  A component on the odd screening resonance ``s`` obeys

``len(C) + m + 2*ell = s``.

Its action coefficient is

``prod_C(-2*pi*i*mu*a_j) * (-2*i*mu)^m/m!``
``                            * (-2*pi*mu^2)^ell/ell!``.

After dividing by the pure-``s``-Yukawa coefficient
``(-2*i*mu)^s/s!``, the exact relative coefficient is

``s!/(m!*ell!*2^ell) * pi^(len(C)+ell) * prod_C(a_j)``.

Thus neither a phase nor a power of ``mu`` remains in the relative weight.
The common physical picture factor is also unchanged by choosing a contact
branch and is not included here.

External labels in this module are incoming-first labels: zero is incoming
and one through four are outgoing.  Only outgoing legs 2, 3, and 4 are in
picture zero in the gauge used by the current five-point implementation, so
only singlets among those three legs can belong to ``C``.
"""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
from itertools import combinations
import math
from typing import Iterable, Mapping, Sequence

from spin23_fivepoint_reconstruction import SectorName, contact_basis
from spin23_fivepoint_resonance_channels import (
    FivePointChannelProjection,
    all_nonzero_projections,
)


ODD_SCREENING_NUMBERS: tuple[int, ...] = (1, 3, 5, 7, 9)
ZERO_PICTURE_OUTGOING_LEGS: tuple[int, int, int] = (2, 3, 4)


def resonance_k(screening_number: int) -> int:
    r"""Return ``k=(s+3)/2`` on the plane ``P_in=i*k``."""

    if isinstance(screening_number, bool) or not isinstance(screening_number, int):
        raise ValueError("the NS five-point screening number must be positive and odd")
    s = screening_number
    if s < 1 or s % 2 == 0:
        raise ValueError("the NS five-point screening number must be positive and odd")
    return (s + 3) // 2


def eligible_contact_legs(
    projection: FivePointChannelProjection,
) -> tuple[int, ...]:
    """Return the picture-zero outgoing legs that are singlets."""

    kinds = projection.incoming_first_kinds
    return tuple(
        leg for leg in ZERO_PICTURE_OUTGOING_LEGS if kinds[leg] == "S"
    )


@dataclass(frozen=True)
class OddResonanceComponent:
    """One external-contact/Yukawa/bosonic component of an odd residue."""

    screening_number: int
    contact_outgoing_legs: tuple[int, ...]
    yukawa_count: int
    bosonic_count: int

    def __post_init__(self) -> None:
        s = resonance_k(self.screening_number) * 2 - 3
        contacts = tuple(sorted(int(leg) for leg in self.contact_outgoing_legs))
        if len(set(contacts)) != len(contacts) or any(
            leg not in ZERO_PICTURE_OUTGOING_LEGS for leg in contacts
        ):
            raise ValueError("contacts must be distinct outgoing legs among 2, 3, 4")
        if self.yukawa_count < 0 or self.bosonic_count < 0:
            raise ValueError("screening multiplicities must be nonnegative")
        if len(contacts) + self.yukawa_count + 2 * self.bosonic_count != s:
            raise ValueError("the component does not carry the requested wall charge")
        object.__setattr__(self, "screening_number", s)
        object.__setattr__(self, "contact_outgoing_legs", contacts)

    @property
    def contact_count(self) -> int:
        return len(self.contact_outgoing_legs)

    @property
    def integrated_screen_count(self) -> int:
        return self.yukawa_count + self.bosonic_count

    @property
    def complex_integration_dimension(self) -> int:
        """Two physical moduli plus all integrated action screens."""

        return 2 + self.integrated_screen_count

    @property
    def klt_word_count(self) -> int:
        return math.factorial(self.complex_integration_dimension)

    @property
    def klt_kernel_entry_count(self) -> int:
        return self.klt_word_count**2

    @property
    def hepp_sectors_per_word(self) -> int:
        """Number of orderings of the ``d+1`` simplex gaps."""

        return math.factorial(self.complex_integration_dimension + 1)

    @property
    def hepp_sectors_per_chirality(self) -> int:
        return self.klt_word_count * self.hepp_sectors_per_word

    @property
    def relative_rational_coefficient(self) -> Fraction:
        """Rational part after stripping the pure-``s``-Yukawa term."""

        return Fraction(
            math.factorial(self.screening_number),
            math.factorial(self.yukawa_count)
            * math.factorial(self.bosonic_count)
            * 2**self.bosonic_count,
        )

    @property
    def relative_pi_power(self) -> int:
        return self.contact_count + self.bosonic_count

    @property
    def relative_weight_expression(self) -> str:
        rational = self.relative_rational_coefficient
        factors: list[str] = []
        if rational != 1 or (self.relative_pi_power == 0 and not self.contact_count):
            factors.append(str(rational))
        if self.relative_pi_power == 1:
            factors.append("pi")
        elif self.relative_pi_power > 1:
            factors.append(f"pi^{self.relative_pi_power}")
        factors.extend(f"a{leg}" for leg in self.contact_outgoing_legs)
        return "*".join(factors) if factors else "1"

    def relative_weight(self, charges: Mapping[int, complex]) -> complex:
        """Numerically evaluate the exact relative-weight formula."""

        value = complex(self.relative_rational_coefficient)
        value *= math.pi**self.relative_pi_power
        for leg in self.contact_outgoing_legs:
            if leg not in charges:
                raise ValueError(f"missing Liouville charge for outgoing leg {leg}")
            value *= complex(charges[leg])
        return value

    def action_weight(
        self,
        charges: Mapping[int, complex],
        *,
        mu: complex = 1.0,
    ) -> complex:
        """Return the unstripped component-action coefficient."""

        value = 1.0 + 0.0j
        for leg in self.contact_outgoing_legs:
            if leg not in charges:
                raise ValueError(f"missing Liouville charge for outgoing leg {leg}")
            value *= -2.0j * math.pi * complex(mu) * complex(charges[leg])
        value *= (-2.0j * complex(mu)) ** self.yukawa_count / math.factorial(
            self.yukawa_count
        )
        value *= (-2.0 * math.pi * complex(mu) ** 2) ** self.bosonic_count
        value /= math.factorial(self.bosonic_count)
        return value

    def pure_yukawa_weight(self, *, mu: complex = 1.0) -> complex:
        return (-2.0j * complex(mu)) ** self.screening_number / math.factorial(
            self.screening_number
        )


def odd_resonance_components(
    projection: FivePointChannelProjection,
    screening_number: int,
) -> tuple[OddResonanceComponent, ...]:
    """Enumerate every action component allowed for one scalar projection."""

    s = 2 * resonance_k(screening_number) - 3
    eligible = eligible_contact_legs(projection)
    result: list[OddResonanceComponent] = []
    for contact_count in range(min(len(eligible), s) + 1):
        for contacts in combinations(eligible, contact_count):
            remaining = s - contact_count
            for bosonic_count in range(remaining // 2 + 1):
                result.append(
                    OddResonanceComponent(
                        screening_number=s,
                        contact_outgoing_legs=contacts,
                        yukawa_count=remaining - 2 * bosonic_count,
                        bosonic_count=bosonic_count,
                    )
                )
    return tuple(result)


def projection_sector(projection: FivePointChannelProjection) -> SectorName:
    """Return the canonical fixed-incoming reconstruction sector."""

    kinds = projection.incoming_first_kinds
    outgoing_vectors = kinds[1:].count("V")
    if kinds[0] == "S":
        if outgoing_vectors == 0:
            return "S_to_SSSS"
        if outgoing_vectors == 2:
            return "S_to_SSVV"
        if outgoing_vectors == 4:
            return "S_to_VVVV"
    else:
        if outgoing_vectors == 1:
            return "V_to_VSSS"
        if outgoing_vectors == 3:
            return "V_to_VVVS"
    raise ValueError("the projection is forbidden by the even-vector rule")


def canonical_outgoing_order(
    projection: FivePointChannelProjection,
) -> tuple[int, int, int, int]:
    """Return original outgoing labels in the canonical sector order."""

    kinds = projection.incoming_first_kinds
    labels = range(1, 5)
    if kinds[0] == "S" and kinds[1:].count("V") == 0:
        return (1, 2, 3, 4)
    # The canonical mixed-sector bases place vectors first for an incoming V
    # and singlets first for an incoming S.
    first_kind = "S" if kinds[0] == "S" else "V"
    order = tuple(label for label in labels if kinds[label] == first_kind)
    order += tuple(label for label in labels if kinds[label] != first_kind)
    if len(order) != 4:
        raise AssertionError("canonical outgoing order lost a leg")
    return order  # type: ignore[return-value]


def canonical_outgoing_x(
    projection: FivePointChannelProjection,
    outgoing_x: Sequence[Fraction | int | float | complex],
) -> tuple[Fraction | int | float | complex, ...]:
    """Permute labelled kinematics into the canonical contact-basis order."""

    values = tuple(outgoing_x)
    if len(values) != 4:
        raise ValueError("four outgoing variables are required")
    return tuple(values[label - 1] for label in canonical_outgoing_order(projection))


@dataclass(frozen=True)
class RationalResonanceSample:
    """One exact outgoing partition on an odd resonance plane."""

    sample_id: str
    screening_number: int
    outgoing_t: tuple[Fraction, Fraction, Fraction, Fraction]

    def __post_init__(self) -> None:
        s = 2 * resonance_k(self.screening_number) - 3
        values = tuple(Fraction(value) for value in self.outgoing_t)
        if len(values) != 4 or any(value <= 0 for value in values):
            raise ValueError("a sample needs four positive outgoing t values")
        if sum(values) != resonance_k(s):
            raise ValueError("the outgoing partition is not on its resonance plane")
        object.__setattr__(self, "screening_number", s)
        object.__setattr__(self, "outgoing_t", values)

    @property
    def incoming_k(self) -> int:
        return resonance_k(self.screening_number)

    @property
    def outgoing_x(self) -> tuple[Fraction, Fraction, Fraction, Fraction]:
        """Return ``x_j=i P_j=-t_j`` for ``P_j=i t_j``."""

        return tuple(-value for value in self.outgoing_t)  # type: ignore[return-value]

    @property
    def outgoing_momenta(self) -> tuple[complex, complex, complex, complex]:
        return tuple(1.0j * float(value) for value in self.outgoing_t)  # type: ignore[return-value]

    @property
    def factorization_divisors(self) -> tuple[Fraction, ...]:
        x = self.outgoing_x
        return tuple(
            1 + sum((x[label] for label in selected), Fraction(0))
            for size in (2, 3)
            for selected in combinations(range(4), size)
        )

    @property
    def minimum_factorization_margin(self) -> Fraction:
        return min(abs(value) for value in self.factorization_divisors)


def _sample(
    sample_id: str,
    screening_number: int,
    numerators: tuple[int, int, int, int],
) -> RationalResonanceSample:
    return RationalResonanceSample(
        sample_id,
        screening_number,
        tuple(Fraction(value, 211) for value in numerators),  # type: ignore[arg-type]
    )


# Five, three, two, one, and one partitions on s=1,3,5,7,9.  The common
# prime denominator makes exact pole and rank checks inexpensive.  These
# twelve points are shared by every tensor projection.
RECONSTRUCTION_SAMPLES: tuple[RationalResonanceSample, ...] = (
    _sample("s1_p01", 1, (39, 98, 42, 243)),
    _sample("s1_p02", 1, (187, 72, 65, 98)),
    _sample("s1_p03", 1, (109, 76, 155, 82)),
    _sample("s1_p04", 1, (57, 137, 46, 182)),
    _sample("s1_p05", 1, (75, 119, 43, 185)),
    _sample("s3_p01", 3, (257, 89, 232, 55)),
    _sample("s3_p02", 3, (34, 81, 255, 263)),
    _sample("s3_p03", 3, (203, 197, 136, 97)),
    _sample("s5_p01", 5, (447, 158, 85, 154)),
    _sample("s5_p02", 5, (257, 144, 219, 224)),
    _sample("s7_p01", 7, (226, 512, 232, 85)),
    _sample("s9_p01", 9, (428, 707, 70, 61)),
)


_SECTOR_SAMPLE_IDS: dict[SectorName, tuple[str, ...]] = {
    "S_to_SSSS": tuple(sample.sample_id for sample in RECONSTRUCTION_SAMPLES),
    "S_to_SSVV": (
        "s1_p01",
        "s1_p02",
        "s1_p03",
        "s1_p04",
        "s1_p05",
        "s3_p01",
        "s3_p02",
        "s5_p01",
    ),
    "V_to_VSSS": (
        "s1_p01",
        "s1_p02",
        "s1_p03",
        "s1_p04",
        "s3_p01",
        "s3_p02",
        "s5_p01",
    ),
    "S_to_VVVV": ("s1_p01",),
    "V_to_VVVS": ("s1_p01",),
}


def reconstruction_samples_for_sector(
    sector: SectorName,
) -> tuple[RationalResonanceSample, ...]:
    """Return the minimal rows used for one canonical contact basis."""

    if sector not in _SECTOR_SAMPLE_IDS:
        raise ValueError(f"unknown five-point sector {sector!r}")
    by_id = {sample.sample_id: sample for sample in RECONSTRUCTION_SAMPLES}
    return tuple(by_id[sample_id] for sample_id in _SECTOR_SAMPLE_IDS[sector])


def reconstruction_design_rows(
    projection: FivePointChannelProjection,
) -> tuple[tuple[Fraction | int | float | complex, ...], ...]:
    """Return exact contact-basis rows for one of the 26 projections."""

    sector = projection_sector(projection)
    return tuple(
        contact_basis(
            sector,
            canonical_outgoing_x(projection, sample.outgoing_x),
        )
        for sample in reconstruction_samples_for_sector(sector)
    )


def component_manifest_rows() -> Iterable[dict[str, object]]:
    """Yield flat, serialization-friendly rows for all 26 projections."""

    for projection in all_nonzero_projections():
        sector = projection_sector(projection)
        eligible = eligible_contact_legs(projection)
        for s in ODD_SCREENING_NUMBERS:
            for component in odd_resonance_components(projection, s):
                yield {
                    "projection_label": projection.label,
                    "incoming_first_kinds": projection.incoming_first_kinds,
                    "vector_pairing": ";".join(
                        f"{first}{second}"
                        for first, second in projection.vector_pairing
                    ),
                    "canonical_sector": sector,
                    "canonical_outgoing_order": ",".join(
                        str(label) for label in canonical_outgoing_order(projection)
                    ),
                    "eligible_contact_legs": ",".join(
                        str(label) for label in eligible
                    ),
                    "screening_number": s,
                    "incoming_k": resonance_k(s),
                    "contact_legs": ",".join(
                        str(label) for label in component.contact_outgoing_legs
                    ),
                    "contact_count": component.contact_count,
                    "yukawa_count": component.yukawa_count,
                    "bosonic_count": component.bosonic_count,
                    "integrated_screen_count": component.integrated_screen_count,
                    "relative_rational": str(
                        component.relative_rational_coefficient
                    ),
                    "relative_pi_power": component.relative_pi_power,
                    "relative_weight": component.relative_weight_expression,
                    "complex_dimension": component.complex_integration_dimension,
                    "klt_word_count": component.klt_word_count,
                    "klt_kernel_entries": component.klt_kernel_entry_count,
                    "hepp_sectors_per_word": component.hepp_sectors_per_word,
                    "hepp_sectors_per_chirality": (
                        component.hepp_sectors_per_chirality
                    ),
                    "status": "combinatorial_input_not_evaluated_residue",
                }


__all__ = [
    "ODD_SCREENING_NUMBERS",
    "OddResonanceComponent",
    "RECONSTRUCTION_SAMPLES",
    "RationalResonanceSample",
    "ZERO_PICTURE_OUTGOING_LEGS",
    "canonical_outgoing_order",
    "canonical_outgoing_x",
    "component_manifest_rows",
    "eligible_contact_legs",
    "odd_resonance_components",
    "projection_sector",
    "reconstruction_design_rows",
    "reconstruction_samples_for_sector",
    "resonance_k",
]
