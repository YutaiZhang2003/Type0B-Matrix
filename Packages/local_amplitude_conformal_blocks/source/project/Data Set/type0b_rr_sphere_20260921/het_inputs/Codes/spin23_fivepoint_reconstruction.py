#!/usr/bin/env python3
r"""Finite fixed-incoming ansatz data for Spin(23) sphere five-point amplitudes.

The module deliberately does not assume crossing between different choices of
the incoming leg.  It records the five outgoing-permutation orbits allowed by
the even-vector selection rule, their *candidate* singlet-exchange
factorization divisors, and the minimal regular contact bases suggested by the
lower-point degree rule

``contact degree <= number_of_singlets - 1``.

All energies use ``x_i = i omega_i`` for the four outgoing legs, so
``x_0 = sum(x_i)``.  A factorization divisor is represented by a tuple of
one-based outgoing-leg labels and means

``d_A = 1 + sum(x_i for i in A)``.

The exact lower-point formulas sharpen the divisor catalog.  Every three-leg
divisor has a four-point factor on which the incoming singlet has ``x=-1``.
Both repository formulas ``S -> SSS`` and ``S -> SVV`` vanish on that locus.
Consequently all triple divisors have zero residue and are removable.  The
nonzero pole part contains only pair divisors and, when two disjoint pair
divisors are compatible, their ordinary double pole.

The pole evaluator below uses the local sewing normalization already visible
at four points, ``Res_d M4 = pi*M3_left*M3_right``.  This fixes the
energy-dependent pole part in the same reduced convention, up to a single
momentum-independent five-point normalization exposed as
``factorization_scale``.  The two incoming-vector sectors fail closed by
default.  They become evaluable when the numerically locked, fixed-incoming
``V -> VSS`` candidate is selected explicitly through
``four_point_v_to_vss_reduced_x``.  That candidate was obtained from its own
block integral, not by crossing the known ``S -> SVV`` formula.

The regular ansatz is applied to the soft-stripped amplitude
``M5/(x0*x1*x2*x3*x4)``.  These data organize a reconstruction after the
lower-point factorization part has been subtracted.  They do not assert that
the empirical degree rule is an all-point theorem.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
import math
from typing import Callable, Iterable, Literal, Sequence


SectorName = Literal[
    "S_to_SSSS",
    "S_to_SSVV",
    "S_to_VVVV",
    "V_to_VSSS",
    "V_to_VVVS",
]
Pole = tuple[int, ...]
PolePair = tuple[Pole, Pole]
VToVSSReduced = Callable[[complex, complex, complex], complex]


class MissingLowerPointAmplitude(RuntimeError):
    """Raised when an incoming-vector residue needs generic ``V -> VSS``."""


def _pole(*labels: int) -> Pole:
    return tuple(sorted(labels))


def _compatible(first: Pole, second: Pole) -> bool:
    left, right = set(first), set(second)
    if len(first) == len(second) == 2:
        return left.isdisjoint(right)
    if len(first) == 2 and len(second) == 3:
        return left < right
    if len(first) == 3 and len(second) == 2:
        return right < left
    return False


@dataclass(frozen=True)
class FixedIncomingSector:
    """One canonical fixed-incoming S/V process."""

    name: SectorName
    states: tuple[str, str, str, str, str]
    tensor_dimension: int
    contact_degree: int
    contact_dimension: int
    poles: tuple[Pole, ...]

    @property
    def nonzero_poles(self) -> tuple[Pole, ...]:
        """Divisors whose residue is nonzero with the known lower-point data."""

        return tuple(pole for pole in self.poles if len(pole) == 2)

    @property
    def removable_poles(self) -> tuple[Pole, ...]:
        """Candidate divisors whose exact lower-point residue vanishes."""

        return tuple(pole for pole in self.poles if len(pole) == 3)

    @property
    def compatible_pole_pairs(self) -> tuple[PolePair, ...]:
        return tuple(
            (first, second)
            for first, second in combinations(self.poles, 2)
            if _compatible(first, second)
        )

    @property
    def genuine_double_pole_pairs(self) -> tuple[PolePair, ...]:
        """Compatible pairs with a nonzero raw iterated residue.

        A nested pair--triple intersection forces the intervening outgoing
        energy to zero.  Such products can occur after the universal soft
        factor is stripped, but their raw iterated residue vanishes.
        """

        return tuple(
            (first, second)
            for first, second in self.compatible_pole_pairs
            if len(first) == len(second) == 2 and set(first).isdisjoint(second)
        )

    @property
    def minimum_resonance_planes(self) -> int:
        """Planes needed to exclude ``prod_k (x0+k) Q`` ambiguities."""

        return self.contact_degree + 1


SECTORS: dict[SectorName, FixedIncomingSector] = {
    "S_to_SSSS": FixedIncomingSector(
        name="S_to_SSSS",
        states=("S", "S", "S", "S", "S"),
        tensor_dimension=1,
        contact_degree=4,
        contact_dimension=12,
        poles=tuple(
            tuple(selected)
            for size in (2, 3)
            for selected in combinations(range(1, 5), size)
        ),
    ),
    "S_to_SSVV": FixedIncomingSector(
        name="S_to_SSVV",
        states=("S", "S", "S", "V", "V"),
        tensor_dimension=1,
        contact_degree=2,
        contact_dimension=8,
        poles=(_pole(1, 2), _pole(3, 4), _pole(1, 3, 4), _pole(2, 3, 4)),
    ),
    "S_to_VVVV": FixedIncomingSector(
        name="S_to_VVVV",
        states=("S", "V", "V", "V", "V"),
        tensor_dimension=3,
        contact_degree=0,
        contact_dimension=1,
        poles=tuple(tuple(selected) for selected in combinations(range(1, 5), 2)),
    ),
    "V_to_VSSS": FixedIncomingSector(
        name="V_to_VSSS",
        states=("V", "V", "S", "S", "S"),
        tensor_dimension=1,
        contact_degree=2,
        contact_dimension=7,
        poles=(_pole(2, 3), _pole(2, 4), _pole(3, 4), _pole(2, 3, 4)),
    ),
    "V_to_VVVS": FixedIncomingSector(
        name="V_to_VVVS",
        states=("V", "V", "V", "V", "S"),
        tensor_dimension=3,
        contact_degree=0,
        contact_dimension=1,
        poles=(
            _pole(1, 2),
            _pole(1, 3),
            _pole(2, 3),
            _pole(1, 2, 4),
            _pole(1, 3, 4),
            _pole(2, 3, 4),
        ),
    ),
}


def _as_outgoing(values: Sequence[complex]) -> tuple[complex, complex, complex, complex]:
    result = tuple(values)
    if len(result) != 4:
        raise ValueError("four outgoing x variables are required")
    return result  # type: ignore[return-value]


def elementary_symmetric(values: Sequence[complex], degree: int) -> complex:
    """Return one elementary symmetric polynomial."""

    selected = tuple(values)
    if not 0 <= degree <= len(selected):
        raise ValueError("elementary-symmetric degree is out of range")
    if degree == 0:
        return 1
    return sum(_product(entries) for entries in combinations(selected, degree))


def _product(values: Iterable[complex]) -> complex:
    result: complex = 1
    for value in values:
        result *= value
    return result


def contact_basis(name: SectorName, outgoing_x: Sequence[complex]) -> tuple[complex, ...]:
    """Evaluate the minimal regular contact basis for one canonical sector."""

    x1, x2, x3, x4 = _as_outgoing(outgoing_x)
    if name == "S_to_SSSS":
        values = (x1, x2, x3, x4)
        e1, e2, e3, e4 = (
            elementary_symmetric(values, degree) for degree in range(1, 5)
        )
        return (
            1,
            e1,
            e1**2,
            e2,
            e1**3,
            e1 * e2,
            e3,
            e1**4,
            e1**2 * e2,
            e2**2,
            e1 * e3,
            e4,
        )
    if name == "S_to_SSVV":
        singlet_sum, vector_sum = x1 + x2, x3 + x4
        singlet_product, vector_product = x1 * x2, x3 * x4
        return (
            1,
            singlet_sum,
            vector_sum,
            singlet_sum**2,
            singlet_sum * vector_sum,
            vector_sum**2,
            singlet_product,
            vector_product,
        )
    if name == "V_to_VSSS":
        singlet_sum = x2 + x3 + x4
        singlet_pair_sum = x2 * x3 + x2 * x4 + x3 * x4
        return (
            1,
            x1,
            singlet_sum,
            x1**2,
            x1 * singlet_sum,
            singlet_sum**2,
            singlet_pair_sum,
        )
    if name in ("S_to_VVVV", "V_to_VVVS"):
        return (1,)
    raise ValueError(f"unknown five-point sector {name!r}")


def tensor_channel_poles(name: SectorName) -> dict[str, tuple[Pole, ...]]:
    """Return pole assignments for independent tensor coefficients.

    Scalar and two-vector sectors have one tensor coefficient.  In a
    four-vector sector, each delta-pairing coefficient sees only the two
    divisors associated with that same pairing.
    """

    if name in ("S_to_SSSS", "S_to_SSVV", "V_to_VSSS"):
        return {"scalar": SECTORS[name].poles}
    if name == "S_to_VVVV":
        return {
            "delta12_delta34": (_pole(1, 2), _pole(3, 4)),
            "delta13_delta24": (_pole(1, 3), _pole(2, 4)),
            "delta14_delta23": (_pole(1, 4), _pole(2, 3)),
        }
    if name == "V_to_VVVS":
        return {
            "delta01_delta23": (_pole(2, 3), _pole(2, 3, 4)),
            "delta02_delta13": (_pole(1, 3), _pole(1, 3, 4)),
            "delta03_delta12": (_pole(1, 2), _pole(1, 2, 4)),
        }
    raise ValueError(f"unknown five-point sector {name!r}")


def nonzero_tensor_channel_poles(name: SectorName) -> dict[str, tuple[Pole, ...]]:
    """Return the nonzero pair poles of each independent tensor coefficient."""

    return {
        tensor: tuple(pole for pole in poles if len(pole) == 2)
        for tensor, poles in tensor_channel_poles(name).items()
    }


def removable_tensor_channel_poles(name: SectorName) -> dict[str, tuple[Pole, ...]]:
    """Return the size-three divisors with identically zero residue."""

    return {
        tensor: tuple(pole for pole in poles if len(pole) == 3)
        for tensor, poles in tensor_channel_poles(name).items()
    }


def denominator(outgoing_x: Sequence[complex], pole: Pole) -> complex:
    """Evaluate ``d_A=1+sum_{i in A} x_i``."""

    values = _as_outgoing(outgoing_x)
    if len(pole) not in (2, 3) or any(label not in range(1, 5) for label in pole):
        raise ValueError("a pole must contain two or three outgoing labels")
    if len(set(pole)) != len(pole):
        raise ValueError("pole labels must be distinct")
    return 1 + sum(values[label - 1] for label in pole)


def soft_factor(outgoing_x: Sequence[complex]) -> complex:
    """Return the universal raw soft factor ``x0*x1*x2*x3*x4``."""

    values = _as_outgoing(outgoing_x)
    return sum(values) * _product(values)


def four_point_s_to_svv_x(
    singlet_x: complex,
    vector_x1: complex,
    vector_x2: complex,
) -> complex:
    r"""Known raw ``S -> S V V`` formula in ``x=i*omega`` variables."""

    singlet_x, vector_x1, vector_x2 = map(
        complex, (singlet_x, vector_x1, vector_x2)
    )
    incoming_x = singlet_x + vector_x1 + vector_x2
    product = incoming_x * singlet_x * vector_x1 * vector_x2
    divisor = 1 + vector_x1 + vector_x2
    if divisor == 0:
        raise ZeroDivisionError("S -> SVV lies on its vector-pair pole")
    return math.pi * product * (
        1 + 2 * incoming_x + incoming_x * singlet_x / divisor
    )


def four_point_s_to_sss_x(
    singlet_x1: complex,
    singlet_x2: complex,
    singlet_x3: complex,
) -> complex:
    r"""Known raw ``S -> S S S`` formula in ``x=i*omega`` variables."""

    values = tuple(map(complex, (singlet_x1, singlet_x2, singlet_x3)))
    incoming_x = sum(values)
    product = incoming_x * _product(values)
    pair_divisors = tuple(
        1 + values[first] + values[second]
        for first, second in combinations(range(3), 2)
    )
    if any(divisor == 0 for divisor in pair_divisors):
        raise ZeroDivisionError("S -> SSS lies on a pair pole")
    second_symmetric = sum(
        values[first] * values[second]
        for first, second in combinations(range(3), 2)
    )
    return math.pi * product * (
        product * sum(1 / divisor for divisor in pair_divisors)
        + (1 + 2 * incoming_x)
        * (1 + second_symmetric - incoming_x**2)
    )


def four_point_v_to_vss_reduced_x(
    vector_x: complex,
    singlet_x1: complex,
    singlet_x2: complex,
) -> complex:
    r"""Numerically locked reduced ``V -> V S S`` candidate in x variables.

    The return value is

    ``M4[V -> V(vector_x) S(singlet_x1) S(singlet_x2)]``
    ``/(x_in*vector_x*singlet_x1*singlet_x2)``.

    It is a fixed-incoming-vector candidate inferred from the direct block
    integral.  Supplying this function to the five-point pole evaluator is
    therefore an explicit use of that conjecture, not an incoming-leg
    crossing assumption.
    """

    vector_x, singlet_x1, singlet_x2 = map(
        complex, (vector_x, singlet_x1, singlet_x2)
    )
    incoming_x = vector_x + singlet_x1 + singlet_x2
    divisor = 1 + singlet_x1 + singlet_x2
    if divisor == 0:
        raise ZeroDivisionError("V -> VSS lies on its singlet-pair pole")
    return math.pi * (
        1 + 2 * incoming_x - singlet_x1 * singlet_x2 / divisor
    )


def four_point_v_to_vss_x(
    vector_x: complex,
    singlet_x1: complex,
    singlet_x2: complex,
) -> complex:
    r"""Numerically locked raw ``V -> V S S`` candidate in x variables."""

    values = tuple(map(complex, (vector_x, singlet_x1, singlet_x2)))
    return sum(values) * _product(values) * four_point_v_to_vss_reduced_x(
        *values
    )


def _canonical_tensor(name: SectorName, tensor: str | None) -> str:
    channels = tensor_channel_poles(name)
    if tensor is None:
        if len(channels) != 1:
            raise ValueError(f"sector {name!r} requires an explicit tensor label")
        return next(iter(channels))
    if tensor not in channels:
        raise ValueError(
            f"unknown tensor {tensor!r} for {name!r}; expected one of "
            f"{tuple(channels)!r}"
        )
    return tensor


def _pair_extra(
    states: tuple[str, str, str, str, str],
    pole: Pole,
    outgoing_x: tuple[complex, complex, complex, complex],
) -> complex:
    """Extra Ward factor in a first-discrete-momentum three-point vertex."""

    first, second = pole
    if states[first] != states[second]:
        raise AssertionError("a nonzero pair pole must join like species")
    if states[first] == "S":
        return outgoing_x[first - 1] * outgoing_x[second - 1]
    return 1


def normalized_channel_residue_lift(
    name: SectorName,
    pole: Pole,
    outgoing_x: Sequence[complex],
    *,
    tensor: str | None = None,
    v_to_vss_reduced: VToVSSReduced | None = None,
    factorization_scale: complex = 1,
) -> complex:
    r"""Return a soft-stripped lift of one exact factorization residue.

    The returned number is the residue of
    ``M5/(x0*x1*x2*x3*x4)`` at ``d_pole=0``.  Its continuation away from the
    divisor preserves the universal soft factor and leaves only the
    complementary pair denominator.  Different lifts differ by a regular
    contact term, so this convention is part of the reconstruction ansatz.

    ``v_to_vss_reduced(v,s1,s2)`` must return
    ``M4[V -> V(v) S(s1) S(s2)]/(x0*v*s1*s2)`` in the same raw convention.
    It is needed only in the two incoming-vector sectors.
    """

    values = _as_outgoing(outgoing_x)
    tensor_name = _canonical_tensor(name, tensor)
    candidate = tensor_channel_poles(name)[tensor_name]
    if pole not in candidate:
        raise ValueError(
            f"pole {pole!r} does not route to tensor {tensor_name!r} in {name!r}"
        )

    # The exact S-incoming four-point factor vanishes when its incoming
    # momentum is x=-1, so every candidate triple pole is removable.
    if len(pole) == 3:
        return 0.0j
    if len(pole) != 2:
        raise ValueError("factorization poles must contain two or three labels")

    states = SECTORS[name].states
    pair_extra = _pair_extra(states, pole, values)
    complement = tuple(label for label in range(1, 5) if label not in pole)
    complement_states = tuple(states[label] for label in complement)
    scale = complex(factorization_scale)

    if states[0] == "S":
        if complement_states[0] != complement_states[1]:
            raise AssertionError("an incoming-S pair residue has mixed complement")
        first = values[complement[0] - 1]
        second = values[complement[1] - 1]
        complement_sum = first + second
        complement_divisor = 1 + complement_sum
        if complement_divisor == 0:
            raise ZeroDivisionError(
                "the residue lift lies on its compatible double pole"
            )
        if complement_states[0] == "V":
            # S -> S(-1) V V =
            # -2*pi*x0*v1*v2*(v1+v2)^2/d_VV on the pair divisor.
            return (
                scale
                * 2j
                * math.pi**2
                * pair_extra
                * complement_sum**2
                / complement_divisor
            )
        ward_polynomial = first**2 + first * second + second**2 - 1
        # S -> S(-1) S S = 2*pi*x0*s1*s2*(s1+s2)^2
        #                          *ward_polynomial/d_SS.
        return (
            -scale
            * 2j
            * math.pi**2
            * pair_extra
            * complement_sum**2
            * ward_polynomial
            / complement_divisor
        )

    if v_to_vss_reduced is None:
        raise MissingLowerPointAmplitude(
            f"{name} pair residue {pole!r} needs the generic raw V -> VSS "
            "four-point amplitude; crossing S -> SVV is not a substitute"
        )
    vector_labels = tuple(
        label for label in complement if states[label] == "V"
    )
    singlet_labels = tuple(
        label for label in complement if states[label] == "S"
    )
    if len(vector_labels) != 1 or len(singlet_labels) != 1:
        raise AssertionError("an incoming-V residue must leave one V and one S")
    vector_x = values[vector_labels[0] - 1]
    singlet_x = values[singlet_labels[0] - 1]
    lower = complex(v_to_vss_reduced(vector_x, -1.0, singlet_x))
    return scale * 1j * math.pi * pair_extra * lower


def normalized_double_residue_lift(
    name: SectorName,
    first: Pole,
    second: Pole,
    outgoing_x: Sequence[complex],
    *,
    tensor: str | None = None,
    factorization_scale: complex = 1,
) -> complex:
    r"""Return the soft-stripped coefficient of a genuine double pole."""

    values = _as_outgoing(outgoing_x)
    tensor_name = _canonical_tensor(name, tensor)
    active = nonzero_tensor_channel_poles(name)[tensor_name]
    if first not in active or second not in active:
        raise ValueError("both double-pole divisors must route to the tensor")
    if len(first) != 2 or len(second) != 2 or set(first) & set(second):
        raise ValueError("a genuine double pole requires disjoint pair divisors")
    if set(first) | set(second) != set(range(1, 5)):
        raise ValueError("the two pair divisors must partition the outgoing legs")
    states = SECTORS[name].states
    if states[0] != "S":
        raise ValueError("no incoming-vector tensor coefficient has a double pole")
    return (
        complex(factorization_scale)
        * 2j
        * math.pi**2
        * _pair_extra(states, first, values)
        * _pair_extra(states, second, values)
    )


def normalized_pole_part(
    name: SectorName,
    outgoing_x: Sequence[complex],
    *,
    tensor: str | None = None,
    v_to_vss_reduced: VToVSSReduced | None = None,
    factorization_scale: complex = 1,
) -> complex:
    r"""Evaluate the complete soft-stripped pole part by inclusion-exclusion."""

    values = _as_outgoing(outgoing_x)
    tensor_name = _canonical_tensor(name, tensor)
    active = nonzero_tensor_channel_poles(name)[tensor_name]
    result = sum(
        normalized_channel_residue_lift(
            name,
            pole,
            values,
            tensor=tensor_name,
            v_to_vss_reduced=v_to_vss_reduced,
            factorization_scale=factorization_scale,
        )
        / denominator(values, pole)
        for pole in active
    )
    for first, second in combinations(active, 2):
        if len(first) == len(second) == 2 and set(first).isdisjoint(second):
            result -= normalized_double_residue_lift(
                name,
                first,
                second,
                values,
                tensor=tensor_name,
                factorization_scale=factorization_scale,
            ) / (denominator(values, first) * denominator(values, second))
    return result


def raw_pole_part(
    name: SectorName,
    outgoing_x: Sequence[complex],
    *,
    tensor: str | None = None,
    v_to_vss_reduced: VToVSSReduced | None = None,
    factorization_scale: complex = 1,
) -> complex:
    """Evaluate the pole part with the universal raw soft factor restored."""

    values = _as_outgoing(outgoing_x)
    return soft_factor(values) * normalized_pole_part(
        name,
        values,
        tensor=tensor,
        v_to_vss_reduced=v_to_vss_reduced,
        factorization_scale=factorization_scale,
    )


def normalized_ansatz(
    name: SectorName,
    outgoing_x: Sequence[complex],
    contact_coefficients: Sequence[complex],
    *,
    tensor: str | None = None,
    v_to_vss_reduced: VToVSSReduced | None = None,
    factorization_scale: complex = 1,
) -> complex:
    """Evaluate the pole part plus the bounded regular contact polynomial."""

    basis = contact_basis(name, outgoing_x)
    coefficients = tuple(map(complex, contact_coefficients))
    if len(coefficients) != len(basis):
        raise ValueError(
            f"{name} needs {len(basis)} contact coefficients, got "
            f"{len(coefficients)}"
        )
    return normalized_pole_part(
        name,
        outgoing_x,
        tensor=tensor,
        v_to_vss_reduced=v_to_vss_reduced,
        factorization_scale=factorization_scale,
    ) + sum(coefficient * value for coefficient, value in zip(coefficients, basis))


def raw_ansatz(
    name: SectorName,
    outgoing_x: Sequence[complex],
    contact_coefficients: Sequence[complex],
    *,
    tensor: str | None = None,
    v_to_vss_reduced: VToVSSReduced | None = None,
    factorization_scale: complex = 1,
) -> complex:
    """Evaluate the raw five-point ansatz including its soft factor."""

    values = _as_outgoing(outgoing_x)
    return soft_factor(values) * normalized_ansatz(
        name,
        values,
        contact_coefficients,
        tensor=tensor,
        v_to_vss_reduced=v_to_vss_reduced,
        factorization_scale=factorization_scale,
    )


@dataclass(frozen=True)
class ReconstructionSample:
    """One raw tensor coefficient used in a contact-term reconstruction."""

    outgoing_x: tuple[complex, complex, complex, complex]
    amplitude: complex
    tensor: str = "scalar"


@dataclass(frozen=True)
class ReconstructionFit:
    """Least-squares diagnostics for the linear contact reconstruction."""

    coefficients: tuple[complex, ...]
    rank: int
    residual_norm: float
    condition_number: float


def fit_contact_coefficients(
    name: SectorName,
    samples: Sequence[ReconstructionSample],
    *,
    v_to_vss_reduced: VToVSSReduced | None = None,
    factorization_scale: complex = 1,
    rcond: float | None = None,
) -> ReconstructionFit:
    """Fit the regular contact polynomial after exact pole subtraction.

    Four-vector sectors may mix samples from their three tensor coefficients;
    outgoing permutation covariance requires the same contact coefficients in
    each tensor channel.
    """

    import numpy as np

    if not samples:
        raise ValueError("at least one reconstruction sample is required")
    rows: list[tuple[complex, ...]] = []
    targets: list[complex] = []
    for sample in samples:
        values = _as_outgoing(sample.outgoing_x)
        common = soft_factor(values)
        if common == 0:
            raise ZeroDivisionError("reconstruction samples must avoid soft loci")
        tensor_name = _canonical_tensor(name, sample.tensor)
        rows.append(contact_basis(name, values))
        targets.append(
            complex(sample.amplitude) / common
            - normalized_pole_part(
                name,
                values,
                tensor=tensor_name,
                v_to_vss_reduced=v_to_vss_reduced,
                factorization_scale=factorization_scale,
            )
        )
    matrix = np.asarray(rows, dtype=np.complex128)
    target = np.asarray(targets, dtype=np.complex128)
    coefficients, _residuals, rank, singular_values = np.linalg.lstsq(
        matrix, target, rcond=rcond
    )
    residual = matrix @ coefficients - target
    residual_norm = float(np.linalg.norm(residual))
    if singular_values.size == 0 or singular_values[-1] == 0:
        condition = math.inf
    else:
        condition = float(singular_values[0] / singular_values[-1])
    return ReconstructionFit(
        coefficients=tuple(complex(value) for value in coefficients),
        rank=int(rank),
        residual_norm=residual_norm,
        condition_number=condition,
    )


def screening_number(resonance_k: int) -> int:
    """Return the five-point NS screening number ``s=2k-3``."""

    if isinstance(resonance_k, bool) or not isinstance(resonance_k, int) or resonance_k < 2:
        raise ValueError("five-point resonances require integer k >= 2")
    return 2 * resonance_k - 3


def design_matrix(
    name: SectorName,
    outgoing_samples: Sequence[Sequence[complex]],
) -> tuple[tuple[complex, ...], ...]:
    """Return contact-basis rows for supplied resonance samples."""

    return tuple(contact_basis(name, sample) for sample in outgoing_samples)


__all__ = [
    "FixedIncomingSector",
    "MissingLowerPointAmplitude",
    "Pole",
    "PolePair",
    "ReconstructionFit",
    "ReconstructionSample",
    "SECTORS",
    "SectorName",
    "VToVSSReduced",
    "contact_basis",
    "denominator",
    "design_matrix",
    "elementary_symmetric",
    "fit_contact_coefficients",
    "four_point_s_to_sss_x",
    "four_point_s_to_svv_x",
    "four_point_v_to_vss_reduced_x",
    "four_point_v_to_vss_x",
    "nonzero_tensor_channel_poles",
    "normalized_ansatz",
    "normalized_channel_residue_lift",
    "normalized_double_residue_lift",
    "normalized_pole_part",
    "raw_ansatz",
    "raw_pole_part",
    "removable_tensor_channel_poles",
    "screening_number",
    "soft_factor",
    "tensor_channel_poles",
]
