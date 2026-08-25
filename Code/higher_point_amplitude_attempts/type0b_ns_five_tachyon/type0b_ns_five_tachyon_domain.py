#!/usr/bin/env python3
r"""Exploratory kinematic-domain audit for the Type-0B NS five-point sphere.

This module is matrix-model blind.  It audits arbitrary complex outgoing
frequencies subject only to ``omega_0=sum_a omega_a`` in the picture choice

``picture zero: labels 0,1,2; picture minus one: labels 3,4``.

For every one of the ten boundary divisors it checks the continuous NS
contour and all crossed endpoint structure-constant poles.  For every one of
the fifteen stable boundary corners it then chooses the better of the two
orientations and checks moving middle-trinion sum/difference residue lines,
left--middle nested residues, and right--middle nested residues.  A strict
positive minimum proves that no local polynomial or logarithmic moduli
subtraction is required in that channel atlas.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from itertools import combinations
import math
from typing import Sequence

from type0b_ns_five_tachyon import (
    BOUNDARY_CORNER_ORDERINGS,
    MINUS_ONE_LABELS,
    ODD_SECTOR_ASSIGNMENTS,
    PICTURE_ZERO_LABELS,
    CrossedNSStructurePole,
    _nested_quotient_structure_poles,
    _positive_contour_structure_poles,
    oriented_tree_orderings,
)


# Historical candidate retained as a regression point.  The corrected
# superghost threshold below proves that it is *not* subtraction free; the
# legacy name is kept temporarily for compatibility with stored pilot data.
CERTIFIED_OUTGOING_FREQUENCIES = (
    0.414 + 0.651j,
    0.414 + 0.651j,
    0.278 + 0.248j,
    0.009 + 0.327j,
)

# Historical one-real-parameter path through that rejected point.
CERTIFIED_RAY_REFERENCE_T = 0.651
CERTIFIED_RAY_COEFFICIENTS = tuple(
    value / CERTIFIED_RAY_REFERENCE_T
    for value in CERTIFIED_OUTGOING_FREQUENCIES
)

# Historical candidate for a one-corner ray.  The complete reflected and
# path-ordered quotient-contour ledger rejects it; it remains only as an
# executable regression against accidentally dropping reflected poles.
MINIMAL_SUBTRACTION_RAY_COEFFICIENTS = (
    0.090746 + 1.067168j,
    -0.270205 + 0.161166j,
    -0.180623 + 0.681918j,
    -0.000016 + 0.681805j,
)
MINIMAL_SUBTRACTION_RAY_INTERVAL = (0.982, 0.998)

# Corrected one-divisor ray found by optimizing the *complete* stable-stratum
# ledger.  Its only non-integrable record on the interval below is the
# continuous raised-pair divisor D_{12}; every endpoint, moving, and nested
# contour-residue stratum has positive radial margin.
ONE_DIVISOR_RAY_COEFFICIENTS = (
    0.0815 + 0.1284j,
    -0.3063 + 0.1280j,
    -0.2409 + 0.6785j,
    -0.5234 + 0.8075j,
)
ONE_DIVISOR_RAY_INTERVAL = (0.96, 1.00)


@dataclass(frozen=True)
class BoundaryMarginRecord:
    """One local radial-integrability inequality."""

    name: str
    kind: str
    pair: tuple[int, int]
    momentum: complex
    channel_energy: complex
    picture_zero_count: int
    threshold: int
    margin: float
    ordering: tuple[int, ...] | None = None
    sectors: tuple[int, int, int] | None = None

    def to_json(self) -> dict[str, object]:
        result = asdict(self)
        result["pair"] = list(self.pair)
        result["momentum"] = {
            "real": self.momentum.real,
            "imag": self.momentum.imag,
        }
        result["channel_energy"] = {
            "real": self.channel_energy.real,
            "imag": self.channel_energy.imag,
        }
        if self.ordering is not None:
            result["ordering"] = list(self.ordering)
        if self.sectors is not None:
            result["sectors"] = list(self.sectors)
        return result


def _finite_complex(name: str, value: complex) -> complex:
    result = complex(value)
    if not math.isfinite(result.real) or not math.isfinite(result.imag):
        raise ValueError(f"{name} must be finite")
    return result


def _nearest_wall_clearance(combination: complex, sector: int) -> float:
    """Distance of an imaginary pole combination from its nearest wall."""

    start = 1 if int(sector) == 0 else 2
    upper = max(start + 2, int(math.ceil(abs(combination.imag))) + 3)
    walls = range(start, upper + 1, 2)
    return float(min(abs(combination.imag - wall) for wall in walls))


class _PoleLedger:
    """Crossed-pole wrapper that also records chamber-boundary clearance."""

    def __init__(self) -> None:
        self.wall_clearances: list[float] = []
        self.real_endpoint_clearances: list[float] = []

    def poles(
        self,
        first: complex,
        second: complex,
        sector: int,
    ) -> tuple[CrossedNSStructurePole, ...]:
        first_value = complex(first)
        second_value = complex(second)
        candidates = [first_value + second_value]
        difference = first_value - second_value
        if difference.imag > 0.0:
            candidates.append(difference)
        elif difference.imag < 0.0:
            candidates.append(-difference)
        for combination in candidates:
            self.wall_clearances.append(
                _nearest_wall_clearance(combination, sector)
            )
            self.real_endpoint_clearances.append(abs(combination.real))
        return _positive_contour_structure_poles(first, second, sector)

    def nested_poles(
        self,
        parent: CrossedNSStructurePole,
        second: complex,
        sector: int,
    ) -> tuple[CrossedNSStructurePole, ...]:
        """Return child crossings after the parent pole's crossing time."""

        result = _nested_quotient_structure_poles(parent, second, sector)
        for pole in result:
            self.wall_clearances.append(
                min(pole.crossing_parameter, 1.0 - pole.crossing_parameter)
            )
            self.real_endpoint_clearances.append(abs(pole.momentum.real))
        return result

    @property
    def minimum_clearance(self) -> float:
        values = self.wall_clearances + self.real_endpoint_clearances
        return min(values) if values else math.inf


def _picture_threshold(pair: Sequence[int]) -> tuple[int, int]:
    count = sum(int(int(label) in PICTURE_ZERO_LABELS) for label in pair)
    selected = frozenset(int(label) for label in pair)
    # The matter OPE gives the threshold count-1.  If the colliding pair is
    # precisely the two picture-minus-one vertices, their nonchiral
    # superghost correlator contributes |z_3-z_4|^-2 and raises the threshold
    # by two units, from -1 to +1.
    threshold = 1 if selected == frozenset(MINUS_ONE_LABELS) else count - 1
    return count, threshold


def _record(
    *,
    name: str,
    kind: str,
    pair: Sequence[int],
    momentum: complex,
    channel_energy: complex,
    ordering: Sequence[int] | None = None,
    sectors: Sequence[int] | None = None,
) -> BoundaryMarginRecord:
    selected_pair = tuple(int(label) for label in pair)
    if len(selected_pair) != 2:
        raise ValueError("a boundary pair must contain two labels")
    count, threshold = _picture_threshold(selected_pair)
    p_value = complex(momentum)
    k_value = complex(channel_energy)
    margin = float((p_value * p_value - k_value * k_value).real - threshold)
    return BoundaryMarginRecord(
        name=name,
        kind=kind,
        pair=selected_pair,  # type: ignore[arg-type]
        momentum=p_value,
        channel_energy=k_value,
        picture_zero_count=count,
        threshold=threshold,
        margin=margin,
        ordering=None if ordering is None else tuple(int(x) for x in ordering),
        sectors=None if sectors is None else tuple(int(x) for x in sectors),  # type: ignore[arg-type]
    )


def _orientation_records(
    *,
    ordering: Sequence[int],
    external_momenta: Sequence[complex],
    signed_energies: Sequence[complex],
    ledger: _PoleLedger,
) -> tuple[BoundaryMarginRecord, ...]:
    selected = tuple(int(label) for label in ordering)
    if len(selected) != 5 or set(selected) != set(range(5)):
        raise ValueError("ordering must permute labels 0,...,4")
    a, b, c, d, e = selected
    left_pair = (a, b)
    right_pair = (d, e)
    left_energy = signed_energies[a] + signed_energies[b]
    right_energy = signed_energies[d] + signed_energies[e]
    records: list[BoundaryMarginRecord] = []

    for sectors in ODD_SECTOR_ASSIGNMENTS:
        sector_left, sector_middle, sector_right = sectors
        left_poles = ledger.poles(
            external_momenta[a], external_momenta[b], sector_left
        )
        right_poles = ledger.poles(
            external_momenta[d], external_momenta[e], sector_right
        )

        for left_pole in left_poles:
            for nested_pole in ledger.nested_poles(
                left_pole,
                external_momenta[c],
                sector_middle,
            ):
                records.append(
                    _record(
                        name=(
                            f"left-{left_pole.family}-m{int(left_pole.wall)}"
                            f"+middle-{nested_pole.family}-m{int(nested_pole.wall)}"
                        ),
                        kind="nested-left-middle",
                        pair=right_pair,
                        momentum=nested_pole.momentum,
                        channel_energy=right_energy,
                        ordering=selected,
                        sectors=sectors,
                    )
                )

        for right_pole in right_poles:
            for nested_pole in ledger.nested_poles(
                right_pole,
                external_momenta[c],
                sector_middle,
            ):
                records.append(
                    _record(
                        name=(
                            f"right-{right_pole.family}-m{int(right_pole.wall)}"
                            f"+middle-{nested_pole.family}-m{int(nested_pole.wall)}"
                        ),
                        kind="nested-right-middle",
                        pair=left_pair,
                        momentum=nested_pole.momentum,
                        channel_energy=left_energy,
                        ordering=selected,
                        sectors=sectors,
                    )
                )

        # For real P1>=0, the moving sum pole P2=P1+P_c-i*m enters the
        # positive quotient contour at Re(P2)=max(Re(P_c),0).  The
        # difference pole P2=P_c-P1-i*m reaches that contour only when
        # Re(P_c)>=0, for 0<=P1<=Re(P_c), and is worst at Re(P2)=0.
        start = 1 if sector_middle == 0 else 2
        wall = start
        while wall < external_momenta[c].imag - 1.0e-12:
            shifted_imaginary = external_momenta[c].imag - wall
            records.append(
                _record(
                    name=f"middle-sum-m{wall}",
                    kind="middle-line-sum",
                    pair=right_pair,
                    momentum=complex(
                        max(external_momenta[c].real, 0.0),
                        shifted_imaginary,
                    ),
                    channel_energy=right_energy,
                    ordering=selected,
                    sectors=sectors,
                )
            )
            if external_momenta[c].real >= 0.0:
                records.append(
                    _record(
                        name=f"middle-difference-m{wall}",
                        kind="middle-line-difference",
                        pair=right_pair,
                        momentum=complex(0.0, shifted_imaginary),
                        channel_energy=right_energy,
                        ordering=selected,
                        sectors=sectors,
                    )
                )
            wall += 2

    # Sector assignments duplicate moving-line inequalities.  Preserve only
    # distinct records so diagnostics stay compact and deterministic.
    unique: dict[tuple[object, ...], BoundaryMarginRecord] = {}
    for item in records:
        key = (
            item.kind,
            item.name,
            item.pair,
            item.momentum,
            item.channel_energy,
        )
        unique.setdefault(key, item)
    return tuple(unique.values())


def general_complex_energy_convergence_audit(
    outgoing_frequencies: Sequence[complex],
) -> dict[str, object]:
    r"""Audit a general momentum-conserving complex-frequency point.

    The returned ``strictly_subtraction_free`` flag requires every selected
    face/corner radial margin to be strictly positive.  For each stable
    corner the orientation with the larger worst margin is selected; this is
    the channel-adaptive atlas used by the eventual moduli integral.
    """

    if len(outgoing_frequencies) != 4:
        raise ValueError("four outgoing frequencies are required")
    outgoing = tuple(
        _finite_complex(f"outgoing_frequencies[{index}]", value)
        for index, value in enumerate(outgoing_frequencies)
    )
    if any(value.imag <= 0.0 for value in outgoing):
        raise ValueError(
            "the certified component requires positive imaginary parts for "
            "all outgoing frequencies"
        )
    incoming = sum(outgoing)
    external = (incoming, *outgoing)
    signed = (incoming, *(-value for value in outgoing))
    ledger = _PoleLedger()
    selected_records: list[BoundaryMarginRecord] = []

    for pair in combinations(range(5), 2):
        channel_energy = signed[pair[0]] + signed[pair[1]]
        selected_records.append(
            _record(
                name="continuous",
                kind="face-continuous",
                pair=pair,
                momentum=0.0j,
                channel_energy=channel_energy,
            )
        )
        for sector in (0, 1):
            for pole in ledger.poles(
                external[pair[0]], external[pair[1]], sector
            ):
                selected_records.append(
                    _record(
                        name=(
                            f"endpoint-sector{sector}-{pole.family}"
                            f"-m{int(pole.wall)}"
                        ),
                        kind="face-endpoint-residue",
                        pair=pair,
                        momentum=pole.momentum,
                        channel_energy=channel_energy,
                    )
                )

    corner_choices: list[dict[str, object]] = []
    for representative in BOUNDARY_CORNER_ORDERINGS:
        forward = tuple(representative)
        reverse = tuple(reversed(representative))
        forward_records = _orientation_records(
            ordering=forward,
            external_momenta=external,
            signed_energies=signed,
            ledger=ledger,
        )
        reverse_records = _orientation_records(
            ordering=reverse,
            external_momenta=external,
            signed_energies=signed,
            ledger=ledger,
        )
        forward_margin = min(
            (item.margin for item in forward_records), default=math.inf
        )
        reverse_margin = min(
            (item.margin for item in reverse_records), default=math.inf
        )
        if forward.index(0) > 2:
            chosen_ordering = reverse
            chosen_records = reverse_records
            chosen_margin = reverse_margin
        elif reverse.index(0) > 2:
            chosen_ordering = forward
            chosen_records = forward_records
            chosen_margin = forward_margin
        elif reverse_margin > forward_margin:
            chosen_ordering = reverse
            chosen_records = reverse_records
            chosen_margin = reverse_margin
        else:
            chosen_ordering = forward
            chosen_records = forward_records
            chosen_margin = forward_margin
        selected_records.extend(chosen_records)
        corner_choices.append(
            {
                "corner": [
                    list(representative[:2]),
                    representative[2],
                    list(representative[3:]),
                ],
                "chosen_ordering": list(chosen_ordering),
                "forward_margin": forward_margin,
                "reverse_margin": reverse_margin,
                "chosen_margin": chosen_margin,
            }
        )

    minimum_record = min(selected_records, key=lambda item: item.margin)
    mixed_wall_slacks = {
        f"{raised},{unraised}": 1.0
        - (outgoing[raised - 1].imag + outgoing[unraised - 1].imag)
        for raised in (1, 2)
        for unraised in (3, 4)
    }
    return {
        "picture_zero_labels": list(PICTURE_ZERO_LABELS),
        "outgoing_frequencies": [
            {"real": value.real, "imag": value.imag} for value in outgoing
        ],
        "incoming_frequency": {
            "real": incoming.real,
            "imag": incoming.imag,
        },
        "momentum_conservation_residual": abs(incoming - sum(outgoing)),
        "integrability_condition": (
            "Re(P_e^2-K_e^2)>tau_e, where tau_e=r_e-1 except "
            "tau_{3,4}=1 from the nonchiral superghost correlator"
        ),
        "strictly_subtraction_free": minimum_record.margin > 0.0,
        "minimum_integrability_margin": minimum_record.margin,
        "minimum_record": minimum_record.to_json(),
        "minimum_pole_chamber_clearance": ledger.minimum_clearance,
        "mixed_first_wall_slacks": mixed_wall_slacks,
        "all_mixed_pairs_below_first_C_wall": all(
            value > 0.0 for value in mixed_wall_slacks.values()
        ),
        "corner_orientation_choices": corner_choices,
        "records": [item.to_json() for item in selected_records],
    }


def three_fixed_pco_subtraction_free_no_go(
    outgoing_frequencies: Sequence[complex],
) -> dict[str, object]:
    r"""Certify the unavoidable middle-line divergence for three fixed PCOs.

    Put ``A=omega_1+omega_2`` and ``B=omega_3+omega_4``.  The continuous
    faces for the two picture-zero outgoing fields and for the two
    picture-minus-one fields both have threshold one.  If either face does
    not converge, a subtraction-free domain already fails.  If both do,
    ``Im(A)>1`` and ``Im(B)>1``.  At the stable corner
    ``(1,2)|0|(3,4)``, the incoming middle field crosses its first moving
    C-sector line.  A sum line for negative ``Re(A+B)`` or a difference line
    for positive ``Re(A+B)`` reaches

    ``P=i*(Im(A)+Im(B)-1)``.

    Comparing this pole with either endpoint-channel energy gives a margin
    strictly below ``-1``.  Hence no open subtraction-free component with
    positive outgoing imaginary parts exists for this picture assignment.
    """

    if len(outgoing_frequencies) != 4:
        raise ValueError("four outgoing frequencies are required")
    outgoing = tuple(
        _finite_complex(f"outgoing_frequencies[{index}]", value)
        for index, value in enumerate(outgoing_frequencies)
    )
    if any(value.imag <= 0.0 for value in outgoing):
        raise ValueError("all outgoing imaginary parts must be positive")
    raised_sum = outgoing[0] + outgoing[1]
    minus_one_sum = outgoing[2] + outgoing[3]
    raised_face_margin = float(-(raised_sum * raised_sum).real - 1.0)
    minus_one_face_margin = float(
        -(minus_one_sum * minus_one_sum).real - 1.0
    )
    faces_converge = raised_face_margin > 0.0 and minus_one_face_margin > 0.0
    pole_imaginary = raised_sum.imag + minus_one_sum.imag - 1.0
    pole = complex(0.0, pole_imaginary)
    middle_margin_to_raised = float(
        (pole * pole - raised_sum * raised_sum).real - 1.0
    )
    middle_margin_to_minus_one = float(
        (pole * pole - minus_one_sum * minus_one_sum).real - 1.0
    )
    if faces_converge and not (
        middle_margin_to_raised < -1.0
        and middle_margin_to_minus_one < -1.0
    ):
        raise AssertionError("the three-fixed-PCO no-go inequality failed")
    return {
        "subtraction_free_domain_exists": False,
        "picture_assignment": {
            "zero": list(PICTURE_ZERO_LABELS),
            "minus_one": list(MINUS_ONE_LABELS),
        },
        "raised_pair": [1, 2],
        "minus_one_pair": [3, 4],
        "raised_sum": {
            "real": raised_sum.real,
            "imag": raised_sum.imag,
        },
        "minus_one_sum": {
            "real": minus_one_sum.real,
            "imag": minus_one_sum.imag,
        },
        "raised_face_margin": raised_face_margin,
        "minus_one_face_margin": minus_one_face_margin,
        "both_endpoint_faces_converge": faces_converge,
        "unavoidable_middle_pole": {
            "real": pole.real,
            "imag": pole.imag,
        },
        "moving_family": (
            "difference" if (raised_sum + minus_one_sum).real >= 0.0 else "sum"
        ),
        "middle_margin_to_raised_pair": middle_margin_to_raised,
        "middle_margin_to_minus_one_pair": middle_margin_to_minus_one,
        "proof": (
            "If either endpoint face margin is nonpositive, the domain is "
            "not subtraction free. If both are positive, Im(A),Im(B)>1 "
            "and the first incoming-middle moving pole has Im(P)=Im(A)+"
            "Im(B)-1>Im(A),Im(B), so both threshold-one corner margins "
            "are strictly below -1."
        ),
    }


def certified_open_neighborhood(
    *,
    radius: float = 1.0e-6,
) -> dict[str, object]:
    r"""Audit an open ball around the historical rejected point.

    Each outgoing frequency may move independently by at most ``radius`` in
    complex modulus.  Every internal residue momentum is an affine external-
    momentum combination with coefficient l1 norm at most 9, while a channel
    energy has norm at most 3.  The displayed Lipschitz bound therefore
    controls every quadratic radial margin in the fixed pole chamber.
    """

    epsilon = float(radius)
    if not math.isfinite(epsilon) or epsilon <= 0.0:
        raise ValueError("radius must be positive and finite")
    audit = general_complex_energy_convergence_audit(
        CERTIFIED_OUTGOING_FREQUENCIES
    )
    records = audit["records"]
    max_p = max(
        abs(complex(item["momentum"]["real"], item["momentum"]["imag"]))
        for item in records
    )
    max_k = max(
        abs(
            complex(
                item["channel_energy"]["real"],
                item["channel_energy"]["imag"],
            )
        )
        for item in records
    )
    delta_p = 9.0 * epsilon
    delta_k = 3.0 * epsilon
    margin_loss_bound = (
        2.0 * max_p * delta_p
        + delta_p * delta_p
        + 2.0 * max_k * delta_k
        + delta_k * delta_k
    )
    certified_lower_margin = (
        float(audit["minimum_integrability_margin"]) - margin_loss_bound
    )
    chamber_safe = (
        13.0 * epsilon
        < 0.5 * float(audit["minimum_pole_chamber_clearance"])
    )
    positivity_safe = epsilon < min(
        min(value.real, value.imag)
        for value in CERTIFIED_OUTGOING_FREQUENCIES
    )
    return {
        "center_audit": audit,
        "independent_complex_linf_radius": epsilon,
        "residue_momentum_coefficient_l1_bound": 9,
        "channel_energy_coefficient_l1_bound": 3,
        "pole_argument_coefficient_l1_bound": 13,
        "quadratic_margin_loss_bound": margin_loss_bound,
        "certified_lower_integrability_margin": certified_lower_margin,
        "pole_chamber_stable": chamber_safe,
        "positive_frequency_component_stable": positivity_safe,
        "open_neighborhood_certified": (
            certified_lower_margin > 0.0
            and chamber_safe
            and positivity_safe
        ),
    }


def certified_ray_frequencies(t: float) -> tuple[complex, ...]:
    r"""Return frequencies on the historical, now rejected ray.

    Momentum conservation is imposed by defining the incoming frequency as
    their sum in :func:`general_complex_energy_convergence_audit`.
    """

    parameter = float(t)
    if not math.isfinite(parameter) or parameter <= 0.0:
        raise ValueError("t must be positive and finite")
    return tuple(value * parameter for value in CERTIFIED_RAY_COEFFICIENTS)


def all_c_atlas_orderings(
    outgoing_frequencies: Sequence[complex],
) -> tuple[tuple[int, ...], ...]:
    r"""Return the symmetric 120-chart all-``c`` plumbing atlas.

    The earlier 60-chart split selected one orientation of each stable tree
    from a boundary-convergence ledger.  That asymmetry was useful only for
    the old mixed ``h``/``c`` implementation.  A fixed-weight ``c`` block is
    available in both plumbing variables in every orientation, so the
    geometric split must instead minimize ``max(|q1|,|q2|)`` over all eight
    orientations of each of the fifteen labelled stable trees.  Boundary
    finite-part data remain separate from this purely geometric Voronoi
    atlas.
    """

    if len(outgoing_frequencies) != 4:
        raise ValueError("four outgoing frequencies are required")
    for index, value in enumerate(outgoing_frequencies):
        _finite_complex(f"outgoing_frequencies[{index}]", value)
    result = tuple(oriented_tree_orderings())
    if len(result) != 120 or len(set(result)) != 120:
        raise AssertionError("the all-c Mbar_0,5 atlas must have 120 charts")
    return result


def certified_ray_atlas_orderings(t: float) -> tuple[tuple[int, ...], ...]:
    r"""Obsolete wrapper for the retracted subtraction-free ray.

    Each of the fifteen stable tree topologies has two side orientations.
    The convergence audit selects the orientation with the larger worst
    residue-stratum margin.  Swapping the labels inside either cherry leaves
    that orientation and its convergence powers unchanged; retaining all
    four such swaps gives the complete importance-sampling atlas used by the
    worldsheet integral.
    """

    audit = general_complex_energy_convergence_audit(
        certified_ray_frequencies(t)
    )
    if not audit["strictly_subtraction_free"]:
        raise ValueError("t is outside the certified subtraction-free ray")
    return all_c_atlas_orderings(certified_ray_frequencies(t))


def _record_signature(item: dict[str, object]) -> tuple[object, ...]:
    ordering = item["ordering"]
    sectors = item["sectors"]
    return (
        item["kind"],
        item["name"],
        tuple(item["pair"]),  # type: ignore[arg-type]
        None if ordering is None else tuple(ordering),  # type: ignore[arg-type]
        None if sectors is None else tuple(sectors),  # type: ignore[arg-type]
    )


def is_unavoidable_three_fixed_pco_record(
    record: dict[str, object],
) -> bool:
    """Return whether ``record`` is the unique theorem-mandated corner line."""

    ordering_value = record["ordering"]
    if ordering_value is None:
        return False
    ordering = tuple(int(value) for value in ordering_value)
    topology = {
        frozenset(ordering[:2]),
        frozenset(ordering[3:]),
    }
    return (
        ordering[2] == 0
        and topology
        == {frozenset((1, 2)), frozenset((3, 4))}
        and record["kind"] in ("middle-line-sum", "middle-line-difference")
        and str(record["name"]).endswith("-m1")
    )


def minimal_subtraction_ray_frequencies(t: float) -> tuple[complex, ...]:
    """Return frequencies on the historical, now rejected one-corner ray."""

    parameter = float(t)
    if not math.isfinite(parameter) or parameter <= 0.0:
        raise ValueError("t must be positive and finite")
    return tuple(
        parameter * value for value in MINIMAL_SUBTRACTION_RAY_COEFFICIENTS
    )


def minimal_subtraction_ray_certificate() -> dict[str, object]:
    r"""Audit the historical one-corner ray and fail closed.

    Within the fixed pole chamber every stored radial margin is quadratic in
    ``t``.  Endpoint/midpoint reconstruction checks its minimum on the closed
    interval.  The complete quotient-contour ledger includes reflected
    negative-half-plane crossings and their path-ordered nested residues.
    Those terms invalidate the earlier claim that only one stratum needs a
    finite part; this function is kept as an executable retraction.
    """

    lower, upper = MINIMAL_SUBTRACTION_RAY_INTERVAL
    midpoint = 0.5 * (lower + upper)
    parameters = (lower, midpoint, upper)
    audits = tuple(
        general_complex_energy_convergence_audit(
            minimal_subtraction_ray_frequencies(parameter)
        )
        for parameter in parameters
    )
    record_maps = tuple(
        {_record_signature(item): item for item in audit["records"]}
        for audit in audits
    )
    signatures = tuple(record_maps[0])
    fixed_records = all(tuple(mapping) == signatures for mapping in record_maps)
    atlas_signatures = tuple(
        tuple(tuple(choice["chosen_ordering"]) for choice in audit["corner_orientation_choices"])
        for audit in audits
    )
    fixed_atlas = atlas_signatures[0] == atlas_signatures[1] == atlas_signatures[2]
    if not fixed_records or not fixed_atlas:
        raise AssertionError("the minimal-subtraction ray changed pole chamber")

    minimum_remainder_margin = math.inf
    quadratic_checks: list[dict[str, object]] = []
    unavoidable_signatures: list[tuple[object, ...]] = []
    for signature in signatures:
        records = tuple(mapping[signature] for mapping in record_maps)
        if is_unavoidable_three_fixed_pco_record(records[1]):
            unavoidable_signatures.append(signature)
            continue
        y0, ym, y1 = (float(record["margin"]) for record in records)
        a_coefficient = 2.0 * (y0 + y1 - 2.0 * ym)
        b_coefficient = y1 - y0 - a_coefficient
        candidates = [(0.0, y0), (1.0, y1)]
        if a_coefficient > 0.0:
            vertex = -b_coefficient / (2.0 * a_coefficient)
            if 0.0 < vertex < 1.0:
                candidates.append(
                    (
                        vertex,
                        a_coefficient * vertex * vertex
                        + b_coefficient * vertex
                        + y0,
                    )
                )
        location, minimum = min(candidates, key=lambda item: item[1])
        minimum_remainder_margin = min(minimum_remainder_margin, minimum)
        quadratic_checks.append(
            {
                "signature": signature,
                "minimum_margin": minimum,
                "minimum_fraction": location,
            }
        )

    midpoint_negative = tuple(
        record for record in audits[1]["records"] if float(record["margin"]) <= 0.0
    )
    exactly_one = (
        len(midpoint_negative) == 1
        and is_unavoidable_three_fixed_pco_record(midpoint_negative[0])
        and len(unavoidable_signatures) == 1
    )
    minimum_clearance = min(
        float(audit["minimum_pole_chamber_clearance"]) for audit in audits
    )
    certified = (
        fixed_records
        and fixed_atlas
        and exactly_one
        and minimum_remainder_margin > 0.0
        and minimum_clearance > 0.0
    )
    return {
        "minimal_subtraction_interval_certified": certified,
        "lower_endpoint": lower,
        "upper_endpoint": upper,
        "midpoint": midpoint,
        "fixed_pole_ledger_on_interval": fixed_records,
        "fixed_all_c_atlas_on_interval": fixed_atlas,
        "subtraction_stratum_count": len(midpoint_negative),
        "theorem_mandated_stratum_count": len(unavoidable_signatures),
        "unavoidable_midpoint_record": (
            midpoint_negative[0] if midpoint_negative else None
        ),
        "minimum_remainder_margin": minimum_remainder_margin,
        "minimum_pole_chamber_clearance": minimum_clearance,
        "quadratic_remainder_checks": quadratic_checks,
        "ten_sampling_parameters": [
            lower + (upper - lower) * index / 9.0 for index in range(10)
        ],
    }


def one_divisor_ray_frequencies(t: float) -> tuple[complex, ...]:
    """Return the four separated frequencies on the corrected scaling ray."""

    parameter = float(t)
    if not math.isfinite(parameter) or parameter <= 0.0:
        raise ValueError("t must be positive and finite")
    return tuple(parameter * value for value in ONE_DIVISOR_RAY_COEFFICIENTS)


def is_one_divisor_subtraction_record(record: dict[str, object]) -> bool:
    """Return whether ``record`` is the sole intended D_12 subtraction."""

    return (
        record["kind"] == "face-continuous"
        and record["name"] == "continuous"
        and tuple(int(label) for label in record["pair"]) == (1, 2)  # type: ignore[index]
    )


def one_divisor_ray_certificate() -> dict[str, object]:
    r"""Certify the corrected ray with exactly one stable-divisor finite part.

    The pole ledger and selected corner orientations are checked throughout
    the interval.  In that fixed chamber every radial margin is quadratic in
    ``t``; endpoint/midpoint reconstruction then gives its exact interval
    minimum.  The sole negative record is the continuous raised-pair divisor
    ``D_12``.  In particular no moving-line or nested residue is subtracted.
    """

    lower, upper = ONE_DIVISOR_RAY_INTERVAL
    midpoint = 0.5 * (lower + upper)
    parameters = (lower, midpoint, upper)
    audits = tuple(
        general_complex_energy_convergence_audit(
            one_divisor_ray_frequencies(parameter)
        )
        for parameter in parameters
    )
    record_maps = tuple(
        {_record_signature(item): item for item in audit["records"]}
        for audit in audits
    )
    signatures = tuple(record_maps[0])
    fixed_records = all(tuple(mapping) == signatures for mapping in record_maps)
    orientation_signatures = tuple(
        tuple(
            tuple(choice["chosen_ordering"])
            for choice in audit["corner_orientation_choices"]
        )
        for audit in audits
    )
    fixed_orientations = (
        orientation_signatures[0]
        == orientation_signatures[1]
        == orientation_signatures[2]
    )
    if not fixed_records or not fixed_orientations:
        raise AssertionError("the one-divisor ray changed pole chamber")

    target_signatures = tuple(
        signature
        for signature in signatures
        if is_one_divisor_subtraction_record(record_maps[1][signature])
    )
    target_margins = tuple(
        float(mapping[target_signatures[0]]["margin"])
        for mapping in record_maps
    )
    minimum_first_descendant_margin = min(
        value + 1.0 for value in target_margins
    )
    minimum_positive_margin = math.inf
    quadratic_checks: list[dict[str, object]] = []
    for signature in signatures:
        if signature in target_signatures:
            continue
        y0, ym, y1 = (
            float(mapping[signature]["margin"]) for mapping in record_maps
        )
        a_coefficient = 2.0 * (y0 + y1 - 2.0 * ym)
        b_coefficient = y1 - y0 - a_coefficient
        candidates = [(0.0, y0), (1.0, y1)]
        if a_coefficient > 0.0:
            vertex = -b_coefficient / (2.0 * a_coefficient)
            if 0.0 < vertex < 1.0:
                candidates.append(
                    (
                        vertex,
                        a_coefficient * vertex * vertex
                        + b_coefficient * vertex
                        + y0,
                    )
                )
        location, minimum = min(candidates, key=lambda item: item[1])
        minimum_positive_margin = min(minimum_positive_margin, minimum)
        quadratic_checks.append(
            {
                "signature": signature,
                "minimum_margin": minimum,
                "minimum_fraction": location,
            }
        )

    grid_parameters = tuple(
        lower + (upper - lower) * index / 30.0 for index in range(31)
    )
    grid_audits = tuple(
        general_complex_energy_convergence_audit(
            one_divisor_ray_frequencies(parameter)
        )
        for parameter in grid_parameters
    )
    grid_has_one_target = all(
        len(negative := [
            record for record in audit["records"]
            if float(record["margin"]) <= 0.0
        ]) == 1
        and is_one_divisor_subtraction_record(negative[0])
        for audit in grid_audits
    )
    minimum_clearance = min(
        float(audit["minimum_pole_chamber_clearance"])
        for audit in grid_audits
    )
    minimum_separation = lower * min(
        abs(
            ONE_DIVISOR_RAY_COEFFICIENTS[first]
            - ONE_DIVISOR_RAY_COEFFICIENTS[second]
        )
        for first in range(4)
        for second in range(first)
    )
    certified = (
        len(target_signatures) == 1
        and fixed_records
        and fixed_orientations
        and grid_has_one_target
        and minimum_positive_margin > 0.0
        and minimum_first_descendant_margin > 0.0
        and minimum_clearance > 0.0
        and minimum_separation > 0.0
    )
    return {
        "one_divisor_interval_certified": certified,
        "lower_endpoint": lower,
        "upper_endpoint": upper,
        "midpoint": midpoint,
        "subtraction_stratum_count": 1,
        "subtraction_record": record_maps[1][target_signatures[0]],
        "fixed_pole_ledger_on_interval": fixed_records,
        "fixed_corner_orientations_on_interval": fixed_orientations,
        "minimum_positive_remainder_margin": minimum_positive_margin,
        "minimum_first_descendant_margin": minimum_first_descendant_margin,
        "minimum_pole_chamber_clearance": minimum_clearance,
        "minimum_frequency_separation": minimum_separation,
        "quadratic_remainder_checks": quadratic_checks,
        "ten_sampling_parameters": [
            lower + (upper - lower) * index / 9.0 for index in range(10)
        ],
    }


def certified_ray_interval() -> dict[str, object]:
    r"""Reject the obsolete subtraction-free interval claim.

    The lower endpoint is where the raised--raised continuous face (1,2)
    becomes marginal.  The upper endpoint is where the wall-one moving
    C-difference line in the middle-incoming mixed/mixed corner becomes
    marginal.  Inside the fixed pole chamber every remaining radial margin
    is a quadratic polynomial in ``t``.  Reconstructing those finitely many
    quadratics from the two endpoints and their midpoint certifies their
    non-negativity on the closed interval and strict positivity away from
    the two displayed endpoint strata.
    """

    raise RuntimeError(
        "the historical ray is not subtraction free: the (3,4) superghost "
        "face has threshold one, and the three-fixed-PCO middle-line no-go "
        "excludes every positive-imaginary-frequency component"
    )

    # Historical derivation retained below only for forensic comparison.
    coefficients = CERTIFIED_RAY_COEFFICIENTS
    raised_real_ratio = coefficients[0].real
    lower = 1.0 / math.sqrt(4.0 * (1.0 - raised_real_ratio**2))

    incoming_imaginary_ratio = sum(value.imag for value in coefficients)
    right_channel_ratio = coefficients[1] + coefficients[3]
    right_channel_square_real = (right_channel_ratio**2).real
    upper_quadratic = (
        incoming_imaginary_ratio**2 + right_channel_square_real
    )
    discriminant_piece = -right_channel_square_real
    if upper_quadratic <= 0.0 or discriminant_piece <= 0.0:
        raise RuntimeError("stored ray does not have the expected upper root")
    upper = (
        incoming_imaginary_ratio + math.sqrt(discriminant_piece)
    ) / upper_quadratic
    midpoint = 0.5 * (lower + upper)

    parameters = (lower, midpoint, upper)
    audits = tuple(
        general_complex_energy_convergence_audit(
            certified_ray_frequencies(parameter)
        )
        for parameter in parameters
    )
    record_maps = tuple(
        {
            _record_signature(item): item
            for item in audit["records"]  # type: ignore[index]
        }
        for audit in audits
    )
    signatures = tuple(record_maps[0])
    fixed_atlas = all(tuple(record_map) == signatures for record_map in record_maps)

    tolerance = 2.0e-12
    minimum_closed_margin = math.inf
    minimum_nonlimiting_margin = math.inf
    limiting_signatures: list[tuple[object, ...]] = []
    quadratic_checks: list[dict[str, object]] = []
    for signature in signatures:
        y0, ym, y1 = (
            float(record_map[signature]["margin"])
            for record_map in record_maps
        )
        # q(s)=a*s^2+b*s+c for s=(t-lower)/(upper-lower).
        a_coefficient = 2.0 * (y0 + y1 - 2.0 * ym)
        b_coefficient = y1 - y0 - a_coefficient
        candidates = [(0.0, y0), (1.0, y1)]
        if a_coefficient > 0.0:
            vertex = -b_coefficient / (2.0 * a_coefficient)
            if 0.0 < vertex < 1.0:
                vertex_value = (
                    a_coefficient * vertex**2
                    + b_coefficient * vertex
                    + y0
                )
                candidates.append((vertex, vertex_value))
        minimizing_s, polynomial_minimum = min(
            candidates, key=lambda value: value[1]
        )
        minimum_closed_margin = min(minimum_closed_margin, polynomial_minimum)
        endpoint_limiter = abs(y0) <= tolerance or abs(y1) <= tolerance
        if endpoint_limiter:
            limiting_signatures.append(signature)
        else:
            minimum_nonlimiting_margin = min(
                minimum_nonlimiting_margin, polynomial_minimum
            )
        quadratic_checks.append(
            {
                "signature": list(signature),
                "normalized_quadratic": [a_coefficient, b_coefficient, y0],
                "minimum_s": minimizing_s,
                "minimum_margin": polynomial_minimum,
                "endpoint_limiter": endpoint_limiter,
            }
        )

    mixed_wall_upper = min(
        1.0 / (coefficients[raised - 1].imag + coefficients[unraised - 1].imag)
        for raised in (1, 2)
        for unraised in (3, 4)
    )
    chamber_clearance = min(
        float(audit["minimum_pole_chamber_clearance"]) for audit in audits
    )
    limiting_kinds = {
        (signature[0], signature[2]) for signature in limiting_signatures
    }
    expected_limiters = {
        ("face-continuous", (1, 2)),
        ("middle-line-difference", (2, 4)),
        ("middle-line-difference", (4, 1)),
    }
    certified = (
        lower < upper < mixed_wall_upper
        and fixed_atlas
        and chamber_clearance > 0.0
        and minimum_closed_margin >= -tolerance
        and minimum_nonlimiting_margin > 0.0
        and limiting_kinds == expected_limiters
        and bool(audits[1]["strictly_subtraction_free"])
    )
    return {
        "ray_definition": [
            {"real": value.real, "imag": value.imag}
            for value in coefficients
        ],
        "ray_parameter": "t=Im(omega_1)=Im(omega_2)",
        "incoming_frequency_is_sum": True,
        "lower_endpoint": lower,
        "lower_endpoint_equation": (
            "4*(1-(0.414/0.651)^2)*t^2-1=0"
        ),
        "lower_limiting_stratum": "continuous raised pair (1,2)",
        "upper_endpoint": upper,
        "upper_endpoint_equation": (
            "-(Im(sum(c_a))*t-1)^2-Re((c_2+c_4)^2)*t^2=0"
        ),
        "upper_limiting_stratum": (
            "middle wall-one C-difference line, mixed pair (2,4)"
        ),
        "mixed_first_wall_upper_bound": mixed_wall_upper,
        "minimum_pole_chamber_clearance": chamber_clearance,
        "minimum_closed_interval_margin": minimum_closed_margin,
        "minimum_nonlimiting_closed_interval_margin": (
            minimum_nonlimiting_margin
        ),
        "fixed_channel_atlas_on_interval": fixed_atlas,
        "limiting_record_signatures": [
            list(signature) for signature in limiting_signatures
        ],
        "quadratic_record_checks": quadratic_checks,
        "strict_open_interval_certified": certified,
        "recommended_sampling_interval": [lower + 5.0e-4, upper - 5.0e-4],
    }


__all__ = [
    "MINIMAL_SUBTRACTION_RAY_COEFFICIENTS",
    "MINIMAL_SUBTRACTION_RAY_INTERVAL",
    "ONE_DIVISOR_RAY_COEFFICIENTS",
    "ONE_DIVISOR_RAY_INTERVAL",
    "BoundaryMarginRecord",
    "CERTIFIED_OUTGOING_FREQUENCIES",
    "CERTIFIED_RAY_COEFFICIENTS",
    "CERTIFIED_RAY_REFERENCE_T",
    "all_c_atlas_orderings",
    "certified_open_neighborhood",
    "certified_ray_atlas_orderings",
    "certified_ray_frequencies",
    "certified_ray_interval",
    "general_complex_energy_convergence_audit",
    "is_unavoidable_three_fixed_pco_record",
    "minimal_subtraction_ray_certificate",
    "minimal_subtraction_ray_frequencies",
    "is_one_divisor_subtraction_record",
    "one_divisor_ray_certificate",
    "one_divisor_ray_frequencies",
    "three_fixed_pco_subtraction_free_no_go",
]
