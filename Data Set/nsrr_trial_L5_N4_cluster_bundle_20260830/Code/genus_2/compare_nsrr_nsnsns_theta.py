#!/usr/bin/env python3
r"""Historical NSRR/NSNSNS comparison and supported all-NS evaluation.

The NSRR nonchiral assembler is disabled pending a certified Ramond ground
projector and compatible free-spin conversion. The old factor-four
prescription is retired; archived data remain available for provenance.
The repaired literal chiral NSRR blocks live in nsrr_double_virasoro_block.

The historical source chart and scalar-star projection must not be reused
as a certified NSRR observable. See nsrr_human_note_geometry.py for the
re-solved NS-at-infinity chart. The target all-NS block continues to use
the direct N=1 genus-two ``c`` recursion, with its existing conventions.

The intended free ``X+psi`` denominator must use the same local plumbing
frame as each numerator. The old source theta-ratio conversion fails its
compatibility test and cannot currently supply that denominator. Its power is

    kappa(b) = c_SL(b)/(3/2) = 1 + 2 (b+1/b)^2,

which cancels the Weyl anomaly only after those frames have been matched.
"""

from __future__ import annotations

import argparse
import cmath
from dataclasses import asdict, dataclass
import json
import math
from pathlib import Path
import sys
import time
from typing import Sequence

import numpy as np
from scipy.special import roots_genlaguerre


HERE = Path(__file__).resolve().parent
CODE_ROOT = HERE.parent
for directory in (
    CODE_ROOT / "c_Recursion",
    CODE_ROOT / "full_ramond_block_runtime",
    CODE_ROOT / "genus_2_cross_channel",
    CODE_ROOT / "h_recursion",
    CODE_ROOT / "ramond_branching_recursion",
    CODE_ROOT / "double_virasoro" / "nsrr",
):
    if str(directory) not in sys.path:
        sys.path.insert(0, str(directory))

from generic_super_liouville_structure_constants import (  # noqa: E402
    GenericSuperLiouvilleConstants,
    hjs_rr_ns_constant,
)
from ns_genus2_partition import NSGenus2CRecursion  # noqa: E402
from theta_partition import (  # noqa: E402
    TYPE0B_NS_PRIMARY_PARITIES,
    theta_diagonal_sector_contribution,
)
from nsrr_double_virasoro_block import NSRRDoubleVirasoroTheta  # noqa: E402
from physical_free_plumbing_resummation import (  # noqa: E402
    physical_superfield_plumbing_partition,
)
from free_boson_plumbing import riemann_theta_constant_genus2  # noqa: E402


SOURCE_Q = (
    -3.6469074416094392e-02 + 2.9601511900343686e-02j,
    -4.0592698064739495e-02 + 2.9788087387553220e-02j,
    -3.4962790965964073e-02 - 2.5186929920020883e-02j,
)
TARGET_Q = (
    +3.4728649738917886e-02 - 2.5129048102859389e-02j,
    -2.4917675594662306e-02 - 1.4728712197212166e-03j,
    -8.0559050181984726e-02 - 7.8089260202630117e-05j,
)
SOURCE_LIFTS = (1, 1, -1)
TARGET_LIFTS = (1, -1, 1)
SOURCE_OMEGA_CHART = np.asarray(
    [[1j, 0.6 + 0.5j], [0.6 + 0.5j, 1j]], dtype=np.complex128
)
# Historical, UNVALIDATED multiplicity. A sum over full chiral Ramond
# doublets is not the irreducible nonchiral ground-state projector.
# Kept for reading old configs and explicitly labelled diagnostic code only.
RAMOND_GROUND_COMPLETENESS = 4.0


@dataclass(frozen=True)
class PartitionEstimate:
    value: float
    sector_values: tuple[float, float]
    quadrature_order: int
    block_order: int
    runtime_seconds: float
    maximum_ward_residual: float | None = None


def _primary_rule(q: complex, order: int) -> tuple[np.ndarray, np.ndarray]:
    """Gauss-Laguerre rule adapted to ``|q|^(P^2) dP/pi``."""

    log_abs = math.log(abs(complex(q)))
    if not log_abs < 0:
        raise ValueError("the continuum rule requires 0<|q|<1")
    nodes, weights = roots_genlaguerre(int(order), -0.5)
    scale = 1.0 / math.sqrt(-log_abs)
    return (
        scale * np.sqrt(nodes),
        scale * weights * np.exp(nodes) / (2 * math.pi),
    )


def _rules(q_values: Sequence[complex], order: int):
    return tuple(_primary_rule(value, order) for value in q_values)


def _measure(rules, indices: Sequence[int]) -> float:
    return float(np.prod([rules[edge][1][indices[edge]] for edge in range(3)]))


def _primary(q_values: Sequence[complex], weights: Sequence[float]) -> complex:
    return cmath.exp(
        sum(
            complex(weight) * cmath.log(complex(q))
            for weight, q in zip(weights, q_values)
        )
    )


def all_ns_node(
    *,
    b: float,
    q_values: Sequence[complex],
    lifts: Sequence[int],
    recursion_order: int,
    momenta: Sequence[float],
    measure: float,
    constants: GenericSuperLiouvilleConstants,
    recursion: NSGenus2CRecursion,
    block_method: str = "direct",
    block_working_precision: int = 70,
) -> tuple[float, float]:
    """Evaluate one weighted all-NS momentum node in Human-Note convention."""

    q_background = b + 1 / b
    central_charge = 1.5 + 3 * q_background * q_background
    weights = tuple(q_background * q_background / 8 + p * p / 2 for p in momenta)
    primary = _primary(q_values, weights)
    c_bottom, c_top = constants.ns_constants(*momenta)
    human_coefficients = (c_bottom, 1j * c_top)
    sectors = []
    for sector in (0, 1):
        structure_weight = human_coefficients[sector] ** 2
        if abs(structure_weight.imag) > 2.0e-8 * max(
            1.0, abs(structure_weight.real)
        ):
            raise ArithmeticError(
                f"all-NS structure weight is not real: {structure_weight}"
            )
        if block_method == "collision_aware_mp":
            block = recursion.collision_aware_block_mp(
                weights=weights,
                sector=sector,
                recursion_order=recursion_order,
                lifts=lifts,
                central_charge=central_charge,
                working_precision=block_working_precision,
                primary_parities=TYPE0B_NS_PRIMARY_PARITIES,
            )
        elif block_method == "direct":
            block = recursion.block(
                weights=weights,
                sector=sector,
                recursion_order=recursion_order,
                lifts=lifts,
                central_charge=central_charge,
                primary_parities=TYPE0B_NS_PRIMARY_PARITIES,
            )
        else:
            raise ValueError(
                "all-NS block_method must be 'direct' or 'collision_aware_mp'"
            )
        sectors.append(
            theta_diagonal_sector_contribution(
                sector=sector,
                measure=measure,
                structure_weight=float(structure_weight.real),
                primary_times_block=primary * block,
                primary_parities=TYPE0B_NS_PRIMARY_PARITIES,
            )
        )
    return float(sectors[0]), float(sectors[1])


def all_ns_partition(
    *,
    b: float,
    q_values: Sequence[complex],
    lifts: Sequence[int],
    recursion_order: int,
    quadrature_order: int,
    constants: GenericSuperLiouvilleConstants,
) -> PartitionEstimate:
    started = time.perf_counter()
    rules = _rules(q_values, quadrature_order)
    recursion = NSGenus2CRecursion(
        channel="theta",
        q_values=q_values,
        global_tolerance=2.0e-10,
        global_max_total_occupation=16,
        vacuum_word_length=7,
        vacuum_max_mode=50,
    )
    sectors = [0.0, 0.0]
    for indices in np.ndindex(*(quadrature_order,) * 3):
        momenta = tuple(float(rules[edge][0][indices[edge]]) for edge in range(3))
        measure = _measure(rules, indices)
        node = all_ns_node(
            b=b,
            q_values=q_values,
            lifts=lifts,
            recursion_order=recursion_order,
            momenta=momenta,
            measure=measure,
            constants=constants,
            recursion=recursion,
        )
        for sector in (0, 1):
            sectors[sector] += node[sector]
    return PartitionEstimate(
        value=float(sum(sectors)),
        sector_values=(float(sectors[0]), float(sectors[1])),
        quadrature_order=quadrature_order,
        block_order=recursion_order,
        runtime_seconds=time.perf_counter() - started,
    )


def require_certified_nsrr_partition_sewing():
    raise NotImplementedError(
        "The old NSRR partition assembler used an unproved Ramond ground "
        "multiplicity and conflated a star character with a physical spin "
        "projection. Literal chiral blocks are repaired and PBW-tested; "
        "nonchiral Ramond sewing and its marked spin-lift dictionary must "
        "be certified before producing another Q comparison. The newly "
        "introduced theta-ratio free-factor conversion also fails its "
        "all-NS compatibility check. None of these are changes to the "
        "PBW-checked double-Virasoro kernel."
    )


def nsrr_node(**kwargs):
    """Fail closed: do not relabel the historical integrand as corrected Z."""
    require_certified_nsrr_partition_sewing()


def _legacy_unvalidated_nsrr_node(
    *,
    b: float,
    q_values: Sequence[complex],
    lifts: Sequence[int],
    block_order: int,
    momenta: Sequence[float],
    measure: float,
    constants: GenericSuperLiouvilleConstants,
    branching_mp_dps: int = 0,
) -> tuple[tuple[float, float], float]:
    """Historical assembly for diagnostics ONLY; not a certified physical Z."""

    # Do not apply this obsolete contraction to the now-corrected chiral
    # series: that would neither reproduce the old run nor repair it.
    raise NotImplementedError(
        "The obsolete factor-four assembler is retired. Read archived "
        "results for provenance; do not evaluate it with the corrected blocks."
    )


def nsrr_partition(
    *,
    b: float,
    q_values: Sequence[complex],
    lifts: Sequence[int],
    block_order: int,
    quadrature_order: int,
    constants: GenericSuperLiouvilleConstants,
) -> PartitionEstimate:
    started = time.perf_counter()
    rules = _rules(q_values, quadrature_order)
    form_totals = [0.0, 0.0]
    maximum_ward_residual = 0.0
    for indices in np.ndindex(*(quadrature_order,) * 3):
        momenta = tuple(float(rules[edge][0][indices[edge]]) for edge in range(3))
        measure = _measure(rules, indices)
        node, ward_residual = nsrr_node(
            b=b,
            q_values=q_values,
            lifts=lifts,
            block_order=block_order,
            momenta=momenta,
            measure=measure,
            constants=constants,
        )
        maximum_ward_residual = max(
            maximum_ward_residual, ward_residual
        )
        for form_parity in (0, 1):
            form_totals[form_parity] += node[form_parity]
    return PartitionEstimate(
        value=float(sum(form_totals)),
        sector_values=(float(form_totals[0]), float(form_totals[1])),
        quadrature_order=quadrature_order,
        block_order=block_order,
        runtime_seconds=time.perf_counter() - started,
        maximum_ward_residual=float(maximum_ward_residual),
    )


def same_frame_free_factors(max_mode: int) -> tuple[float, float, float]:
    """Return source NRR, target NNN, and the source spin-change ratio."""

    from audit_nsrr_free_spin_conversion import require_compatible_theta_ratio
    require_compatible_theta_ratio(SOURCE_Q, SOURCE_OMEGA_CHART, max_mode=max_mode)

    source_ns_reference = physical_superfield_plumbing_partition(
        "theta", SOURCE_Q, SOURCE_LIFTS, max_mode=max_mode
    ).one_superfield_value
    target = physical_superfield_plumbing_partition(
        "theta", TARGET_Q, TARGET_LIFTS, max_mode=max_mode
    ).one_superfield_value
    theta_ratio = abs(
        riemann_theta_constant_genus2(
            SOURCE_OMEGA_CHART, ((0, 1), (1, 0)), tol=1.0e-15
        )
        / riemann_theta_constant_genus2(
            SOURCE_OMEGA_CHART, ((0, 0), (1, 0)), tol=1.0e-15
        )
    )
    return float(source_ns_reference * theta_ratio), float(target), float(theta_ratio)


def run(
    *,
    b: float,
    block_order: int,
    quadrature_order: int,
    structure_dps: int,
    free_mode: int,
    mu: complex,
    include_cosmological_prefactor: bool,
) -> dict[str, object]:
    constants = GenericSuperLiouvilleConstants(
        b,
        dps=structure_dps,
        mu=mu,
        include_cosmological_prefactor=include_cosmological_prefactor,
    )
    source_free, target_free, spin_ratio = same_frame_free_factors(free_mode)
    source = nsrr_partition(
        b=b,
        q_values=SOURCE_Q,
        lifts=SOURCE_LIFTS,
        block_order=block_order,
        quadrature_order=quadrature_order,
        constants=constants,
    )
    target = all_ns_partition(
        b=b,
        q_values=TARGET_Q,
        lifts=TARGET_LIFTS,
        recursion_order=block_order,
        quadrature_order=quadrature_order,
        constants=constants,
    )
    q_background = b + 1 / b
    central_charge = 1.5 + 3 * q_background * q_background
    kappa = central_charge / 1.5
    source_q = source.value / source_free**kappa
    target_q = target.value / target_free**kappa
    cosmological_pants = constants.cosmological_three_point_factor()
    return {
        "calculation": "generic-b NSRR theta / NSNSNS theta modular check",
        "parameters": {
            "b": b,
            "Q_background": q_background,
            "c_super_liouville": central_charge,
            "weyl_cancelling_free_superfield_power": kappa,
            "block_order": block_order,
            "quadrature_order_per_edge": quadrature_order,
            "structure_dps": structure_dps,
            "mu": [complex(mu).real, complex(mu).imag],
            "include_cosmological_prefactor": include_cosmological_prefactor,
            "one_pants_cosmological_factor": [
                cosmological_pants.real,
                cosmological_pants.imag,
            ],
        },
        "marked_charts": {
            "source": {
                "sector": "NSRR",
                "q": [[z.real, z.imag] for z in SOURCE_Q],
                "lifts": list(SOURCE_LIFTS),
                "characteristic": [[0, 1], [1, 0]],
                "period_residual": 2.344568910754979e-11,
                "integer_period_branch": [[0, 1], [1, 0]],
            },
            "target": {
                "sector": "NSNSNS",
                "q": [[z.real, z.imag] for z in TARGET_Q],
                "lifts": list(TARGET_LIFTS),
                "characteristic": [[0, 0], [0, 0]],
                "period_residual": 6.33129168259162e-11,
                "integer_period_branch": [[0, 0], [0, -1]],
            },
        },
        "free_superfield_same_local_frame": {
            "source_NSRR": source_free,
            "target_NSNSNS": target_free,
            "source_majorana_spin_change_ratio": spin_ratio,
            "mode_cutoff": free_mode,
        },
        "partitions": {
            "source_NSRR": asdict(source),
            "target_NSNSNS": asdict(target),
        },
        "Q_observable": {
            "source_NSRR": source_q,
            "target_NSNSNS": target_q,
            "ratio_source_over_target": source_q / target_q,
            "symmetric_relative_difference": 2 * (source_q - target_q) / (source_q + target_q),
        },
        "conventions": {
            "spectral_measure": "dP_NS dP_R1 dP_R2 / pi^3",
            "generic_b_leg_normalization": "Poghossian reflection-symmetric metrics; common b^(-3) per pants relative to the bare Upsilon square-root convention",
            "all_NS_three_point": "C_HN(0)=C_BRY; C_HN(1)=i*tilde_C_BRY exactly once",
            "NSRR_HJS_map": "eta=+ -> C_even; eta=- -> C_odd; overall HJS completeness 1/2",
            "NSRR_internal_R_ground_completeness": RAMOND_GROUND_COMPLETENESS,
            "source_block": "double-Virasoro branching + ordinary genus-two Virasoro c-recursion + fixed-spin auxiliary-Majorana quotient",
            "target_block": "direct N=1 genus-two c-recursion",
            "cosmological_restoration": "multiply either genus-two partition and either Q by K_b^(-Q/b) when the prefactor is omitted",
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--b", type=float, default=7 / 5)
    parser.add_argument("--block-order", type=int, default=2)
    parser.add_argument("--quadrature-order", type=int, default=2)
    parser.add_argument("--structure-dps", type=int, default=30)
    parser.add_argument("--free-mode", type=int, default=16)
    parser.add_argument("--mu", type=complex, default=1.0)
    parser.add_argument("--include-cosmological-prefactor", action="store_true")
    parser.add_argument(
        "--output",
        type=Path,
        default=HERE.parent.parent / "Data Set" / "nsrr_nsnsns_theta_pilot.json",
    )
    args = parser.parse_args()
    payload = run(
        b=args.b,
        block_order=args.block_order,
        quadrature_order=args.quadrature_order,
        structure_dps=args.structure_dps,
        free_mode=args.free_mode,
        mu=args.mu,
        include_cosmological_prefactor=args.include_cosmological_prefactor,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
